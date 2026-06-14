//===----------------------------------------------------------------------===//
//
// axis.cpp — AXIS (axi-stream struct) disagg / pop-push / side-channel passes.
//
// Carved out of VXXPrep.cpp into a focused module; the pass definitions live
// in the `hlsrs::vxx` namespace (they cross translation units, so they are not
// file-static). Compiled statically into rustc. See vxx_passes.h.
//
//===----------------------------------------------------------------------===//

#include "VXXPrep.h"
#include "VXXShared.h"
#include "vxx_internal.h"  // cross-cutting helpers (hlsrs::vxx:: namespace)
#include "vxx_passes.h"    // domain-pass declarations (hlsrs::vxx:: namespace)

#include <map>
#include <set>
#include "vxx_llvm.h"  // consolidated common LLVM headers

#include <string>

using namespace llvm;
using namespace hlsrs::vxx;

namespace hlsrs { namespace vxx {

// Build a pad-stripped clone of an AXIS element struct for use as a LOCAL
// temp alloca in axis.pop/push lowering. rustc's `#[repr(C)]` Packet emits
// a trailing `[N x i8]` alignment-pad field (8 fields total); the HLS
// backend's AXIS side-channel pass SIGSEGVs when it visits an alloca of that
// 8-field struct because the trailing array field has no corresponding AXIS
// sub-channel. The canonical `struct.hls::axis<...>` has exactly 7 fields
// (implicit tail padding, no explicit array field). We keep the first 7
// fields verbatim (so the per-channel GEP indices and the axis.pop pointer
// arg types are unchanged) and drop the trailing pad. {i32,i8×6} is still 12
// bytes (i32 forces align 4 → size rounds up), matching the 12-byte element,
// so the i96 bitcast-load of the temp still round-trips. Cached per axis
// struct via a `.nopad` suffixed name.
// Build (cached) the canonical single-level wrapper chain for one AXIS
// sub-channel: `ap_{int,uint}<W>` = { `ap_int_base<W,sign>` } =
// { `ssdm_int<W,sign>` } = { iW }. The HLS backend's AXIS solver reads each
// channel's width+sign from these nested struct type NAMES; a bare iN field
// gives it nothing → null-deref SIGSEGV. The leaf integer is the true narrow
// width (i4/i2/...) so the byte layout matches the canonical element.
StructType *getApIntWrapper(Module &M, unsigned W, bool Signed) {
  LLVMContext &Ctx = M.getContext();
  std::string Sign = Signed ? "true" : "false";
  std::string SsdmName = "struct.ssdm_int<" + std::to_string(W) + ", " + Sign + ">";
  std::string BaseName = "struct.ap_int_base<" + std::to_string(W) + ", " + Sign + ">";
  std::string TopName = (Signed ? "struct.ap_int<" : "struct.ap_uint<") +
                        std::to_string(W) + ">";
  if (StructType *T = M.getTypeByName(TopName)) return T;
  StructType *Ssdm = M.getTypeByName(SsdmName);
  if (!Ssdm) {
    // Leaf int width MUST be the real channel width (i4/i2/i1/i5/i6)
    // (`ssdm_int<2,false> = { i2 }`, etc.). Rounding sub-byte leaves to
    // i8 makes ssdm_int<4/2/1/5/6> all
    // structurally `{i8}` → the HLS backend collapses ap_uint<2/1/5/6> into a
    // single ap_uint<4> → all 6 side-channel pointers become the SAME type →
    // BasicAA/LazyValueInfo recurses across the aliasing nested-struct pointers
    // → stack-overflow SIGSEGV in CorrelatedValuePropagation (HLS 200-1715).
    // Distinct narrow leaves keep the types distinct so AA terminates.
    unsigned LeafW = W;
    Ssdm = StructType::create(Ctx, {Type::getIntNTy(Ctx, LeafW)}, SsdmName);
  }
  StructType *Base = M.getTypeByName(BaseName);
  if (!Base)
    Base = StructType::create(Ctx, {Ssdm}, BaseName);
  return StructType::create(Ctx, {Base}, TopName);
}

// Build (cached) the canonical nested AXIS element struct
//   struct.hls::axis<ap_int<W>, U, TI, TD> =
//     { ap_int<W>, ap_uint<W/8>, ap_uint<W/8>, ap_uint<U>, ap_uint<1>,
//       ap_uint<TI>, ap_uint<TD> }
// from the bare rustc element struct (i32 + i8×6 [+ pad]). Same byte layout
// (each ap_uint<≤8> occupies 1 byte), so callers can size-safe-bitcast the
// per-field GEPs back to the i8*/i32* leaf pointers the body/axis.pop expect.
// The true `ap_axis_full!` user width (e.g. 13) isn't recoverable from the Rust
// packet's i16 storage field, but the `#[top]` macro records it in the
// `__vxx_axis_w(stream, keep, strb, USER, last, id, dest)` marker (arg 3, from
// `AxisPacket::USER_BITS`). Return that width, or 0 if no axis_user packet.
static unsigned axisUserWidthFromMarker(Module &M) {
  // The width is cached into the `vxx.axis_user_w` module flag at EarlyPrep
  // (cacheAxisUserWidth), because the AXIS passes consume+erase the
  // `__vxx_axis_w` marker long before this canonicalisation runs.
  if (auto *MD = M.getModuleFlag("vxx.axis_user_w"))
    if (auto *C = mdconst::dyn_extract_or_null<ConstantInt>(MD))
      return (unsigned)C->getZExtValue();
  return 0;
}

StructType *getCppNestedAxisStruct(Module &M, StructType *AxisST) {
  // ---- axis_user (data-less side-channel packet) ----
  // `ap_axis_user!` lowers to a struct whose data/keep/strb/id/dest are
  // `AxisDisabled` and whose `user` (i16, carrying W<=16 bits) forces a
  // [1 x i8] alignment pad, so the fields are NOT the plain 7 channels
  // `ap_axis!` has. clang lowers `hls::axis_user<W, USER|LAST>` to
  // `struct.hls::axis<void, W, 0, 0, 'P', true>` = { disabled, disabled,
  // disabled, ap_uint<W>, ap_uint<1>, disabled, disabled } (7 fields, no pad).
  // Rebuild that here. Gated on the __vxx_axis_w marker's user width (non-zero
  // only for the axis_user packet) so ap_axis! streams are completely
  // unaffected. The true user width comes from the marker, not the i16 field.
  if (unsigned UW = axisUserWidthFromMarker(M)) {
    if (hlsrs::vxx::isAxisDisabledType(AxisST->getElementType(0))) {
      // Collect the 7 channels in order, skipping [K x i8] alignment pads.
      SmallVector<Type *, 8> Ch;
      for (unsigned i = 0; i < AxisST->getNumElements(); ++i) {
        Type *FT = AxisST->getElementType(i);
        if (auto *AT = dyn_cast<ArrayType>(FT))
          if (AT->getElementType()->isIntegerTy(8)) continue;  // pad
        Ch.push_back(FT);
      }
      if (Ch.size() >= 7) {
        std::string Name = "struct.hls::axis<void, " + std::to_string(UW) +
                           ", 0, 0, 'P', true>.nest";
        if (StructType *T = M.getTypeByName(Name)) return T;
        auto disabledTy = [&]() -> Type * {
          if (StructType *T =
                  M.getTypeByName("struct.hls::axis_disabled_signal"))
            return T;
          return StructType::create(M.getContext(),
                                    {Type::getInt8Ty(M.getContext())},
                                    "struct.hls::axis_disabled_signal");
        };
        bool DisUser = hlsrs::vxx::isAxisDisabledType(Ch[3]);
        bool DisLast = hlsrs::vxx::isAxisDisabledType(Ch[4]);
        SmallVector<Type *, 7> F = {
            disabledTy(),                                          // data
            disabledTy(),                                          // keep
            disabledTy(),                                          // strb
            DisUser ? disabledTy() : getApIntWrapper(M, UW, false),  // user
            DisLast ? disabledTy() : getApIntWrapper(M, 1, false),   // last
            disabledTy(),                                          // id
            disabledTy(),                                          // dest
        };
        return StructType::create(M.getContext(), F, Name);
      }
    }
  }
  auto *DataT = dyn_cast<IntegerType>(AxisST->getElementType(0));
  if (!DataT) return AxisST;
  unsigned DW = DataT->getBitWidth();
  unsigned U = 2, TI = 5, TD = 6;
  // 0-width channels: the Rust struct stores them as `AxisDisabled` (1 byte).
  // Mirror `hls::axis<T,0,0,0>`, whose nest keeps the slot but types it
  // `hls::axis_disabled_signal` — the axis_user (ap_axis_full!) shape.
  bool DisU = AxisST->getNumElements() > 3 &&
              hlsrs::vxx::isAxisDisabledType(AxisST->getElementType(3));
  bool DisI = AxisST->getNumElements() > 5 &&
              hlsrs::vxx::isAxisDisabledType(AxisST->getElementType(5));
  bool DisD = AxisST->getNumElements() > 6 &&
              hlsrs::vxx::isAxisDisabledType(AxisST->getElementType(6));
  if (DisU) U = 0;
  if (DisI) TI = 0;
  if (DisD) TD = 0;
  unsigned KW = (DW + 7) / 8;  // keep/strb width = data bytes
  std::string Name = "struct.hls::axis<ap_int<" + std::to_string(DW) + ">, " +
                     std::to_string(U) + ", " + std::to_string(TI) + ", " +
                     std::to_string(TD) + ">.nest";
  if (StructType *T = M.getTypeByName(Name)) return T;
  auto mkField = [&](unsigned W, bool Signed) -> Type * {
    return getApIntWrapper(M, W, Signed);
  };
  auto disabledTy = [&]() -> Type * {
    if (StructType *T = M.getTypeByName("struct.hls::axis_disabled_signal"))
      return T;
    return StructType::create(M.getContext(),
                              {Type::getInt8Ty(M.getContext())},
                              "struct.hls::axis_disabled_signal");
  };
  SmallVector<Type *, 7> F = {
      mkField(DW, /*Signed=*/true),             // data
      mkField(KW, false),                       // keep
      mkField(KW, false),                       // strb
      DisU ? disabledTy() : mkField(U, false),  // user
      mkField(1, false),                        // last
      DisI ? disabledTy() : mkField(TI, false), // id
      DisD ? disabledTy() : mkField(TD, false), // dest
  };
  return StructType::create(M.getContext(), F, Name);
}

StructType *getNoPadAxisStruct(Module &M, StructType *AxisST) {
  unsigned N = AxisST->getNumElements();
  if (N < 2) return AxisST;
  // Only strip when the last field is a trailing [K x i8] pad array.
  auto *Last = dyn_cast<ArrayType>(AxisST->getElementType(N - 1));
  if (!Last || !Last->getElementType()->isIntegerTy(8)) return AxisST;
  // Canonical nested element type (the solver requires the nested ssdm_int<W>
  // wrappers to read channel width+sign; bare i8/i32 → SIGSEGV). Same byte
  // layout, so callers size-safe-bitcast the per-field GEPs back to i8*/i32*.
  return getCppNestedAxisStruct(M, AxisST);
}

// Convert EVERY alloca of a padded AXIS element struct (8 fields, trailing
// [N x i8] pad) into a pad-stripped 7-field alloca. rustc emits such allocas
// not only for our axis.pop/push temps but also for `Packet { .. }` struct
// literals (e.g. the value passed to `B.write(..)`); the HLS backend's AXIS
// solver SIGSEGVs on the trailing array field regardless of which alloca it is.
// The first 7 fields are identical, so per-field GEP result pointer types
// (i32*/i8*) are unchanged and the axis.pop/push calls stay valid; whole-struct
// bitcasts (to i8*/i96* for lifetime/load/store) also work since {i32,i8×6} is
// still 12 bytes. Gated by `__vxx_axis_packed`.
bool convertPaddedAxisAllocasToNoPad(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) return false;
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<AllocaInst *, 8> Targets;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          auto *ST = dyn_cast<StructType>(AI->getAllocatedType());
          if (!ST || !ST->hasName()) continue;
          if (!ST->getName().startswith("struct.hls::axis<")) continue;
          if (ST->getName().endswith(".nopad")) continue;
          unsigned N = ST->getNumElements();
          if (N < 2) continue;
          auto *Last = dyn_cast<ArrayType>(ST->getElementType(N - 1));
          if (Last && Last->getElementType()->isIntegerTy(8))
            Targets.push_back(AI);
        }
    for (AllocaInst *AI : Targets) {
      auto *OldST = cast<StructType>(AI->getAllocatedType());
      StructType *NewST = getNoPadAxisStruct(M, OldST);
      if (NewST == OldST) continue;
      IRBuilder<> EB(AI);
      AllocaInst *NewAI =
          EB.CreateAlloca(NewST, AI->getArraySize(), AI->getName() + ".np");
      NewAI->setAlignment(Align(AI->getAlignment()));
      SmallVector<Use *, 16> Uses;
      for (Use &U : AI->uses()) Uses.push_back(&U);
      for (Use *U : Uses) {
        User *Usr = U->getUser();
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
          if (GEP->getSourceElementType() == OldST) {
            IRBuilder<> GB(GEP);
            SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
            Value *NG = GB.CreateInBoundsGEP(NewST, NewAI, Idx);
            // NewST is the nested type, so the per-field GEP yields a
            // struct pointer (ap_int<W>*/ap_uint<W>*); the body expects the
            // original leaf pointer (i32*/i8*). Size-safe-bitcast back.
            if (NG->getType() != GEP->getType())
              NG = GB.CreateBitCast(NG, GEP->getType());
            GEP->replaceAllUsesWith(NG);
            GEP->eraseFromParent();
            Changed = true;
            continue;
          }
        }
        if (auto *BC = dyn_cast<BitCastInst>(Usr)) {
          IRBuilder<> BB2(BC);
          Value *NBC = BB2.CreateBitCast(NewAI, BC->getDestTy());
          BC->replaceAllUsesWith(NBC);
          BC->eraseFromParent();
          Changed = true;
          continue;
        }
        // Generic user (call/store/load of the whole struct ptr): repoint
        // through a NewST*→OldST* bitcast (same 12-byte size).
        IRBuilder<> UB(cast<Instruction>(Usr));
        U->set(UB.CreateBitCast(NewAI, AI->getType()));
        Changed = true;
      }
      AI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

bool collapseScalarCopyBridge(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    return false;
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!fnHasAxisRMWBody(F)) continue;
    SmallVector<StoreInst *, 4> Stores;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *SI = dyn_cast<StoreInst>(&I))
          Stores.push_back(SI);
    for (StoreInst *SI : Stores) {
      auto *LI = dyn_cast<LoadInst>(SI->getValueOperand());
      if (!LI || !LI->getType()->isIntegerTy()) continue;
      auto *Q = dyn_cast<AllocaInst>(SI->getPointerOperand());
      if (!Q) continue;
      // Q must be an iN-typed alloca written ONLY by this store (other uses
      // are reads / bitcasts / lifetime / GEP).
      if (Q->getAllocatedType() != LI->getType()) continue;
      bool SingleStore = true;
      for (User *U : Q->users()) {
        if (U == SI) continue;
        if (auto *OtherSI = dyn_cast<StoreInst>(U))
          if (OtherSI->getPointerOperand() == Q) { SingleStore = false; break; }
      }
      if (!SingleStore) continue;
      Value *P = LI->getPointerOperand();  // iN* into the axis temp
      if (P->getType() != Q->getType()) continue;
      Q->replaceAllUsesWith(P);
      SI->eraseFromParent();
      if (LI->use_empty()) LI->eraseFromParent();
      Q->eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: collapseScalarCopyBridge collapsed axis i96 bridge\n";
  return Changed;
}

// Scalarize a wide integer load (`load iN`, N>8) whose only uses are byte-
// aligned extractions (`trunc iN to iM`, or `lshr iN, C` then `trunc`), into
// field-wise narrow loads at the corresponding byte offsets. rustc's SROA packs
// the AXIS side channels into an i64 (`load i64` from the axis.pop dest temp,
// then trunc/lshr per channel); the HLS backend's
// CorrelatedValuePropagation/LazyValueInfo SIGSEGVs on that wide-int chain
// (the desired form is field-wise — 0 trunc/lshr). The temp is written by the
// opaque axis.pop so SROA/InstCombine can't split it; we do it here. Gated by
// `__vxx_axis_packed`.
bool scalarizeAxisWideLoads(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    return false;
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!fnHasAxisRMWBody(F)) continue;
    SmallVector<LoadInst *, 8> Wide;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (auto *IT = dyn_cast<IntegerType>(LI->getType()))
            if (IT->getBitWidth() > 8 && (IT->getBitWidth() % 8) == 0)
              Wide.push_back(LI);
    for (LoadInst *LI : Wide) {
      // Each use must be `trunc LI to iM` (offset 0) or `trunc (lshr LI, C)`
      // with C a byte multiple. Collect (offsetBytes, truncInst).
      SmallVector<std::pair<uint64_t, Instruction *>, 8> Extracts;
      bool Ok = true;
      for (User *U : LI->users()) {
        if (auto *TR = dyn_cast<TruncInst>(U)) {
          Extracts.push_back({0, TR});
        } else if (auto *BO = dyn_cast<BinaryOperator>(U)) {
          if (BO->getOpcode() != Instruction::LShr) { Ok = false; break; }
          auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (!CI || (CI->getZExtValue() % 8) != 0) { Ok = false; break; }
          uint64_t off = CI->getZExtValue() / 8;
          // lshr must feed only trunc(s).
          for (User *LU : BO->users()) {
            auto *TR = dyn_cast<TruncInst>(LU);
            if (!TR) { Ok = false; break; }
            Extracts.push_back({off, TR});
          }
          if (!Ok) break;
        } else { Ok = false; break; }
      }
      if (!Ok || Extracts.empty()) continue;
      // Trace the wide load's pointer back to a nest-struct alloca + byte
      // offset, so we can emit TYPED nest-field GEPs (GEP nest,0,fieldIdx,...
      // → leaf) instead of raw i8 byte arithmetic — the HLS backend rejects
      // pointer address-computation on its disaggregated scalar ports (HLS
      // 214-323), but accepts direct typed field accesses (get_*_ptr).
      const DataLayout &DLY = M.getDataLayout();
      Value *Cur = LI->getPointerOperand();
      uint64_t BaseOff = 0;
      AllocaInst *Root = nullptr;
      while (true) {
        if (auto *BC = dyn_cast<BitCastInst>(Cur)) { Cur = BC->getOperand(0); continue; }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
          APInt Off(64, 0);
          if (!GEP->accumulateConstantOffset(DLY, Off)) break;
          BaseOff += Off.getZExtValue();
          Cur = GEP->getPointerOperand();
          continue;
        }
        if (auto *AI = dyn_cast<AllocaInst>(Cur)) Root = AI;
        break;
      }
      auto *NestST = Root ? dyn_cast<StructType>(Root->getAllocatedType()) : nullptr;
      if (!NestST) continue;
      const StructLayout *SL = DLY.getStructLayout(NestST);
      bool AllMapped = true;
      SmallVector<std::pair<Instruction *, Value *>, 8> Repl;
      for (auto &E : Extracts) {
        Instruction *TR = E.second;
        uint64_t AbsOff = BaseOff + E.first;
        if (AbsOff >= SL->getSizeInBytes()) { AllMapped = false; break; }
        unsigned FieldIdx = SL->getElementContainingOffset(AbsOff);
        IRBuilder<> TB(TR);
        // Build GEP indices descending through the nested struct chain
        // (ap_X<W> → ap_int_base → ssdm_int → leaf int) to the leaf at AbsOff.
        SmallVector<Value *, 6> Idx;
        Idx.push_back(TB.getInt32(0));
        Type *CurTy = NestST;
        uint64_t Rem = AbsOff;
        bool Bad = false;
        while (auto *ST = dyn_cast<StructType>(CurTy)) {
          const StructLayout *L = DLY.getStructLayout(ST);
          unsigned Fi = L->getElementContainingOffset(Rem);
          Idx.push_back(TB.getInt32(Fi));
          Rem -= L->getElementOffset(Fi);
          CurTy = ST->getElementType(Fi);
        }
        if (Bad || !CurTy->isIntegerTy()) { AllMapped = false; break; }
        Value *LeafPtr = TB.CreateInBoundsGEP(NestST, Root, Idx);
        Value *NV = TB.CreateLoad(CurTy, LeafPtr);
        Type *DstTy = TR->getType();
        if (NV->getType() != DstTy) {
          if (NV->getType()->getIntegerBitWidth() > DstTy->getIntegerBitWidth())
            NV = TB.CreateTrunc(NV, DstTy);
          else
            NV = TB.CreateZExt(NV, DstTy);
        }
        Repl.push_back({TR, NV});
      }
      if (!AllMapped) continue;
      for (auto &R : Repl) R.first->replaceAllUsesWith(R.second);
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: scalarizeAxisWideLoads split wide axis load(s)\n";
  return Changed;
}

// Run SROA + InstCombine + EarlyCSE + DCE over functions that contain AXIS
// pop/push intrinsics, to eliminate rustc's i96 struct round-trip (the
// `load i96` from the read-dest → split to i32/i64 → +5 → repack to i96 →
// trunc/lshr into fields chain). The HLS backend's CorrelatedValuePropagation /
// LazyValueInfo pass SIGSEGVs on that wide-int def-use chain (the desired form
// has no i96 — it uses field-wise ap_int ops). InstCombine folds
// `trunc/lshr(or(shl(zext rest,32), zext data))` back into the original field
// values, so the i96 alloca-bridge + pack/unpack DCEs away and the body
// becomes field-wise. Gated by `__vxx_axis_packed` (only AXIS kernels).
// rustc passes the modified Packet by value to write() via an `alloca iN` (N>=64)
// copy: `store iN <pack>, %a ; ... ; load iN %a` (the load feeds the unpack →
// axis.push). The alloca also has an `i8*` bitcast + lifetime markers, which make
// SROA treat it as address-taken → it's NOT promoted → InstCombine can't see
// through `load iN %a` to fold the pack/unpack → the i96 wide-int chain survives →
// downstream CVP SIGSEGV. Promote these straight-line single-store wide copies
// (RAUW loads with the stored value) so the subsequent InstCombine folds the i96
// away. Safe only for straight-line code: store must precede every load in the
// SAME BB.
bool promoteWideCopyAllocas(Function &F) {
  bool Changed = false;
  SmallVector<AllocaInst *, 8> Cands;
  for (Instruction &I : F.getEntryBlock())
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (auto *IT = dyn_cast<IntegerType>(AI->getAllocatedType()))
        if (IT->getBitWidth() >= 64)
          Cands.push_back(AI);
  for (AllocaInst *AI : Cands) {
    StoreInst *TheStore = nullptr;
    SmallVector<LoadInst *, 4> Loads;
    SmallVector<Instruction *, 4> Lifetimes;  // bitcast + lifetime to erase
    bool Ok = true;
    // Collect direct users + users of an i8* bitcast of the alloca.
    SmallVector<User *, 8> Users(AI->users().begin(), AI->users().end());
    for (User *U : Users) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() != AI || SI->getValueOperand() == AI) { Ok = false; break; }
        if (TheStore) { Ok = false; break; }  // require exactly one store
        TheStore = SI;
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (LI->getType() != AI->getAllocatedType()) { Ok = false; break; }
        Loads.push_back(LI);
      } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
        // Only an i8* bitcast feeding lifetime intrinsics is allowed.
        for (User *BU : BC->users()) {
          auto *CI = dyn_cast<CallInst>(BU);
          if (!CI || !CI->getCalledFunction() ||
              !CI->getCalledFunction()->getName().startswith("llvm.lifetime")) { Ok = false; break; }
          Lifetimes.push_back(CI);
        }
        if (!Ok) break;
        Lifetimes.push_back(BC);
      } else { Ok = false; break; }
    }
    if (!Ok || !TheStore || Loads.empty()) continue;
    // Straight-line safety: store and all loads in one BB, store before each load.
    BasicBlock *BB = TheStore->getParent();
    bool SameBB = true;
    for (LoadInst *LI : Loads) if (LI->getParent() != BB) { SameBB = false; break; }
    if (!SameBB) continue;
    bool OrderOk = true;
    for (LoadInst *LI : Loads) if (!TheStore->comesBefore(LI)) { OrderOk = false; break; }
    if (!OrderOk) continue;
    Value *V = TheStore->getValueOperand();
    for (LoadInst *LI : Loads) { LI->replaceAllUsesWith(V); LI->eraseFromParent(); }
    TheStore->eraseFromParent();
    for (Instruction *I : Lifetimes) if (I->use_empty()) I->eraseFromParent();
    if (AI->use_empty()) AI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// General: forward a small per-field COPY-BRIDGE alloca that survives SROA when a
// value is widened on store but read back narrower. rustc's by-value AXIS push
// copies each side channel as `%z = zext iN %orig to i8; store i8 %z, i8* %p;
// … = load iN, iN* bitcast(%p)`. SROA leaves the `alloca i8` because the store
// width (i8) != the load width (iN<8), so it can't prove store-to-load forwarding;
// InstCombine likewise bails through the bitcast. The bridge is a pure identity
// (the load reads the low N bits = %orig). We forward each load to the store's
// value (peeling a zext so the result is the original narrow SSA value), leaving
// the alloca/store/zext dead for DCE. Eliminating these flattens the body
// to the field-wise shape so the HLS backend's CorrelatedValuePropagation stops
// SIGSEGVing on the residual sub-byte (i1/i2/i4/i5/i6) pack chain. Structural,
// straight-line only (store precedes every load in one BB); no field hardcoding.
bool forwardNarrowCopyBridge(Function &F) {
  bool Changed = false;
  SmallVector<AllocaInst *, 8> Cands;
  for (Instruction &I : F.getEntryBlock())
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (auto *IT = dyn_cast<IntegerType>(AI->getAllocatedType()))
        if (IT->getBitWidth() <= 32) // small per-field copy buffers only
          Cands.push_back(AI);
  for (AllocaInst *AI : Cands) {
    auto *AllocTy = cast<IntegerType>(AI->getAllocatedType());
    StoreInst *TheStore = nullptr;
    SmallVector<LoadInst *, 4> Loads;
    SmallVector<Instruction *, 4> Casts; // bitcasts + lifetime calls to clean up
    SmallVector<Value *, 8> Work{AI};
    SmallPtrSet<Value *, 8> Seen;
    bool Ok = true;
    while (!Work.empty() && Ok) {
      Value *V = Work.pop_back_val();
      for (User *U : V->users()) {
        if (!Seen.insert(U).second)
          continue;
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getValueOperand() == V) { Ok = false; break; } // alloca escapes
          if (TheStore) { Ok = false; break; }                   // require one store
          if (SI->getValueOperand()->getType() != AllocTy) { Ok = false; break; }
          TheStore = SI;
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          auto *LT = dyn_cast<IntegerType>(LI->getType());
          if (!LT || LT->getBitWidth() > AllocTy->getBitWidth()) { Ok = false; break; }
          Loads.push_back(LI);
        } else if (isa<BitCastInst>(U)) {
          Casts.push_back(cast<Instruction>(U));
          Work.push_back(U);
        } else if (auto *CI = dyn_cast<CallInst>(U)) {
          if (!CI->getCalledFunction() ||
              !CI->getCalledFunction()->getName().startswith("llvm.lifetime")) {
            Ok = false; break;
          }
          Casts.push_back(CI);
        } else { Ok = false; break; }
      }
    }
    if (!Ok || !TheStore || Loads.empty())
      continue;
    // Straight-line safety: store + all loads in one BB, store before each load.
    BasicBlock *BB = TheStore->getParent();
    bool SameBB = true;
    for (LoadInst *LI : Loads)
      if (LI->getParent() != BB) { SameBB = false; break; }
    if (!SameBB)
      continue;
    bool OrderOk = true;
    for (LoadInst *LI : Loads)
      if (!TheStore->comesBefore(LI)) { OrderOk = false; break; }
    if (!OrderOk)
      continue;
    Value *SV = TheStore->getValueOperand();
    Value *Base = SV;
    if (auto *ZE = dyn_cast<ZExtInst>(SV)) // peel the widening zext
      Base = ZE->getOperand(0);
    unsigned bw = Base->getType()->getIntegerBitWidth();
    for (LoadInst *LI : Loads) {
      unsigned lw = LI->getType()->getIntegerBitWidth();
      IRBuilder<> B(LI);
      Value *NV = lw == bw ? Base
                : lw < bw  ? B.CreateTrunc(Base, LI->getType())
                           : B.CreateZExt(Base, LI->getType());
      LI->replaceAllUsesWith(NV);
    }
    for (LoadInst *LI : Loads)
      LI->eraseFromParent();
    TheStore->eraseFromParent();
    for (Instruction *C : Casts)
      if (C->use_empty())
        C->eraseFromParent();
    if (AI->use_empty())
      AI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// General: scalar-replace an integer alloca that rustc uses to REPACK a by-value
// aggregate (e.g. the AXIS side channels keep/strb/user/last/id/dest packed into
// an `alloca i64`). It is written ONLY by typed stores at distinct constant byte
// offsets and read ONLY by full-width `load iN` whose uses are `trunc`(byte 0) or
// `lshr C`(byte C/8) + `trunc`. Each extraction is forwarded to the store value at
// the matching byte offset (zext/trunc to the extraction width — the consumer
// re-truncates to the channel width, so the extra bits are don't-care). This kills
// the wide-int load that the HLS backend's CorrelatedValuePropagation SIGSEGVs on;
// SROA bails because the stores reach the alloca through `i8*` raw-offset
// GEP/bitcasts.
// Purely structural — no example/field-count/value hardcoding.
bool scalarizeBytePackedIntAlloca(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  bool Changed = false;
  SmallVector<AllocaInst *, 4> Cands;
  for (Instruction &I : F.getEntryBlock())
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (auto *IT = dyn_cast<IntegerType>(AI->getAllocatedType()))
        if (IT->getBitWidth() >= 16 && (IT->getBitWidth() % 8) == 0)
          Cands.push_back(AI);
  for (AllocaInst *AI : Cands) {
    auto *AllocTy = cast<IntegerType>(AI->getAllocatedType());
    auto traceToAI = [&](Value *P, uint64_t &OffOut) -> bool {
      uint64_t Off = 0; Value *Cur = P;
      while (true) {
        if (auto *BC = dyn_cast<BitCastInst>(Cur)) { Cur = BC->getOperand(0); continue; }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
          APInt A(64, 0);
          if (!GEP->accumulateConstantOffset(DL, A)) return false;
          Off += A.getZExtValue(); Cur = GEP->getPointerOperand(); continue;
        }
        break;
      }
      if (Cur != AI) return false;
      OffOut = Off; return true;
    };
    // Gather stores (byte-offset -> value) + full-width loads over all transitive
    // pointer users; any other use disqualifies the alloca.
    SmallVector<std::pair<uint64_t, Value *>, 8> Stores;
    SmallVector<LoadInst *, 4> Loads;
    SmallVector<Value *, 16> Work{AI};
    SmallPtrSet<Value *, 16> Seen;
    bool Ok = true;
    while (!Work.empty() && Ok) {
      Value *V = Work.pop_back_val();
      for (User *U : V->users()) {
        if (!Seen.insert(U).second) continue;
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getValueOperand() == V) { Ok = false; break; }
          uint64_t off;
          if (!traceToAI(SI->getPointerOperand(), off)) { Ok = false; break; }
          Stores.push_back({off, SI->getValueOperand()});
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          uint64_t off;
          if (LI->getType() != AllocTy || !traceToAI(LI->getPointerOperand(), off) || off) { Ok = false; break; }
          Loads.push_back(LI);
        } else if (isa<BitCastInst>(U) || isa<GetElementPtrInst>(U)) {
          Work.push_back(U);
        } else if (auto *CI = dyn_cast<CallInst>(U)) {
          if (!CI->getCalledFunction() || !CI->getCalledFunction()->getName().startswith("llvm.lifetime")) { Ok = false; break; }
        } else { Ok = false; break; }
      }
    }
    if (!Ok || Loads.empty() || Stores.empty()) continue;
    auto findStore = [&](uint64_t off) -> Value * {
      for (auto &S : Stores) if (S.first == off) return S.second;
      return nullptr;
    };
    // Validate + collect forwards: every load use is trunc(0) or lshr(C)+trunc.
    SmallVector<std::pair<Instruction *, uint64_t>, 16> Extracts; // (truncInst, byteOff)
    bool AllOk = true;
    for (LoadInst *LI : Loads) {
      for (User *U : LI->users()) {
        if (isa<TruncInst>(U)) { Extracts.push_back({cast<Instruction>(U), 0}); continue; }
        auto *BO = dyn_cast<BinaryOperator>(U);
        if (!BO || BO->getOpcode() != Instruction::LShr) { AllOk = false; break; }
        auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
        if (!C || (C->getZExtValue() % 8)) { AllOk = false; break; }
        uint64_t off = C->getZExtValue() / 8;
        for (User *LU : BO->users()) {
          if (!isa<TruncInst>(LU)) { AllOk = false; break; }
          Extracts.push_back({cast<Instruction>(LU), off});
        }
        if (!AllOk) break;
      }
      if (!AllOk) break;
    }
    if (!AllOk || Extracts.empty()) continue;
    // Require a store for every extraction offset before mutating.
    for (auto &E : Extracts) if (!findStore(E.second)) { AllOk = false; break; }
    if (!AllOk) continue;
    for (auto &E : Extracts) {
      Instruction *TR = E.first;
      Value *SV = findStore(E.second);
      IRBuilder<> B(TR);
      unsigned dw = TR->getType()->getIntegerBitWidth();
      unsigned sw = SV->getType()->getIntegerBitWidth();
      Value *NV = sw > dw ? B.CreateTrunc(SV, TR->getType())
                : sw < dw ? B.CreateZExt(SV, TR->getType()) : SV;
      TR->replaceAllUsesWith(NV);
    }
    Changed = true; // dead load/store/alloca left for DCE
  }
  return Changed;
}

bool simplifyAxisI96Body(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) return false;
  SmallPtrSet<Function *, 4> AxisFns;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction()) {
            StringRef CN = Callee->getName();
            // Match both the helper bodies (axis.pop/push) AND their callers
            // (e.g. @example calls hlsaxis_read/write) — the i96 pack/unpack
            // chain lives in the CALLER, so it must be cleaned there too.
            if (CN.startswith("llvm.fpga.axis.") || CN.startswith("hlsaxis_"))
              AxisFns.insert(&F);
          }
  }
  if (AxisFns.empty()) return false;
  // Promote rustc's by-value wide-int copy allocas first so InstCombine (below)
  // can fold the i96 pack/unpack into field-wise values.
  for (Function *F : AxisFns) promoteWideCopyAllocas(*F);
  legacy::FunctionPassManager FPM(&M);
  FPM.add(createSROAPass());
  FPM.add(createEarlyCSEPass());
  // InstCombine runs after SROA+EarlyCSE (and the wide-copy allocas promoted
  // above), so it folds the i96 pack/unpack into field-wise values without
  // canonicalizing the axis.pop/push struct-ptr field GEPs into wrong
  // byte-offset bitcasts (data/keep would otherwise alias at offset 0 → field
  // miscompile + LVI/CVP SIGSEGV).
  FPM.add(createInstructionCombiningPass());
  FPM.add(createAggressiveDCEPass());
  FPM.doInitialization();
  bool Changed = false;
  for (Function *F : AxisFns) Changed |= FPM.run(*F);
  FPM.doFinalization();
  if (Changed)
    vxxDbg() << "vxx: simplifyAxisI96Body ran on " << AxisFns.size()
           << " axis fn(s)\n";
  return Changed;
}


// Remove any _ssdm_op_SpecInterface(arg, ...) emitted on a kernel arg that ALSO
// carries an xlx_axis op-bundle. An AXIS array-pointer arg must carry ONLY the
// xlx_axis bundle; a leftover SpecInterface (typically ap_memory from
// injectDefaultApAutoSpec's array-ptr default, which runs before the axis
// lowering and fires on pure-loop AXIS kernels) makes HLS see two interface
// modes → XFORM 203-801 "Stream port has invalid interface mode 'ap_memory'".
// Order-independent: runs after both injectDefaultApAutoSpec and
// injectArrayAxisSpec.
bool stripSpecInterfaceOnAxisArgs(Module &M) {
  Function *SIFn = M.getFunction("_ssdm_op_SpecInterface");
  Function *Side = M.getFunction("llvm.sideeffect");
  if (!SIFn || !Side) return false;
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.hasFnAttribute("fpga.top.func")) continue;
    DenseSet<unsigned> AxisArgs;
    for (User *U : Side->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getFunction() != &F) continue;
      for (unsigned BI = 0; BI < CI->getNumOperandBundles(); ++BI) {
        auto OB = CI->getOperandBundleAt(BI);
        if (OB.getTagName() != "xlx_axis") continue;
        if (OB.Inputs.empty()) continue;
        if (Argument *A = resolveMarkerArg(OB.Inputs[0].get()))
          if (A->getParent() == &F) AxisArgs.insert(A->getArgNo());
      }
    }
    if (AxisArgs.empty()) continue;
    SmallVector<CallInst*, 4> ToErase;
    for (User *U : SIFn->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getFunction() != &F || CI->arg_size() < 1) continue;
      Argument *A = resolveMarkerArg(CI->getArgOperand(0));
      if (A && A->getParent() == &F && AxisArgs.count(A->getArgNo()))
        ToErase.push_back(CI);
    }
    for (CallInst *CI : ToErase) { CI->eraseFromParent(); Changed = true; }
  }
  return Changed;
}

// Disaggregate the kernel signature for top fns with arg type
// `%"class.hls::stream<hls::axis<T>>"*`. Each AXIS struct arg is replaced
// by 7 sub-channel ptr args (data: T*, others: i8*) named
// `<oldname>_V_{data,keep,strb,user,last,id,dest}_V`. Body uses are
// remapped via a temp alloca that's loaded from / stored to the new args.
//
// Without sig disagg, HLS 200-92 rejects the kernel: a struct ptr arg used
// with GEP+load/store is seen as "both read and write" on the AXIS port.
// The disagg produces 14 sub-channel ptr args directly.
bool disaggAxisStructKernelSig(Module &M) {
  // `__vxx_axis_packed` experiment: skip the AXIS sig disagg so the packed
  // `class.hls::stream<hls::axis<...>>*` arg reaches the HLS backend. The
  // backend itself disaggregates into 14 ch ports AND records stream metadata
  // so cosim TB reconstructs a 2-arg stream hw_stub (needed for cosim N
  // transactions). Up-front disagg matches the synthesized interface but breaks
  // cosim (cosim TB sees 14 scalar ports → 1 transaction → 212-359).
  if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);

  static const char *ChanSuffixes[7] = {
    "_V_data_V", "_V_keep_V", "_V_strb_V",
    "_V_user_V", "_V_last_V", "_V_id_V", "_V_dest_V"
  };

  // Collect AXIS sub-channel widths from `__vxx_axis_w(stream, keep, strb,
  // user, last, id, dest)` markers, keyed by the stream Argument*. Lets us
  // emit the per-channel ports at the exact ap_uint<N> widths (the cosim TB
  // generator recognises the AXIS stream by these widths). Channel order:
  // [data, keep, strb, user, last, id, dest]; data width stays from the
  // struct (i32). Absent marker → fall back to the struct field widths (i8).
  // Collected BEFORE the Targets scan: the marker also serves as the AXIS
  // gate for stream classes whose element kept its Rust name (e.g.
  // `class.hls::stream<Packet>` for data-disabled hls::axis_user packets —
  // detect/rename bail on those, but disagg can still split them).
  DenseMap<Argument *, std::array<unsigned, 7>> AxisChanW;
  if (Function *WM = M.getFunction("__vxx_axis_w")) {
    for (User *U : WM->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() != 7) continue;
      Argument *A = resolveMarkerArg(CI->getArgOperand(0));
      if (!A) continue;
      std::array<unsigned, 7> W = {0, 0, 0, 0, 0, 0, 0};
      bool ok = true;
      for (unsigned i = 1; i < 7; ++i) {
        if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(i)))
          W[i] = (unsigned)C->getZExtValue();
        else { ok = false; break; }
      }
      if (ok) AxisChanW[A] = W;  // W[0]=data placeholder (filled from struct)
    }
  }

  SmallVector<Function *, 4> Targets;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("fpga.top.func")) continue;
    for (Argument &A : F.args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT) continue;
      auto *ST = dyn_cast<StructType>(PT->getElementType());
      if (!ST || !ST->hasName()) continue;
      if (ST->getName().startswith("class.hls::stream<hls::axis<") ||
          (ST->getName().startswith("class.hls::stream<") &&
           AxisChanW.count(&A))) {
        Targets.push_back(&F);
        break;
      }
    }
  }
  if (Targets.empty()) return false;

  bool Changed = false;
  for (Function *Old : Targets) {
    struct ArgInfo {
      Argument *Old;
      bool IsAxis;
      StructType *StreamST;
      StructType *AxisST;
      SmallVector<Type *, 7> ChanPtrTys;
      SmallVector<std::string, 7> ChanNames;
      // Struct field index of each emitted channel (skips AxisDisabled
      // 0-width channels, as for hls::axis<T,0,0,0>).
      SmallVector<unsigned, 7> ActiveIdx;
      // AXIS channel slot (0=data..6=dest) of each emitted channel. Equal to
      // ActiveIdx only when the packet struct has no rustc repr(C) alignment
      // pads ([N x i8] interior fields, e.g. axis_user's pad before u16).
      // SACArgs slot fill and the _V_<chan>_V name suffix use THIS, while
      // struct GEPs use ActiveIdx.
      SmallVector<unsigned, 7> ChanSlot;
    };
    SmallVector<ArgInfo, 8> Infos;
    SmallVector<Type *, 16> NewParams;
    for (Argument &A : Old->args()) {
      ArgInfo AI{};
      AI.Old = &A;
      AI.IsAxis = false;
      AI.StreamST = nullptr; AI.AxisST = nullptr;
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (PT) {
        auto *ST = dyn_cast<StructType>(PT->getElementType());
        if (ST && ST->hasName() &&
            (ST->getName().startswith("class.hls::stream<hls::axis<") ||
             (ST->getName().startswith("class.hls::stream<") &&
              AxisChanW.count(&A))) &&
            ST->getNumElements() == 3) {
          auto *AS = dyn_cast<StructType>(ST->getElementType(1));
          // Map struct fields → the 7 AXIS channel slots, skipping rustc
          // repr(C) alignment pads ([N x i8] interior/trailing arrays; e.g.
          // axis_user's `{Dis,Dis,Dis,[1xi8],u16,u8,Dis,Dis,[1xi8]}` where
          // user needs 2-align). slot k = k-th non-pad field; exactly 7
          // slots required.
          SmallVector<int, 9> SlotOfField;
          unsigned NSlots = 0;
          bool MapOk = (AS != nullptr);
          if (AS) {
            for (unsigned i = 0; i < AS->getNumElements(); ++i) {
              Type *FT = AS->getElementType(i);
              if (auto *AT = dyn_cast<ArrayType>(FT)) {
                if (AT->getElementType()->isIntegerTy(8)) {
                  SlotOfField.push_back(-1);  // pad
                  continue;
                }
              }
              if (NSlots >= 7) { MapOk = false; break; }
              SlotOfField.push_back((int)NSlots++);
            }
          }
          if (MapOk && NSlots == 7) {
            AI.IsAxis = true;
            AI.StreamST = ST;
            AI.AxisST = AS;
            auto WIt = AxisChanW.find(&A);
            for (unsigned i = 0; i < SlotOfField.size(); ++i) {
              if (SlotOfField[i] < 0) continue;  // pad field
              unsigned Slot = (unsigned)SlotOfField[i];
              Type *FT = AS->getElementType(i);
              // 0-width channel (AxisDisabled placeholder): no kernel arg —
              // the channel is dropped for hls::axis<T,0,0,0>.
              if (hlsrs::vxx::isAxisDisabledType(FT)) continue;
              // Narrow side channels (slot>=1) to the exact ap_uint<N>
              // width from the __vxx_axis_w marker; data (slot 0) stays as
              // the struct type. The cosim TB generator matches the stream by
              // these widths (e.g. axis_user u16 storage → i13).
              if (Slot >= 1 && WIt != AxisChanW.end() &&
                  WIt->second[Slot] > 0)
                FT = IntegerType::get(Ctx, WIt->second[Slot]);
              AI.ChanPtrTys.push_back(PointerType::get(FT, 0));
              AI.ChanNames.push_back(A.getName().str() + ChanSuffixes[Slot]);
              AI.ActiveIdx.push_back(i);
              AI.ChanSlot.push_back(Slot);
            }
            if (AI.ChanPtrTys.empty()) AI.IsAxis = false;  // all-disabled
          }
        }
      }
      if (AI.IsAxis) {
        for (Type *T : AI.ChanPtrTys) NewParams.push_back(T);
      } else {
        NewParams.push_back(A.getType());
      }
      Infos.push_back(AI);
    }

    FunctionType *NewFT = FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF = Function::Create(NewFT, Old->getLinkage(),
                                       Old->getAddressSpace(),
                                       Old->getName() + ".axis_disagg", &M);
    NewF->copyAttributesFrom(Old);

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "axis.entry", NewF);
    IRBuilder<> EB(EntryBB);

    ValueToValueMapTy VMap;
    auto NewArgIt = NewF->arg_begin();
    SmallVector<std::tuple<AllocaInst*, SmallVector<Argument*, 7>, StructType*, StructType*, std::string,
                           SmallVector<unsigned, 7>, SmallVector<unsigned, 7>>, 4> AxisAllocas;

    for (ArgInfo &AI : Infos) {
      if (!AI.IsAxis) {
        NewArgIt->setName(AI.Old->getName());
        // Copy noalias attribute
        if (AI.Old->hasAttribute(Attribute::NoAlias))
          NewArgIt->addAttr(Attribute::NoAlias);
        VMap[AI.Old] = &*NewArgIt;
        ++NewArgIt;
      } else {
        AllocaInst *AL = EB.CreateAlloca(AI.StreamST, nullptr,
                                         AI.Old->getName() + ".tmp");
        VMap[AI.Old] = AL;
        SmallVector<Argument*, 7> ChanArgs;
        for (unsigned i = 0; i < AI.ChanNames.size(); ++i) {
          NewArgIt->setName(AI.ChanNames[i]);
          NewArgIt->addAttr(Attribute::NoAlias);
          ChanArgs.push_back(&*NewArgIt);
          ++NewArgIt;
        }
        AxisAllocas.push_back({AL, ChanArgs, AI.StreamST, AI.AxisST,
                               AI.Old->getName().str(), AI.ActiveIdx,
                               AI.ChanSlot});
        // Bridge to the per-channel args is emitted AFTER clone via grouped
        // _ssdm_op_IfRead.Stream / IfWrite.Stream (RW-aware) — see below.
        // Raw per-channel load/store here would make cosim TB model each
        // sub-channel as a scalar Register (reads *param = queue-handle = 0)
        // instead of a Stream<Byte<N>>; the IfRead/IfWrite.Stream intrinsic is
        // what flags it as a streaming port.
      }
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap,
                      /*ModuleLevelChanges=*/false, Returns);

    // Find the cloned entry block (not our EntryBB)
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF) {
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    }
    if (!ClonedEntry) {
      NewF->eraseFromParent();
      continue;
    }
    // Emit grouped _ssdm_op_IfRead.Stream (input, in EntryBB before the br) /
    // IfWrite.Stream (output, before each return) instead of raw load/store, so
    // cosim TB models the AXIS port as a Stream<Byte<N>> (pops/pushes its queue)
    // rather than a scalar Register. Per-arg RW is classified from the cloned
    // body: a stream the body LOADS is an input (IfRead only); one it STORES is
    // an output (IfWrite only). Emitting both directions would add a spurious
    // IfRead on an output / IfWrite on an input → handshake deadlock. Args =
    // (7 stream chan ptrs, 7 alloca field ptrs); mangled name encodes both
    // halves' element widths.
    auto emitIfStream = [&](IRBuilder<> &B, bool IsRead, AllocaInst *AL,
                            const SmallVector<Argument *, 7> &ChanArgs,
                            StructType *StreamST, StructType *AxisST,
                            const SmallVector<unsigned, 7> &ActiveIdx) {
      Value *AlAxisPtr = B.CreateInBoundsGEP(StreamST, AL,
          {ConstantInt::get(I64, 0), ConstantInt::get(I32, 1)});
      // The channel args are at the EXACT ap_uint<N> widths (narrowed
      // from __vxx_axis_w). The body alloca fields are the Rust struct's
      // widened types (i8 side channels). IfRead/IfWrite.Stream must be
      // fully narrow (both halves) for cosim TB recognition,
      // so route them through per-channel NARROW temp allocas and bridge to
      // the i8 body fields with zext (read) / trunc (write). When no width
      // marker is present the narrow type == field type → no-op trunc/zext.
      BasicBlock &FEntry = AL->getFunction()->getEntryBlock();
      SmallVector<Value *, 7> FieldPtrs;
      SmallVector<AllocaInst *, 7> NarrowTmps;
      SmallVector<Type *, 7> NarrowTys;
      for (unsigned i = 0; i < ChanArgs.size(); ++i) {
        FieldPtrs.push_back(B.CreateInBoundsGEP(AxisST, AlAxisPtr,
            {ConstantInt::get(I64, 0), ConstantInt::get(I32, ActiveIdx[i])}));
        Type *NT = cast<PointerType>(ChanArgs[i]->getType())->getElementType();
        NarrowTys.push_back(NT);
        IRBuilder<> AB(&FEntry, FEntry.getFirstInsertionPt());
        NarrowTmps.push_back(AB.CreateAlloca(NT, nullptr));
      }
      // Write: body field (wide) → narrow temp (trunc) BEFORE the call.
      if (!IsRead) {
        for (unsigned i = 0; i < ChanArgs.size(); ++i) {
          Type *FldTy = AxisST->getElementType(ActiveIdx[i]);
          Value *V = B.CreateLoad(FldTy, FieldPtrs[i]);
          if (NarrowTys[i] != FldTy) V = B.CreateTrunc(V, NarrowTys[i]);
          B.CreateStore(V, NarrowTmps[i]);
        }
      }
      std::string Name =
          std::string("_ssdm_op_If") + (IsRead ? "Read" : "Write") + ".Stream";
      SmallVector<Type *, 14> PTys;
      SmallVector<Value *, 14> Args;
      for (unsigned half = 0; half < 2; ++half)
        for (unsigned i = 0; i < ChanArgs.size(); ++i) {
          Value *P = (half == 0) ? (Value *)ChanArgs[i] : (Value *)NarrowTmps[i];
          PTys.push_back(P->getType());
          Args.push_back(P);
          Name += ".p0i" + std::to_string(NarrowTys[i]->getIntegerBitWidth());
        }
      FunctionType *IfFT =
          FunctionType::get(Type::getVoidTy(Ctx), PTys, false);
      FunctionCallee IfFn = M.getOrInsertFunction(Name, IfFT);
      B.CreateCall(IfFn, Args);
      // Read: narrow temp → body field (zext) AFTER the call.
      if (IsRead) {
        for (unsigned i = 0; i < ChanArgs.size(); ++i) {
          Type *FldTy = AxisST->getElementType(ActiveIdx[i]);
          Value *V = B.CreateLoad(NarrowTys[i], NarrowTmps[i]);
          if (NarrowTys[i] != FldTy) V = B.CreateZExt(V, FldTy);
          B.CreateStore(V, FieldPtrs[i]);
        }
      }
    };
    auto allocaIsLoadedStored = [](AllocaInst *AL, bool &HasLoad,
                                   bool &HasStore) {
      SmallVector<Value *, 16> WL{AL};
      SmallPtrSet<Value *, 16> Seen;
      while (!WL.empty()) {
        Value *V = WL.pop_back_val();
        if (!Seen.insert(V).second) continue;
        for (User *U : V->users()) {
          if (isa<LoadInst>(U)) HasLoad = true;
          else if (auto *S = dyn_cast<StoreInst>(U)) {
            if (S->getPointerOperand() == V) HasStore = true;
          } else if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U))
            WL.push_back(U);
        }
      }
    };
    // Classify each AXIS alloca's body RW ONCE, BEFORE emitting any bridges.
    // The read bridge stores narrow-temp values into the alloca fields, which
    // would otherwise make the (later) write classifier see a spurious store
    // on an input stream → emit a phantom IfWrite → SYNCHK 200-92 ("port has
    // both read and write operations").
    SmallVector<std::pair<bool, bool>, 4> AxisRW;  // (HasLoad, HasStore)
    for (auto &Tup : AxisAllocas) {
      bool HasLoad = false, HasStore = false;
      allocaIsLoadedStored(std::get<0>(Tup), HasLoad, HasStore);
      AxisRW.push_back({HasLoad, HasStore});
    }
    // Branch to the cloned body, then emit the input IfRead.Stream at the
    // START of the cloned body — NOT in the entry block. The per-channel
    // SpecInterface/SpecAXISSideChannel declarations (emitted into the entry
    // block below) must come BEFORE the IfRead.Stream access; with the
    // read eagerly in the entry block ahead of those Spec* calls, the HLS
    // backend does not group the channels into a stream-of-axis and cosim TB
    // models them as 14 scalar Registers (212-359). Emitting IfRead at the body
    // entry keeps it after the entry-block Spec* declarations.
    EB.CreateBr(ClonedEntry);
    // Emit the grouped IfRead/IfWrite AT THE ORIGINAL ACCESS SITES (each
    // llvm.fpga.axis.pop/push call on the stream alloca), NOT once at body
    // entry / return. IfRead/IfWrite belong INSIDE the do-while
    // loop block (one per iteration); entry/return emission models exactly
    // one beat per call, which is only correct for single-shot kernels
    // (side_channel) — looped kernels (with_struct/to_master) synthesized a
    // one-read FSM whose A_TREADY never rises again -> xsim 0/N stall.
    // Falls back to entry/return emission when a stream alloca has no
    // pop/push site (defensive).
    {
      // Map each pop/push call to its stream alloca via the first arg's
      // GEP/bitcast base.
      auto baseAlloca = [](Value *V) -> AllocaInst * {
        while (true) {
          if (auto *AL = dyn_cast<AllocaInst>(V)) return AL;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) { V = GEP->getPointerOperand(); continue; }
          if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
          if (auto *CE = dyn_cast<ConstantExpr>(V)) { V = CE->getOperand(0); continue; }
          return nullptr;
        }
      };
      SmallVector<bool, 4> SiteEmittedRead(AxisAllocas.size(), false);
      SmallVector<bool, 4> SiteEmittedWrite(AxisAllocas.size(), false);
      SmallVector<std::pair<CallInst *, bool>, 8> Sites;  // (call, isPop)
      for (BasicBlock &BB : *NewF) {
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          StringRef N = CI->getCalledFunction()->getName();
          if (N.startswith("llvm.fpga.axis.pop")) Sites.push_back({CI, true});
          else if (N.startswith("llvm.fpga.axis.push")) Sites.push_back({CI, false});
        }
      }
      for (auto &SP : Sites) {
        CallInst *CI = SP.first;
        bool IsPop = SP.second;
        if (CI->arg_size() == 0) continue;
        AllocaInst *Base = baseAlloca(CI->getArgOperand(0));
        for (size_t k = 0; k < AxisAllocas.size(); ++k) {
          auto &Tup = AxisAllocas[k];
          if (std::get<0>(Tup) != Base) continue;
          if (IsPop) {
            // IfRead refreshes the alloca's element from the channel args
            // right BEFORE the pop copies alloca -> local temp.
            IRBuilder<> B(CI);
            emitIfStream(B, /*IsRead=*/true, std::get<0>(Tup), std::get<1>(Tup),
                         std::get<2>(Tup), std::get<3>(Tup), std::get<5>(Tup));
            SiteEmittedRead[k] = true;
          } else {
            // IfWrite publishes the alloca's element to the channel args
            // right AFTER the push copied local temp -> alloca.
            IRBuilder<> B(CI->getNextNode());
            emitIfStream(B, /*IsRead=*/false, std::get<0>(Tup), std::get<1>(Tup),
                         std::get<2>(Tup), std::get<3>(Tup), std::get<5>(Tup));
            SiteEmittedWrite[k] = true;
          }
          break;
        }
      }
      // Defensive fallback: keep the old entry/return emission for any
      // stream the body reads/writes WITHOUT a pop/push intrinsic.
      IRBuilder<> CEB(&*ClonedEntry->getFirstInsertionPt());
      for (size_t k = 0; k < AxisAllocas.size(); ++k) {
        auto &Tup = AxisAllocas[k];
        if (AxisRW[k].first && !SiteEmittedRead[k])
          emitIfStream(CEB, /*IsRead=*/true, std::get<0>(Tup), std::get<1>(Tup),
                       std::get<2>(Tup), std::get<3>(Tup), std::get<5>(Tup));
      }
      for (ReturnInst *RI : Returns) {
        IRBuilder<> RB(RI);
        for (size_t k = 0; k < AxisAllocas.size(); ++k) {
          auto &Tup = AxisAllocas[k];
          if (AxisRW[k].second && !SiteEmittedWrite[k])
            emitIfStream(RB, /*IsRead=*/false, std::get<0>(Tup), std::get<1>(Tup),
                         std::get<2>(Tup), std::get<3>(Tup), std::get<5>(Tup));
        }
      }
    }

    // Remove old SpecInterface/SpecBitsMap calls that reference the AXIS
    // allocas (they were referencing the old AXIS arg; via VMap remap they
    // now point to allocas, which is wrong — we'll re-emit per-channel
    // SpecInterface).
    SmallVector<Instruction *, 8> Dead;
    for (BasicBlock &BB : *NewF) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N != "_ssdm_op_SpecInterface" && N != "_ssdm_op_SpecBitsMap")
          continue;
        if (CI->arg_size() < 1) continue;
        Value *V = CI->getArgOperand(0);
        while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
        for (auto &Tup : AxisAllocas) {
          if (V == std::get<0>(Tup)) {
            Dead.push_back(CI);
            break;
          }
        }
      }
    }
    for (auto *I : Dead) I->eraseFromParent();

    // Emit per-channel SpecInterface + SpecBitsMap for each AXIS slot.
    // Use existing emitAxisDisaggSpec-style format.
    {
      FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), true);
      FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
      FunctionCallee SBMFn  = M.getOrInsertFunction("_ssdm_op_SpecBitsMap",   SpecTy);
      FunctionCallee SACFn  = M.getOrInsertFunction("_ssdm_op_SpecAXISSideChannel", SpecTy);
      if (auto *FF = dyn_cast<Function>(SpecFn.getCallee())) FF->addFnAttr(Attribute::NoUnwind);
      if (auto *FF = dyn_cast<Function>(SBMFn.getCallee()))  FF->addFnAttr(Attribute::NoUnwind);
      if (auto *FF = dyn_cast<Function>(SACFn.getCallee()))  FF->addFnAttr(Attribute::NoUnwind);
      GlobalVariable *AxisStr  = getOrCreateCStrGlobal(M, "axis");
      GlobalVariable *BothStr  = getOrCreateCStrGlobal(M, "both");
      GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
      MDNode *EmptyMD = MDNode::get(Ctx, ArrayRef<Metadata*>{});
      MDNode *MapMD   = MDNode::get(Ctx, {EmptyMD});
      // Insert at start of cloned entry (after our pop-wiring is in axis.entry — different BB).
      // Actually insert at start of axis.entry, before the branch.
      Instruction *BR = EntryBB->getTerminator();
      IRBuilder<> SB(BR);
      for (auto &Tup : AxisAllocas) {
        const auto &ChanArgs = std::get<1>(Tup);
        const std::string &PortName = std::get<4>(Tup);
        // SpecInterface: 7 chan ptrs + axis + 1,1 + "both" + 0,0 + "","",""
        //                + 0,0,0,0 + "","" + -1 + 0,0,0
        SmallVector<Value*, 32> SIArgs;
        for (unsigned i = 0; i < ChanArgs.size(); ++i) SIArgs.push_back(ChanArgs[i]);
        SIArgs.push_back(AxisStr);
        SIArgs.push_back(ConstantInt::get(I32, 1));
        SIArgs.push_back(ConstantInt::get(I32, 1));
        SIArgs.push_back(BothStr);
        SIArgs.push_back(ConstantInt::get(I32, 0));
        SIArgs.push_back(ConstantInt::get(I32, 0));
        for (int k = 0; k < 3; ++k) SIArgs.push_back(EmptyStr);
        for (int k = 0; k < 4; ++k) SIArgs.push_back(ConstantInt::get(I32, 0));
        SIArgs.push_back(EmptyStr);
        SIArgs.push_back(EmptyStr);
        SIArgs.push_back(ConstantInt::getSigned(I32, -1));
        for (int k = 0; k < 3; ++k) SIArgs.push_back(ConstantInt::get(I32, 0));
        SB.CreateCall(SpecFn, SIArgs);
        // SpecBitsMap per channel
        for (unsigned i = 0; i < ChanArgs.size(); ++i) {
          CallInst *SBM = SB.CreateCall(SBMFn, ArrayRef<Value*>{(Value*)ChanArgs[i]});
          SBM->setMetadata("map", MapMD);
        }
        // SpecAXISSideChannel: 7 chan ptrs + bundle_name_str
        // Bundle name = port name (e.g. "A", "B"). Without this, HLS emits
        // 7 separate AXIS bundles instead of 1 bundle with 7 sub-channels.
        GlobalVariable *BundleNameStr = getOrCreateCStrGlobal(M, PortName);
        // SpecAXISSideChannel has a FIXED 7-slot channel layout — the HLS
        // backend reads fixed operand indices and SIGSEGVs CDFG if slots are
        // missing. Fill 0-width channels with `i1* null`.
        const auto &Slot = std::get<6>(Tup);
        PointerType *I1P = PointerType::get(Type::getInt1Ty(Ctx), 0);
        SmallVector<Value*, 8> SACArgs(7,
            (Value*)ConstantPointerNull::get(I1P));
        for (unsigned k = 0; k < ChanArgs.size(); ++k)
          SACArgs[Slot[k]] = ChanArgs[k];
        SACArgs.push_back(BundleNameStr);
        SB.CreateCall(SACFn, SACArgs);
      }
    }

    // Replace old function with new (rename + erase)
    std::string OldName = Old->getName().str();
    if (Old->hasNUses(0)) {
      Old->eraseFromParent();
    } else {
      // Cast to old type and RAUW (cosim TB may reference by old name later)
      Old->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, Old->getType()));
      Old->eraseFromParent();
    }
    NewF->setName(OldName);
    Changed = true;
  }
  return Changed;
}

// Rewrite `llvm.fpga.axis.pop/push` body intrinsics to per-channel
// load+store pairs. The 14-arg axis intrinsic triggers the HLS backend's GVN
// pass to recurse infinitely in `BasicAAResult::getModRefInfo`, crashing
// csynth with HLS 200-1715 "Encountered problem during source synthesis"
// (observed on `using_axi_stream_with_side_channel_data`).
//
// Semantics: `llvm.fpga.axis.pop(src1..src7, dst1..dst7)` reads from src
// channels and writes to dst channels. Replace with 7× `load src; store dst`.
// `llvm.fpga.axis.push(dst1..dst7, src1..src7)` mirrors — but our IR's push
// call also has src in last 7. Identical lowering.
//
// Effect on HLS: per-channel load/store on the struct GEPs is what the HLS
// backend's auto-disagg recognizes — kernel sig stays struct-ptr but the
// body is in disaggregated form.
bool rewriteAxisPopPushToLoadStore(Module &M) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N.startswith("llvm.fpga.axis.pop.") ||
            N.startswith("llvm.fpga.axis.push.")) {
          Calls.push_back(CI);
        }
      }
  }
  if (Calls.empty()) return false;

  bool Changed = false;
  for (CallInst *CI : Calls) {
    unsigned N = CI->arg_size();
    if (N == 0 || (N & 1) != 0) continue; // need even count
    unsigned Half = N / 2;
    // pop: first half = src (from stream), last half = dst (local)
    // push: first half = dst (stream), last half = src (local)
    // Both lower to N/2 load+store pairs (load from src ptr, store to dst).
    bool IsPush = CI->getCalledFunction()->getName().startswith(
        "llvm.fpga.axis.push.");
    IRBuilder<> B(CI);
    for (unsigned i = 0; i < Half; ++i) {
      Value *FirstHalf = CI->getArgOperand(i);
      Value *LastHalf = CI->getArgOperand(i + Half);
      Value *SrcPtr = IsPush ? LastHalf : FirstHalf;
      Value *DstPtr = IsPush ? FirstHalf : LastHalf;
      auto *SrcPT = dyn_cast<PointerType>(SrcPtr->getType());
      auto *DstPT = dyn_cast<PointerType>(DstPtr->getType());
      if (!SrcPT || !DstPT) continue;
      Type *SrcElem = SrcPT->getElementType();
      // Use src type for load; bitcast dst if types differ (shouldn't but
      // be safe).
      Value *DstUsed = DstPtr;
      if (DstPT->getElementType() != SrcElem) {
        DstUsed = B.CreateBitCast(DstPtr,
                                  PointerType::get(SrcElem, DstPT->getAddressSpace()));
      }
      LoadInst *LD = B.CreateLoad(SrcElem, SrcPtr);
      B.CreateStore(LD, DstUsed);
    }
    CI->eraseFromParent();
    Changed = true;
    (void)Ctx;
  }
  // Erase now-dead pop/push DECLARATIONS too: LLVM-7 llvm-as re-derives
  // intrinsic attributes for `llvm.fpga.axis.*` by NAME at parse time and
  // rejects our overloaded variants ("Attribute after last parameter!" on
  // e.g. `llvm.fpga.axis.pop.p0i16.p0i8(i16*, i8*, i16*, i8*)`,
  // custom_side_2). A use-empty declaration carries zero information for
  // the HLS backend, so drop it.
  SmallVector<Function *, 4> DeadDecls;
  for (Function &F : M)
    if (F.isDeclaration() && F.use_empty() &&
        (F.getName().startswith("llvm.fpga.axis.pop.") ||
         F.getName().startswith("llvm.fpga.axis.push.")))
      DeadDecls.push_back(&F);
  for (Function *F : DeadDecls) F->eraseFromParent();
  return Changed;
}

bool injectArrayAxis(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_axis_array");
}

// AXIS disagg: emit `_ssdm_op_SpecInterface` + `_ssdm_op_SpecBitsMap` +
// `_ssdm_op_SpecAXISSideChannel` for explicit-disagg AXIS args
// (`<port>_V_<chan>_V` named args grouped per AXIS interface).
//
// Emitted IR shape, e.g. axis_4ch (data+keep+strb+last):
//   call void @_ssdm_op_SpecInterface(
//     i32* %A_V_data_V, i4* %A_V_keep_V, i4* %A_V_strb_V, i1* %A_V_last_V,
//     [5 x i8]* @0 "axis", i32 1, i32 1, [5 x i8]* @1 "both",
//     i32 0, i32 0,
//     [1 x i8]* @2 "", [1 x i8]* @2 "", [1 x i8]* @2 "",
//     i32 0, i32 0, i32 0, i32 0,
//     [1 x i8]* @2 "", [1 x i8]* @2 "",
//     i32 -1, i32 0, i32 0, i32 0)
//   call void @_ssdm_op_SpecBitsMap(<each chan ptr>), !map !6
//   call void @_ssdm_op_SpecAXISSideChannel(
//     i32* %data, i4* %keep, i4* %strb,
//     i1* null /*user*/, i1* %last,
//     i1* null /*id*/, i1* null /*dest*/,
//     [2 x i8]* @4 "A")
//
// Marker calls (`__vxx_axis_disagg`, `__vxx_axis_user_disagg`,
// `__vxx_axis_7ch_disagg`) carry the channel ptrs as `i8*` args (bitcast
// from the function arg). We trace the bitcast to recover the original
// `Argument`, deduce the port name from arg name (`A_V_data_V` → "A"),
// and emit the Spec ops at the marker call site.
//
// SlotMap: marker-arg index → 7-slot index (0=data,1=keep,2=strb,3=user,4=last,5=id,6=dest)
bool emitAxisDisaggSpec(Module &M, StringRef MarkerName,
                                ArrayRef<int> SlotMap) {
  Function *Marker = M.getFunction(MarkerName);
  if (!Marker)
    return false;

  LLVMContext &Ctx = M.getContext();
  Type *I8 = IntegerType::get(Ctx, 8);
  Type *I32 = IntegerType::get(Ctx, 32);
  Type *I1 = IntegerType::get(Ctx, 1);
  PointerType *I1Ptr = PointerType::getUnqual(I1);

  FunctionType *SITy =
      FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/true);
  FunctionCallee SIFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SITy);
  FunctionCallee SBMFn =
      M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SITy);
  FunctionCallee SACFn =
      M.getOrInsertFunction("_ssdm_op_SpecAXISSideChannel", SITy);

  // Get-or-create private string global, returns pointer.
  StringMap<Constant *> StrGlobals;
  auto strPtr = [&](StringRef s) -> Constant * {
    auto It = StrGlobals.find(s);
    if (It != StrGlobals.end())
      return It->second;
    Constant *Init = ConstantDataArray::getString(Ctx, s, /*AddNull=*/true);
    auto *GV = new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                                  GlobalValue::PrivateLinkage, Init,
                                  ".str.axis");
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    Constant *Ptr = GV;
    StrGlobals[s] = Ptr;
    return Ptr;
  };
  Constant *AxisStr = strPtr("axis");
  Constant *BothStr = strPtr("both");
  Constant *EmptyStr = strPtr("");

  // Channel-name suffixes (index = 7-slot position).
  static const char *ChanSuffixes[7] = {
    "_V_data_V", "_V_keep_V", "_V_strb_V",
    "_V_user_V", "_V_last_V", "_V_id_V", "_V_dest_V"
  };

  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;

  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    Dead.push_back(CI);

    // Collect 7-slot pointer values (nullptr = absent).
    SmallVector<Value *, 7> ChanPtrs(7, nullptr);
    StringRef PortName;
    for (unsigned i = 0; i < CI->arg_size() && i < SlotMap.size(); ++i) {
      int Slot = SlotMap[i];
      if (Slot < 0 || Slot >= 7) continue;
      Value *V = CI->getArgOperand(i);
      // Trace through bitcasts (Rust opaque-ptr lowering inserts i8* casts).
      while (true) {
        if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
        if (auto *BCC = dyn_cast<BitCastOperator>(V)) { V = BCC->getOperand(0); continue; }
        break;
      }
      ChanPtrs[Slot] = V;
      // Capture port name from first arg's underlying Argument.
      if (PortName.empty()) {
        if (auto *Arg = dyn_cast<Argument>(V)) {
          StringRef N = Arg->getName();
          if (N.endswith(ChanSuffixes[Slot]))
            PortName = N.drop_back(strlen(ChanSuffixes[Slot]));
        }
      }
    }

    if (PortName.empty()) {
      vxxDbg() << "VXXPrep: " << MarkerName << ": no port name found, skip\n";
      continue;
    }

    Constant *PortStr = strPtr(PortName);

    IRBuilder<> B(CI);

    // SpecInterface arg layout:
    //   [channels...] axis "both" 0 0 "" "" "" 0 0 0 0 "" "" -1 0 0 0
    SmallVector<Value *, 32> SIArgs;
    // For SpecInterface: pass each present channel ptr (bitcast to its
    // narrow iN* if needed). The Spec op expects iN* (i32, i4, i1 etc) but the
    // Rust args are [N x iN]* — we cast to the element type for the
    // Spec op so the HLS backend sees a scalar stream ptr.
    auto castToElemPtr = [&](Value *V) -> Value * {
      Type *T = V->getType();
      if (auto *PT = dyn_cast<PointerType>(T)) {
        Type *Elem = PT->getElementType();
        if (auto *AT = dyn_cast<ArrayType>(Elem)) {
          // [N x iN]* → iN* via bitcast
          Type *NewT = PointerType::get(AT->getElementType(),
                                         PT->getAddressSpace());
          return B.CreateBitCast(V, NewT, V->getName() + ".axisptr");
        }
      }
      return V;
    };
    // Build SpecInterface: only present channels go into the leading slots
    // (the active set per axis_user / axis_4ch).
    SmallVector<Value *, 7> ActiveCasts;
    for (int s = 0; s < 7; ++s) {
      if (ChanPtrs[s]) {
        Value *Cast = castToElemPtr(ChanPtrs[s]);
        ActiveCasts.push_back(Cast);
        SIArgs.push_back(Cast);
      }
    }
    SIArgs.push_back(AxisStr);
    SIArgs.push_back(ConstantInt::get(I32, 1));
    SIArgs.push_back(ConstantInt::get(I32, 1));
    SIArgs.push_back(BothStr);
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(EmptyStr);
    SIArgs.push_back(EmptyStr);
    SIArgs.push_back(EmptyStr);
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(EmptyStr);
    SIArgs.push_back(EmptyStr);
    SIArgs.push_back(ConstantInt::get(I32, (uint64_t)-1, true));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    SIArgs.push_back(ConstantInt::get(I32, 0));
    {
      // Channel args are kept as `[N x iN]*` arrays (see
      // decayKernelArrayParams) so cosim TB feeds them as A2Stream vs
      // Register (decayed scalar). But SpecInterface(axis) on an
      // array arg conflicts with HLS's ap_memory inference (XFORM 203-801). The
      // single-axis path (injectArrayAxisSpec) emits an `xlx_axis` op-bundle ON
      // the array ptr, which overrides ap_memory (array-compatible). Mirror that
      // here per channel; SpecAXISSideChannel below still groups them into one
      // AXIS interface with sub-channels.
      Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
      Type *I64t = IntegerType::get(Ctx, 64);
      Constant *Zero8 = ConstantAggregateZero::get(ArrayType::get(I8, 0));
      for (int s = 0; s < 7; ++s) {
        if (!ChanPtrs[s]) continue;
        Value *BArgs[] = {ChanPtrs[s], ConstantInt::get(I32, 1),
                          ConstantInt::get(I64t, 2),
                          ConstantInt::get(I64t, (uint64_t)-1, true),
                          (Value *)Zero8, (Value *)Zero8};
        OperandBundleDef OBD("xlx_axis", ArrayRef<Value *>(BArgs, 6));
        CallInst *C = B.CreateCall(SE, None, ArrayRef<OperandBundleDef>{OBD});
        C->setOnlyAccessesInaccessibleMemory();
        C->setDoesNotThrow();
      }
    }

    // SpecBitsMap per active channel.
    for (Value *V : ActiveCasts)
      B.CreateCall(SBMFn, ArrayRef<Value *>{V});

    // SpecAXISSideChannel: 7 fixed slots + port name.
    // Use i1* null for absent channels.
    SmallVector<Value *, 8> SACArgs;
    for (int s = 0; s < 7; ++s) {
      if (ChanPtrs[s]) {
        // Active: pass the same cast we built for SpecInterface.
        // Rebuild a cast at this point so it dominates.
        SACArgs.push_back(castToElemPtr(ChanPtrs[s]));
      } else {
        SACArgs.push_back(ConstantPointerNull::get(I1Ptr));
      }
    }
    SACArgs.push_back(PortStr);
    B.CreateCall(SACFn, SACArgs);

    Changed = true;
  }

  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

// For single-ptr `__vxx_axis(ptr)` markers (e.g. aliasing_axi_master_ports
// uses `barista_hls::axis(inputStream)` on `&[u32; SIZE]` array arg), emit
// `_ssdm_op_SpecInterface(arg, "axis", 1, 1, "both", 0, 0, "", "", "", 0,
// 0, 0, 0, "", "", -1, 0, 0, 0)` + `_ssdm_op_SpecBitsMap(arg)` at function
// entry. This is the form expected for `hls::stream<T>&` args.
bool injectArrayAxisSpec(Module &M) {
  Function *Marker = M.getFunction("__vxx_axis");
  if (!Marker) return false;
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  // The AXIS pragma is carried as
  //   `llvm.sideeffect() [ "xlx_axis"(ptr, i32 isReg, i64 regMode, i64 depth,
  //                                    [0 x i8] sigName, [0 x i8] bundle) ]`
  //   "xlx_axis"(ptr %A, i32 1, i64 2, i64 -1, [0 x i8] zero, [0 x i8] zero)
  // 6 args. The HLS backend converts this to SpecInterface(axis). Emitting
  // SpecInterface here directly causes 4 SpecInterface downstream (2 auto
  // ap_auto + 2 mine) → cosim TB 212-361. Use op-bundle instead.
  Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  ArrayType *ZeroByteArrTy = ArrayType::get(Type::getInt8Ty(Ctx), 0);
  Constant *ZeroByteArr = ConstantAggregateZero::get(ZeroByteArrTy);

  SmallVector<Argument *, 4> Targets;
  SmallVector<CallInst *, 8> Dead;
  SmallPtrSet<Argument *, 4> Seen;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 1) continue;
    Dead.push_back(CI);
    Value *V = CI->getArgOperand(0);
    while (auto *BC = dyn_cast<BitCastInst>(V)) V = BC->getOperand(0);
    while (auto *BCC = dyn_cast<BitCastOperator>(V)) V = BCC->getOperand(0);
    auto *A = dyn_cast<Argument>(V);
    if (!A || !Seen.insert(A).second) continue;
    Targets.push_back(A);
  }
  for (Argument *A : Targets) {
    Function *F = A->getParent();
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    Value *BundleArgs[] = {
        (Value*)A,
        ConstantInt::get(I32, 1),              // isRegister
        ConstantInt::get(I64, 2),              // registerMode
        ConstantInt::get(I64, -1, true),       // depth=-1 (default)
        (Value*)ZeroByteArr,                   // signalName=""
        (Value*)ZeroByteArr};                  // bundle="" (6th arg)
    OperandBundleDef OBD("xlx_axis", ArrayRef<Value*>(BundleArgs, 6));
    CallInst *SECall = B.CreateCall(SEFn, None, ArrayRef<OperandBundleDef>{OBD});
    SECall->setOnlyAccessesInaccessibleMemory();
    SECall->setDoesNotThrow();
    // `__vxx_axis_packed`: for packed AXIS struct streams (arg type
    // `class.hls::stream<hls::axis<...>>*`), also emit the `stream_interface`
    // op-bundle alongside xlx_axis. Without it the HLS backend may not
    // recognise the packed arg as a stream to disaggregate.
    if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
      if (auto *PT = dyn_cast<PointerType>(A->getType())) {
        if (auto *ST = dyn_cast<StructType>(PT->getElementType())) {
          if (ST->hasName() &&
              ST->getName().startswith("class.hls::stream<hls::axis<")) {
            OperandBundleDef SI("stream_interface", ArrayRef<Value *>((Value *)A));
            CallInst *SICall =
                B.CreateCall(SEFn, None, ArrayRef<OperandBundleDef>{SI});
            SICall->setOnlyAccessesInaccessibleMemory();
            SICall->setDoesNotThrow();
          }
        }
      }
    }
    (void)I32; (void)I64;
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

bool injectAxisDisaggSpecInterface(Module &M) {
  // axis_4ch marker: args (data, keep, strb, last) → slots {0, 1, 2, 4}.
  static const int SlotMap[4] = {0, 1, 2, 4};
  return emitAxisDisaggSpec(M, "__vxx_axis_disagg", SlotMap);
}

bool injectAxisUserDisaggSpecInterface(Module &M) {
  // axis_user marker: args (user, last) → slots {3, 4}.
  static const int SlotMap[2] = {3, 4};
  return emitAxisDisaggSpec(M, "__vxx_axis_user_disagg", SlotMap);
}

bool injectAxis7chDisaggSpecInterface(Module &M) {
  // axis_7ch marker: args (data, keep, strb, user, last, id, dest) → slots {0..6}.
  static const int SlotMap[7] = {0, 1, 2, 3, 4, 5, 6};
  return emitAxisDisaggSpec(M, "__vxx_axis_7ch_disagg", SlotMap);
}

// Region-scoped (intra-function) dataflow scope. Top-level dataflow goes
// through injectDataflowAttribute. Drop the marker here; intra-fn dataflow
// For each `__vxx_dataflow()` marker call, emit a
// `call void (...) @_ssdm_op_SpecDataflowPipeline(i32 -1, i32 0)` at
// the same site. C++ `#pragma HLS DATAFLOW` lowers to this op call;
// without it HLS's TOP directive analysis rejects kernels that fan
// out into helper functions (TLP/Control_driven middle_bypass etc).
// Rename `%"barista_hls::Stream<T>"` named struct → `%"class.hls::stream<T>"`.
// HLS TOP-directive analysis only recognizes `class.hls::stream<*>` as a
// kernel-arg stream type; the rustc-emitted `barista_hls::Stream<T>` name is
// treated as opaque, triggering HLS 200-1986 (TOP directive invalid).
// Rename happens by direct setName on the named StructType — keeps the
// inner element type as-is. Tries to canonicalise the inner Rust type name
// (e.g. `Packet` → `hls::axis<int,2,5,6>`) only if recognisable, else
// keeps the inner literally.
// Helper: detect if a named struct matches the AXIS-packet layout
// (from `ap_axis!` macro: `{ iN, u8, u8, u8, u8, u8, u8, [P x i8]? }`).
// If yes, return the AXIS data type bit width N; else 0.
unsigned detectAxisStructBits(StructType *ST, const DataLayout &DL) {
  if (ST->isOpaque() || !ST->hasName()) return 0;
  unsigned N = ST->getNumElements();
  if (N != 7 && N != 8) return 0;
  // Field 0: integer (data)
  auto *T0 = dyn_cast<IntegerType>(ST->getElementType(0));
  if (!T0) return 0;
  unsigned DataBits = T0->getBitWidth();
  if (DataBits != 8 && DataBits != 16 && DataBits != 32 && DataBits != 64) return 0;
  // Fields 1-6: i8 (keep, strb, user, last, id, dest) or the 1-byte
  // `barista_hls::AxisDisabled` placeholder (0-width channel,
  // `hls::axis<T,0,0,0>` shape — disagg drops those channels).
  for (unsigned i = 1; i < 7; ++i) {
    Type *FT = ST->getElementType(i);
    if (auto *Ti = dyn_cast<IntegerType>(FT)) {
      if (Ti->getBitWidth() != 8) return 0;
      continue;
    }
    if (hlsrs::vxx::isAxisDisabledType(FT)) continue;
    return 0;
  }
  // Field 7 (if present): [N x i8] padding
  if (N == 8) {
    auto *AT = dyn_cast<ArrayType>(ST->getElementType(7));
    if (!AT || !AT->getElementType()->isIntegerTy(8)) return 0;
  }
  (void)DL;
  return DataBits;
}

const char *bitsToCName(unsigned bits) {
  switch (bits) {
    case 8: return "char";
    case 16: return "short";
    case 32: return "int";
    case 64: return "long";
    default: return nullptr;
  }
}

bool renameStreamWrapperToCanonical(Module &M) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  // Pass 1: rename ap_axis-derived structs to `struct.hls::axis<DataType>`.
  // Maps the bare Rust struct name (e.g. `%TransPkt`) → canonical name.
  // The renamed type identity stays the same (StructType* by pointer is
  // shared), so all uses (incl. as inner of Stream<T>) auto-update.
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (ST->isOpaque() || !ST->hasName()) continue;
    StringRef N = ST->getName();
    if (N.startswith("struct.hls::axis<") ||
        N.startswith("class.hls::stream<") ||
        N.startswith("barista_hls::Stream<")) continue;
    // axis_user (data-less side-channel packet): data channel disabled + a
    // full 7-channel AXIS packet after skipping [K x i8] alignment pads. clang
    // names `hls::axis_user<W>` as `struct.hls::axis<void, W, 0, 0, 'P', true>`.
    // Marker-gated (user width non-zero only for the axis_user packet) so
    // ap_axis! streams are untouched; the true W comes from __vxx_axis_w.
    if (unsigned UW = axisUserWidthFromMarker(M)) {
      if (ST->getNumElements() >= 7 &&
          hlsrs::vxx::isAxisDisabledType(ST->getElementType(0))) {
        unsigned NCh = 0;
        for (unsigned i = 0; i < ST->getNumElements(); ++i) {
          Type *FT = ST->getElementType(i);
          if (auto *AT = dyn_cast<ArrayType>(FT))
            if (AT->getElementType()->isIntegerTy(8)) continue;  // pad
          ++NCh;
        }
        if (NCh >= 7) {
          ST->setName("struct.hls::axis<void, " + std::to_string(UW) +
                      ", 0, 0, 'P', true>");
          Changed = true;
          continue;
        }
      }
    }
    unsigned Bits = detectAxisStructBits(ST, DL);
    if (Bits == 0) continue;
    const char *DataCName = bitsToCName(Bits);
    if (!DataCName) continue;
    std::string NewName;
    if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
      // Canonical element name: `struct.hls::axis<ap_int<W>, U, TI, TD>`.
      // The cosim TB generator disaggregates a `hls::stream<hls::axis<...>>`
      // into the 7 sub-channel sim::Stream<Byte<N>> ports ONLY when the element
      // type name matches this canonical AXIS form. The bare `hls::axis<int>`
      // form makes cosim TB treat the whole struct as one packed Byte<12>.
      // U/TI/TD aren't recoverable from the i8 Rust fields, so they are fixed
      // at 2,5,6 (the side_channel example's dims). Channel byte-sizes still
      // come from the struct fields.
      unsigned U = 2, TI = 5, TD = 6;
      NewName = "struct.hls::axis<ap_int<" + std::to_string(Bits) + ">, " +
                std::to_string(U) + ", " + std::to_string(TI) + ", " +
                std::to_string(TD) + ">";
    } else {
      NewName = "struct.hls::axis<";
      NewName += DataCName;
      NewName += ">";
    }
    ST->setName(NewName);
    Changed = true;
  }
  // Pass 2: rename Stream<T> outer wrapper.
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (ST->isOpaque() || !ST->hasName()) continue;
    StringRef N = ST->getName();
    if (!N.startswith("barista_hls::Stream<")) continue;
    std::string Inner = N.substr(strlen("barista_hls::Stream<")).str();
    if (!Inner.empty() && Inner.back() == '>')
      Inner.pop_back();
    // Map simple Rust ints to canonical C type names.
    std::string CName = Inner;
    if (Inner == "i32") CName = "int";
    else if (Inner == "i64") CName = "long";
    else if (Inner == "i16") CName = "short";
    else if (Inner == "i8")  CName = "char";
    else if (Inner == "u32") CName = "unsigned int";
    else if (Inner == "u64") CName = "unsigned long";
    else if (Inner == "u16") CName = "unsigned short";
    else if (Inner == "u8")  CName = "unsigned char";
    else {
      // Look up the renamed struct from pass 1: if the Stream<T>'s inner
      // type is now `struct.hls::axis<...>`, use that as canonical.
      if (ST->getNumElements() >= 1) {
        // Stream<T> shape can be `{ T }` or `{ [0 x T'], T, [0 x T'] }`.
        unsigned MidIdx = (ST->getNumElements() == 3) ? 1 : 0;
        if (auto *InnerST = dyn_cast<StructType>(ST->getElementType(MidIdx))) {
          StringRef IName = InnerST->hasName() ? InnerST->getName() : "";
          if (IName.startswith("struct.hls::axis<")) {
            // Use renamed inner name verbatim.
            CName = IName.str();
            // Strip leading "struct." — the template arg is "hls::axis<...>"
            // (no "struct." prefix in template names).
            if (CName.compare(0, strlen("struct."), "struct.") == 0)
              CName.erase(0, strlen("struct."));
          }
        }
      }
    }
    std::string NewName;
    if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed") &&
        CName.compare(0, strlen("hls::axis<ap_int<"),
                      "hls::axis<ap_int<") == 0) {
      // Canonical wrapper: insert the trailing template defaults
      // `, '8', false` into the axis args and wrap as `<...>, 0>`.
      //   CName = "hls::axis<ap_int<32>, 2, 5, 6>"
      //   →      "class.hls::stream<hls::axis<ap_int<32>, 2, 5, 6, '8', false>, 0>"
      std::string Axis = CName;
      if (!Axis.empty() && Axis.back() == '>') Axis.pop_back();
      Axis += ", '8', false>";
      NewName = "class.hls::stream<" + Axis + ", 0>";
    } else if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed") &&
               CName.compare(0, strlen("hls::axis<void,"),
                             "hls::axis<void,") == 0) {
      // axis_user: CName is already the full 6-arg
      // `hls::axis<void, W, 0, 0, 'P', true>` form → just add the stream depth.
      NewName = "class.hls::stream<" + CName + ", 0>";
    } else {
      NewName = "class.hls::stream<" + CName + ">";
    }
    if (N == NewName) continue;
    ST->setName(NewName);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: renamed barista_hls::Stream<*> → class.hls::stream<*>\n";
  return Changed;
}

// `__vxx_axis_packed`: the HLS backend derives the AXIS side-channel ports
// (TKEEP/TSTRB/TUSER/TLAST/TID/TDEST) from the *field types* of the stream's
// inner axis struct. rustc emits a bare `{i32, i8×6, [pad]}` element → the
// backend cannot recover the per-channel widths and drops all side channels
// (12 ports vs the expected 24). getCppNestedAxisStruct already builds the
// nested `{ap_int<32>, ap_uint<4>, ...}` variant (the body's write side uses
// it). This pass repoints the kernel's `class.hls::stream<hls::axis<...>>*`
// args (+ their fifo.pop/push intrinsics + body GEP/load/store) to that nested
// element type so the backend emits all 24 ports. The x86_64 cosim
// build is a separate module and is unaffected. Gated by `__vxx_axis_packed`.
bool retypeAxisStreamToNested(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) return false;
  LLVMContext &Ctx = M.getContext();
  // Find wrapper W = class.hls::stream<hls::axis<...>> + its bare inner axis
  // struct (the struct-typed element whose name lacks the `.nest` suffix).
  StructType *W = nullptr, *Bare = nullptr;
  unsigned InnerIdx = 0;
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (!ST->hasName()) continue;
    if (!ST->getName().startswith("class.hls::stream<hls::axis<")) continue;
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      auto *E = dyn_cast<StructType>(ST->getElementType(i));
      if (E && E->hasName() &&
          E->getName().startswith("struct.hls::axis<") &&
          !E->getName().endswith(".nest")) {
        W = ST; Bare = E; InnerIdx = i; break;
      }
    }
    if (W) break;
  }
  if (!W || !Bare) return false;
  StructType *N = getCppNestedAxisStruct(M, Bare);
  if (N == Bare) return false;
  // The HLS backend keys AXIS side-channel emission off the EXACT canonical
  // type names. Move the bare structs aside and give the nested
  // element + wrapper the canonical `struct.hls::axis<...>` /
  // `class.hls::stream<hls::axis<...>>` names so the backend recognises them as
  // a full ap_axis<W,U,TI,TD> stream and emits all 24 ports.
  std::string BareName = Bare->getName().str();
  std::string WName = W->getName().str();
  Bare->setName(BareName + ".flat");
  N->setName(BareName);              // nested element gets canonical name
  W->setName(WName + ".flat");
  // Build W' = same shape as W, inner element replaced with the nested type.
  SmallVector<Type *, 3> WFields(W->element_begin(), W->element_end());
  WFields[InnerIdx] = N;
  StructType *Wp = StructType::create(Ctx, WFields, WName);
  DenseMap<Type *, Type *> TypeMap;
  TypeMap[Bare] = N;
  TypeMap[W] = Wp;
  Type *WPtr = PointerType::get(W, 0);
  Type *WpPtr = PointerType::get(Wp, 0);

  SmallVector<Function *, 2> Todo;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (auto &A : F.args())
      if (A.getType() == WPtr) { Todo.push_back(&F); break; }
  }
  bool Changed = false;
  for (Function *F : Todo) {
    FunctionType *OldFT = F->getFunctionType();
    SmallVector<Type *, 4> NP;
    for (unsigned i = 0; i < OldFT->getNumParams(); ++i)
      NP.push_back(OldFT->getParamType(i) == WPtr ? WpPtr
                                                  : OldFT->getParamType(i));
    FunctionType *NFT =
        FunctionType::get(F->getReturnType(), NP, F->isVarArg());
    Function *NewF = Function::Create(NFT, F->getLinkage(),
                                      F->getName() + ".axnst", &M);
    NewF->copyAttributesFrom(F);
    NewF->setDLLStorageClass(GlobalValue::DefaultStorageClass);
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);
    // Map old args → new args; for the retyped (W*→W'*) args the body's
    // operand uses are redirected via the seeded VMap and the dependent
    // GEP/load/store/fifo are recreated with remapped types by
    // substituteTypeInFunction. Same-typed args RAUW directly.
    SmallVector<std::pair<Value *, Value *>, 4> Seed;
    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < OldFT->getNumParams(); ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (OldA->getType() == NewA->getType())
        OldA->replaceAllUsesWith(&*NewA);
      else
        Seed.push_back({&*OldA, &*NewA});
    }
    hlsrs::vxx::substituteTypeInFunction(NewF, TypeMap, Seed);
    std::string Nm = F->getName().str();
    F->eraseFromParent();
    NewF->setName(Nm);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: retyped AXIS stream inner bare→nested "
              "(side-channel ports)\n";
  return Changed;
}

// `__vxx_axis_packed`: wrap the kernel's whole-struct fifo.pop/push on the
// nested AXIS element into read()/write() HELPER functions that
// contain the 14-arg `llvm.fpga.axis.pop`/`push`. The HLS backend's "AXIS
// element accessed once" rule applies only to the TOP kernel's interface arg;
// inlining the 7 per-channel GEPs into the kernel triggers HLS 200-1715.
// Placing them in a noinline helper (kernel calls it once per
// stream op) lets the backend propagate the side-channel interface from the
// axis.pop through the call and emit all 7 sub-channel RTL ports. Runs AFTER
// retypeAxisStreamToNested (needs nested element field types for the
// per-channel intrinsic arg types).
bool wrapAxisReadWriteInHelpers(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) return false;
  LLVMContext &Ctx = M.getContext();

  // Mark read/write + get_<chan>_ptr accessors with
  //   !fpga.function.pragma !{ !{!"fpga.inline", !"user", null} }
  // THIS is how the HLS backend inlines them into the kernel DURING the AXIS
  // interface analysis (so the axis.pop + accessor calls land on the kernel's
  // stream port → 7 side channels). The LLVM `alwaysinline`/`noinline` attr is
  // NOT the signal the backend uses — it keys on the fpga.function.pragma
  // metadata, emitted on each read/write/accessor.
  MDNode *FpgaInlinePragma = MDNode::get(
      Ctx, {MDNode::get(Ctx, {MDString::get(Ctx, "fpga.inline"),
                              MDString::get(Ctx, "user"),
                              (Metadata *)nullptr})});

  auto isNestedAxis = [](StructType *ST) -> bool {
    return ST && ST->hasName() &&
           ST->getName().startswith("struct.hls::axis<") &&
           !ST->getName().endswith(".flat") &&
           !ST->getName().endswith(".nest") &&
           ST->getNumElements() >= 7;
  };
  auto fieldPtrTys = [&](StructType *E) {
    SmallVector<Type *, 7> v;
    for (unsigned i = 0; i < E->getNumElements(); ++i)
      v.push_back(PointerType::get(E->getElementType(i), 0));
    return v;
  };

  // C++ channel-accessor names by field index (data,keep,strb,user,last,id,dest).
  static const char *ChanName[7] = {"get_data_ptr", "get_keep_ptr",
                                    "get_strb_ptr", "get_user_ptr",
                                    "get_last_ptr", "get_id_ptr",
                                    "get_dest_ptr"};
  // Per-(ElemST,field) accessor fn: `ptr get_<chan>_ptr(ElemST* this)` = GEP
  // field i, marked alwaysinline + `fpga.demangled.name="get_<chan>_ptr"`.
  // The HLS backend's AXIS pass maps struct field → AXIS sub-channel by this
  // demangled name (every axis.pop ptr is routed through these).
  DenseMap<std::pair<StructType *, unsigned>, Function *> AccH;
  auto getAccessor = [&](StructType *E, unsigned i) -> Function * {
    auto Key = std::make_pair(E, i);
    auto It = AccH.find(Key);
    if (It != AccH.end()) return It->second;
    Type *FldPtr = PointerType::get(E->getElementType(i), 0);
    Type *EP = PointerType::get(E, 0);
    FunctionType *FT = FunctionType::get(FldPtr, {EP}, false);
    // NOTE: name must NOT start with `__vxx_` — eraseUnimplementedMarkers
    // strips every `__vxx_*` function (and its calls) as an unimplemented
    // marker, which silently deletes these real helpers (the kernel is then left
    // reading uninitialized axis.rd.tmp → 12 data-only ports + 212-360).
    // The HLS backend keys on the fpga.demangled.name attr, not the symbol name.
    Function *A = Function::Create(
        FT, GlobalValue::LinkOnceODRLinkage,
        std::string("hlsaxis_") + ChanName[i], &M);
    A->addFnAttr(Attribute::NoInline);  // survive rustc opt-level=1; backend inlines via pragma
    A->setMetadata("fpga.function.pragma", FpgaInlinePragma);
    A->addFnAttr("fpga.demangled.name", ChanName[i]);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", A);
    IRBuilder<> B(BB);
    B.CreateRet(B.CreateConstGEP2_32(E, A->getArg(0), 0, i));
    AccH[Key] = A;
    return A;
  };

  DenseMap<StructType *, Function *> ReadH, WriteH;
  // WrapTy/EltFld: stream-class mode — the stream arg is the
  // `class.hls::stream*` WRAPPER, and the helper GEPs(this,0,EltFld) to the axis
  // element internally (stream::read GEP(this,0,0)). The HLS backend keys on the
  // stream-class arg type to disagg; the bare element makes it mis-specialize
  // the out-of-line helper into a CVP-crashing clone. WrapTy==nullptr → legacy.
  auto getHelper = [&](StructType *E, bool IsWrite, StructType *WrapTy,
                       int EltFld) -> Function * {
    auto &Cache = IsWrite ? WriteH : ReadH;
    auto It = Cache.find(E);
    if (It != Cache.end()) return It->second;
    Type *EP = PointerType::get(E, 0);
    Type *StreamP = WrapTy ? PointerType::get(WrapTy, 0) : EP;
    // arg order: (stream, local).
    FunctionType *FT =
        FunctionType::get(Type::getVoidTy(Ctx), {StreamP, EP}, false);
    // Name must NOT start with `__vxx_` (eraseUnimplementedMarkers strips those).
    Function *H = Function::Create(FT, GlobalValue::LinkOnceODRLinkage,
                                   IsWrite ? "hlsaxis_write"
                                           : "hlsaxis_read",
                                   &M);
    // fpga.inline pragma (NOT the LLVM alwaysinline attr): the HLS backend
    // inlines this into the kernel DURING AXIS interface analysis, so the
    // axis.pop + get_<chan>_ptr accessor calls land on the kernel's stream port
    // → 7 side channels.
    H->addFnAttr(Attribute::NoInline);  // helper is manually inlined below (after
    // wrap) so the axis.pop lands in @example BEFORE scalarizeAxisWideLoads/
    // simplifyAxisI96Body run (they gate on the fn containing llvm.fpga.axis.*).
    H->setMetadata("fpga.function.pragma", FpgaInlinePragma);
    Argument *A0 = H->getArg(0), *A1 = H->getArg(1);
    Argument *StreamArg, *LocalArg; // StreamArg = AXIS interface, LocalArg = temp
    A0->setName("stream");
    A1->setName(IsWrite ? "src" : "dst");
    StreamArg = A0;
    LocalArg = A1;
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", H);
    IRBuilder<> B(BB);
    unsigned NCh = E->getNumElements();
    // stream-class mode: reach the axis element from the wrapper. EltFld<0 → the
    // rustc bitcast form (element at offset 0); else GEP(this,0,EltFld).
    Value *StreamElem =
        !WrapTy ? (Value *)StreamArg
        : EltFld < 0 ? B.CreateBitCast(StreamArg, EP)
                     : B.CreateConstGEP2_32(WrapTy, StreamArg, 0, EltFld);
    SmallVector<Value *, 14> Args;       // stream-side first, then local-side
    for (unsigned i = 0; i < NCh; ++i)
      Args.push_back(B.CreateCall(getAccessor(E, i), {StreamElem}));
    for (unsigned i = 0; i < NCh; ++i)
      Args.push_back(B.CreateCall(getAccessor(E, i), {LocalArg}));
    Function *Intr =
        hlsrs::vxx::getOrInsertAxisPopOrPush(M, IsWrite ? "push" : "pop", fieldPtrTys(E));
    B.CreateCall(Intr, Args);
    B.CreateRetVoid();
    Cache[E] = H;
    return H;
  };

  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (F.getName().startswith("__vxx_axis_")) continue;
    SmallVector<CallInst *, 8> Pops, Pushes;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *CF = CI->getCalledFunction();
          if (!CF) continue;
          StringRef N = CF->getName();
          if (N.startswith("llvm.fpga.fifo.pop.") &&
              isNestedAxis(dyn_cast<StructType>(CI->getType())))
            Pops.push_back(CI);
          else if (N.startswith("llvm.fpga.fifo.push.") &&
                   CI->getNumArgOperands() >= 1 &&
                   isNestedAxis(
                       dyn_cast<StructType>(CI->getArgOperand(0)->getType())))
            Pushes.push_back(CI);
        }
    IRBuilder<> EB(&F.getEntryBlock(),
                   F.getEntryBlock().getFirstInsertionPt());
    SmallVector<CallInst *, 8> ToInline;  // helper calls to inline into kernel
    // Trace a stream pointer GEP(wrapper, 0, fld) back to the class.hls::stream
    // WRAPPER + element field — the stream-class arg. Returns the
    // bare pointer + WrapTy=nullptr if not a recognizable wrapper GEP (legacy).
    auto traceWrap = [&](Value *P, StructType *&WrapTy, int &Fld) -> Value * {
      WrapTy = nullptr; Fld = -1;
      return P;
    };
    for (CallInst *CI : Pops) {
      auto *E = cast<StructType>(CI->getType());
      StructType *WrapTy; int Fld;
      Value *StreamArg = traceWrap(CI->getArgOperand(0), WrapTy, Fld);
      AllocaInst *Tmp = EB.CreateAlloca(E, nullptr, "axis.rd.tmp");
      IRBuilder<> B(CI);
      // sret layout: read(out=Tmp [sret], stream=StreamArg); else (stream, dst).
      CallInst *HC =
          B.CreateCall(getHelper(E, false, nullptr, -1), {StreamArg, Tmp});
      ToInline.push_back(HC);
      Value *NV = B.CreateLoad(E, Tmp);
      CI->replaceAllUsesWith(NV);
      CI->eraseFromParent();
      Changed = true;
    }
    for (CallInst *CI : Pushes) {
      Value *Val = CI->getArgOperand(0);
      StructType *WrapTy; int Fld;
      Value *StreamArg = traceWrap(CI->getArgOperand(1), WrapTy, Fld);
      auto *E = cast<StructType>(Val->getType());
      AllocaInst *Tmp = EB.CreateAlloca(E, nullptr, "axis.wr.tmp");
      IRBuilder<> B(CI);
      B.CreateStore(Val, Tmp);
      ToInline.push_back(
          B.CreateCall(getHelper(E, true, WrapTy, Fld), {StreamArg, Tmp}));
      CI->eraseFromParent();
      Changed = true;
    }
    // Manually inline read/write into @example: keeping them as separate noinline
    // fns leaves a standalone helper that the HLS backend runs GVN/CVP on →
    // BasicAA stack-overflow SIGSEGV (this is avoided only when read/write are
    // alwaysinline and inlined AFTER GVN/CVP — a timing we can't get, since rustc
    // opt runs before the backend). Inlining removes the standalone helper
    // entirely; @example then carries the axis.pop with noalias channel args +
    // narrow distinct types, so the backend's GVN/CVP BasicAA terminates. The i96
    // chain @example picks up is cleaned by simplifyAxisI96Body (which matches
    // axis.pop-containing fns). Accessors stay noinline w/ fpga.inline.
    for (CallInst *Call : ToInline) {
        InlineFunctionInfo IFI;
        InlineFunction(*Call, IFI);
      }
  }
  if (Changed)
    vxxDbg() << "vxx: wrapped AXIS read/write in helper fns "
              "(C++-faithful axis.pop/push for side channels)\n";
  return Changed;
}

// Strip the `[0 x i32]` zero-size-array (ZSA) sandwich fields from
// `class.hls::stream<hls::axis<...>>` wrapper structs. rustc emits the Stream
// wrapper as `{ [0 x i32], <inner axis>, [0 x i32] }` (repr-C ZSA padding). For
// the array-disaggregated AXIS ports the wrapper is fully rewritten away, but
// for a kernel that keeps a *local* `hls::stream<hls::axis<T>>` scratch buffer
// (ap_hs datamover, side_channel_data), the wrapper survives to csynth as an
// alloca (`in_r.tmp`/`out_r.tmp`) and the HLS back-end rejects the `[0 x i32]`
// field with `214-309 Detected unsupported array/vector as field with size 0`.
//
// This runs in vxxLowerVitis Phase 10 — AFTER renameStreamWrapperToCanonical
// (so the wrapper carries its canonical `class.hls::stream<hls::axis<...>>`
// name) and AFTER the Phase-4 stream read/write lowering (so nothing downstream
// still relies on the 3-field padded shape). It rewrites the wrapper to a bare
// `{ <inner axis> }` and remaps every GEP field index (`i32 0, i32 1` →
// `i32 0, i32 0`) module-wide via remapStructsInModule.
//
// Scoped to `class.hls::stream<hls::axis<` only: plain `hls::stream<scalar>`
// (simple_fifos) has no struct inner and a non-AXIS `hls::stream<Struct>`
// internal DATAFLOW channel does not match the `hls::axis<` infix, so neither
// is perturbed.
bool stripAxisStreamWrapperZsa(Module &M) {
  SmallVector<StructType *, 4> Targets;
  DenseMap<StructType *, std::vector<int>> Mapping;
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (ST->isOpaque() || !ST->hasName()) continue;
    // Any `class.hls::stream<T>` wrapper whose element T is a struct — `hls::axis`
    // side-channel packets AND custom user packets (custom_side_channel_data_2's
    // `class.hls::stream<Packet>` with `[0 x i16]` ZSA). The struct-inner check
    // below excludes scalar streams (`class.hls::stream<i32>`, simple_fifos),
    // which neither carry a ZSA sandwich here nor need stripping.
    if (!ST->getName().startswith("class.hls::stream<")) continue;
    // ZSA sandwich? Exactly one real field, ≥1 zero-size `[0 x ?]` field.
    StructType *Inner = nullptr;
    int RealCount = 0;
    bool HasZSA = false;
    std::vector<int> Map(ST->getNumElements(), -1);
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *FT = ST->getElementType(i);
      if (auto *AT = dyn_cast<ArrayType>(FT))
        if (AT->getNumElements() == 0) { HasZSA = true; continue; }
      Map[i] = RealCount++;
      Inner = dyn_cast<StructType>(FT);
    }
    if (!HasZSA || RealCount != 1 || !Inner) continue;
    Targets.push_back(ST);
    Mapping[ST] = std::move(Map);
  }
  if (Targets.empty()) return false;

  DenseMap<Type *, Type *> TypeMap;
  DenseMap<StructType *, std::vector<int>> FieldMaps;
  for (StructType *ST : Targets) {
    std::string Nm = ST->getName().str();
    ST->setName(Nm + ".pad"); // free the canonical name for the replacement
    StructType *NewST = StructType::create(M.getContext(), Nm); // opaque shell
    TypeMap[ST] = NewST;
    FieldMaps[ST] = Mapping[ST];
  }
  for (StructType *ST : Targets) {
    const std::vector<int> &Map = Mapping[ST];
    SmallVector<Type *, 2> NewFields;
    for (unsigned i = 0; i < ST->getNumElements(); ++i)
      if (i < Map.size() && Map[i] != -1)
        NewFields.push_back(remapTypeRecursive(ST->getElementType(i), TypeMap,
                                               M.getContext()));
    cast<StructType>(TypeMap[ST])->setBody(NewFields, ST->isPacked());
  }
  remapStructsInModule(M, TypeMap, &FieldMaps);
  vxxDbg() << "vxx: stripped [0 x i32] ZSA from " << TypeMap.size()
           << " class.hls::stream<hls::axis<...>> wrapper(s)\n";

  // After the AXIS write has been disaggregated to the out_r side-channel ports
  // (an `_ssdm_op_IfWrite.Stream` call), the original `Stream::write` lowering
  // leaves a now-dead `volatile memcpy` that copies the packed packet into the
  // local stream-wrapper staging alloca. It is scheduled AFTER the IfWrite (the
  // staging buffer was already read to feed the disaggregated ports), so it is a
  // dead store — but `volatile` keeps DCE from removing it, and the HLS back-end
  // then rejects the 12-byte memcpy into the `hls::axis` field 0 with
  // `214-211 illegal out-of-bounds access`. Erase these redundant volatile
  // memcpys whose destination is a `class.hls::stream<hls::axis<...>>` alloca,
  // gated on the function already owning an IfWrite.Stream (so we know the
  // disaggregated write is the real one).
  unsigned NErased = 0;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    bool HasIfWrite = false;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Cal = CI->getCalledFunction())
            if (Cal->getName().startswith("_ssdm_op_IfWrite.Stream")) {
              HasIfWrite = true;
            }
    if (!HasIfWrite) continue;
    SmallVector<MemCpyInst *, 4> DeadMemcpys;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *MC = dyn_cast<MemCpyInst>(&I);
        if (!MC || !MC->isVolatile()) continue;
        Value *Dst = MC->getDest();
        while (auto *BC = dyn_cast<BitCastInst>(Dst)) Dst = BC->getOperand(0);
        while (auto *G = dyn_cast<GetElementPtrInst>(Dst)) Dst = G->getPointerOperand();
        auto *AL = dyn_cast<AllocaInst>(Dst);
        if (!AL) continue;
        auto *ST = dyn_cast<StructType>(AL->getAllocatedType());
        if (ST && ST->hasName() &&
            ST->getName().startswith("class.hls::stream<"))
          DeadMemcpys.push_back(MC);
      }
    for (MemCpyInst *MC : DeadMemcpys) { MC->eraseFromParent(); ++NErased; }
  }
  if (NErased)
    vxxDbg() << "vxx: erased " << NErased
             << " dead volatile memcpy into AXIS stream staging alloca\n";
  return true;
}

} } // namespace hlsrs::vxx
