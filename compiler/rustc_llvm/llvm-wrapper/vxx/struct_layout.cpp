//===----------------------------------------------------------------------===//
//
// struct_layout.cpp — aggregate / array-partition / reshape / memcpy-bridge / array-decay passes.
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

// C-array-decay for top kernel parameters. Rust's `*const [T; N]` lowers
// to `[N x T]*` in IR, but C `T arr[N]` parameters decay to `T*` (and
// `T arr[H][W]` to `T(*)[W]` = `[W x T]*`). Vitis HLS' m_axi rewriter
// only recognises the C-decayed shape — handed a Rust-style fixed-array
// pointer it silently drops the body.
//
// This pass strips one outer array dim from each kernel arg whose type
// is a fixed-size array pointer AND which has at least one
// `__vxx_m_axi*` marker on it (= the user applied `#pragma HLS
// INTERFACE m_axi`). It rewrites leading-zero GEPs (`gep [N x T], [N x
// T]* %p, i64 0, ...rest` → `gep T, T* %p_decayed, ...rest`);
// non-GEP uses (op-bundle args,
// stray bitcasts, etc.) get a bitcast back to the original type so
// resolveMarkerArg / downstream injectors are unaffected.
//
// Must run BEFORE `injectMAxiSideeffect`, so that pass sees the decayed
// pointee type and computes the correct `fpga.decayed.dim.hint`.
bool decayKernelArrayParams(Module &M) {
  // Identify (function, arg-idx) tuples that hold m_axi pragma markers.
  // Use a sorted vector + dedup since SmallSet<std::pair<...>> is awkward.
  SmallVector<std::pair<Function *, unsigned>, 8> Cand;
  static const char *const MAxiMarkers[] = {
      "__vxx_m_axi",
  };
  // Helper to record a candidate (function, arg-idx) — pushed if the arg
  // is a `[N x T]*` pointer-to-array. Lambda captures Cand.
  auto considerArg = [&](Argument *Arg) {
    if (!Arg) return;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) return;
    auto *AT = dyn_cast<ArrayType>(PT->getElementType());
    if (!AT) return;
    Cand.push_back({Arg->getParent(), Arg->getArgNo()});
  };

  // Detect functions with `__vxx_bind_storage` markers via the marker
  // function (xlx_bind_storage sideeffects are emitted later by
  // injectPragmaSideeffects, so we look at the original markers here).
  if (Function *BSMarker = M.getFunction("__vxx_bind_storage")) {
    SmallPtrSet<Function *, 4> BindStorageFns;
    for (User *U : BSMarker->users()) {
      if (auto *CI = dyn_cast<CallInst>(U)) {
        if (Function *F = CI->getFunction())
          BindStorageFns.insert(F);
      }
    }
    for (Function *F : BindStorageFns) {
      if (F->isDeclaration()) continue;
      F->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
      for (Argument &A : F->args()) {
        A.removeAttr(Attribute::NoCapture);
        A.removeAttr(Attribute::ReadOnly);
        A.removeAttr(Attribute::Alignment);
        A.removeAttr(Attribute::Dereferenceable);
        A.removeAttr(Attribute::DereferenceableOrNull);
      }
    }
  }
  // Also trigger decay for args with `__vxx_array_partition` markers —
  // partition'd kernel args must be decayed to `T*` with
  // `fpga.decayed.dim.hint`, then have the partition pragma applied. Without
  // the decay, the cosim TB's wrapper sees the wrong arg shape and
  // produces ABI-mismatched stubs (causing cosim SIGSEGV / hang).
  //
  // ADDITIONAL: rustc often already lowers `&[T; N]` to `T* dereferenceable(N*sizeof(T))`
  // directly (no `[N x T]*` intermediate). In that case we can't decay
  // further, but we still need `fpga.decayed.dim.hint` so Vitis HLS knows
  // the original dim count for complete partition into per-element ports.
  // Stamp the hint directly here from dereferenceable_bytes / pointee size.
  if (Function *APart = M.getFunction("__vxx_array_partition")) {
    for (User *U : APart->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg) continue;
      auto *PT = dyn_cast<PointerType>(Arg->getType());
      if (!PT) continue;
      if (auto *AT = dyn_cast<ArrayType>(PT->getElementType())) {
        (void)AT;
        Cand.push_back({Arg->getParent(), Arg->getArgNo()});
        continue;
      }
      // Pre-decayed `T*` form (rustc already lowered `&[T;N]` straight to
      // `T* dereferenceable(N*sizeof(T))`): stamp dim.hint inline so Vitis
      // HLS can complete-partition into per-element ports without a decay
      // step.
      Function *AF = Arg->getParent();
      if (AF->getAttributes().hasParamAttr(Arg->getArgNo(),
              "fpga.decayed.dim.hint")) continue;
      uint64_t Deref = Arg->getDereferenceableBytes();
      if (!Deref) continue;
      Type *Pointee = PT->getElementType();
      if (!Pointee->isSized()) continue;
      uint64_t ElemBytes = M.getDataLayout().getTypeAllocSize(Pointee);
      if (ElemBytes == 0 || Deref % ElemBytes != 0) continue;
      uint64_t Dim = Deref / ElemBytes;
      if (Dim <= 1) continue;
      Arg->addAttr(Attribute::get(M.getContext(),
          "fpga.decayed.dim.hint", std::to_string(Dim)));
    }
  }
  // AXIS disagg markers (axis_4ch / axis_user / axis_7ch) — also need
  // their channel args decayed from `[N x iN]*` to `iN*` so HLS sees them
  // as scalar stream ptrs (not BRAM array ptrs). Without this, the
  // explicit SpecInterface("axis") call conflicts with HLS's default
  // ap_memory inference → XFORM 203-801 "more than one interface mode".
  // KEEP the multi-channel AXIS per-channel args as `[N x T]*` (do NOT
  // decay). Decayed scalar `iN*` makes the cosim TB classify each channel
  // as hls::sim::Register; the array `[N x T]*` shape makes the cosim TB
  // use A2Stream (a real stream feeder), matching the single-axis path.
  // The 203-801 ap_memory/axis conflict that the decay used to avoid is
  // instead handled in emitAxisDisaggSpec by emitting an `xlx_axis`
  // op-bundle (array-compatible) instead of SpecInterface(axis).
  // AXIS top-arg (`__vxx_axis(arr)`) decay to addrspace 0 T* would follow
  // the same path as multi-channel disagg markers (axis_4ch/user/7ch).
  // It is DISABLED: decay breaks the cosim TB-side `using_axis_array_stream_*`
  // cosim (the cosim TB expects [N x T]* shape for TV file capture; T* decay
  // produces the wrong rtl.example.autotvout_*.dat layout → "Error on TV file"
  // 212-361). Multi-channel disagg markers (axis_4ch/user/7ch) still get
  // decay below — they need it and don't conflict with cosim TB capture
  // (they explicitly split sub-channel args).
  SmallPtrSet<Argument *, 8> AxisTopArgs;  // unused — addrspace stays 0
  (void)AxisTopArgs;
  // if (Function *AxisM = M.getFunction("__vxx_axis")) {
  //   for (User *U : AxisM->users()) {
  //     auto *CI = dyn_cast<CallInst>(U);
  //     if (!CI || CI->arg_size() < 1) continue;
  //     Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
  //     if (!Arg) continue;
  //     auto *PT = dyn_cast<PointerType>(Arg->getType());
  //     if (!PT) continue;
  //     if (!isa<ArrayType>(PT->getElementType())) continue;
  //     considerArg(Arg);  // decay [N x iN]* → iN* (addrspace 0, default)
  //   }
  // }
  // ap_fifo decay: convert `[N x T]*` → `T*`.
  // Combined with SpecStream + SpecInterface(ap_fifo) emit in injectApFifo,
  // this prevents the HLS backend from adding ap_auto (which causes SYN 201-504).
  // __vxx_ap_fifo sig: (ptr, register: u32, name: *const u8, name_len: u32,
  //                     depth: i64) — depth is arg index 4.
  if (Function *FifoMarker = M.getFunction("__vxx_ap_fifo")) {
    for (User *U : FifoMarker->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      if (CI->getNumArgOperands() < 5) continue;
      auto *DepthC = dyn_cast<ConstantInt>(CI->getArgOperand(4));
      if (!DepthC) continue;
      int64_t Depth = DepthC->getSExtValue();
      if (Depth <= 0) continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg) continue;
      auto *PT = dyn_cast<PointerType>(Arg->getType());
      if (!PT) continue;
      auto *AT = dyn_cast<ArrayType>(PT->getElementType());
      if (!AT) continue;
      Cand.push_back({Arg->getParent(), Arg->getArgNo()});
    }
  }

  for (const char *MN : MAxiMarkers) {
    Function *F = M.getFunction(MN);
    if (!F)
      continue;
    for (User *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg)
        continue;
      auto *PT = dyn_cast<PointerType>(Arg->getType());
      if (!PT)
        continue;
      auto *AT = dyn_cast<ArrayType>(PT->getElementType());
      if (!AT)
        continue;
      // Decay 2D+ arrays unconditionally (pointee is itself an array).
      // Decay 1D arrays only when the marker carries an explicit depth
      // — cosim TB sizes the buffer from the depth arg, so `[N x T]*` →
      // `T*` doesn't lose information. With no depth (`-1` sentinel)
      // cosim TB falls back to inferring from the array type, so 1D must
      // be preserved — decaying would give nbytes=sizeof(T) underrun
      // and SIGSEGV during DUMP_OUTPUTS.
      //
      // The 1D-with-depth decay is what unlocks Vitis HLS' m_axi
      // rewriter: it converts `T*` + s_axilite address-register pair
      // → `T addrspace(1)* %gmem` + `i64 %arg_r`, putting the function
      // into `ap_ctrl_hs` (cosim-friendly) instead of `ap_ctrl_none`
      // (cosim 212-345 reject for non-streaming designs).
      if (!isa<ArrayType>(AT->getElementType())) {
        // 1D — require explicit depth on the marker (3rd arg of
        // __vxx_m_axi(arr, depth, ...) and friends; arg index 1).
        if (CI->getNumArgOperands() < 2)
          continue;
        auto *DepthC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
        if (!DepthC)
          continue;
        uint64_t Depth = DepthC->getZExtValue();
        if (Depth == 0 || Depth == (uint64_t)-1 ||
            Depth == (uint64_t)(uint32_t)-1) {
          // The depth clause is optional now — injectMAxiSideeffect derives
          // the explicit element count from the pointee array when the
          // marker carries the "unset" sentinel. Mirror that derivation
          // here so the decay still fires: an UN-decayed 1D m_axi arg
          // sends Vitis's m_axi rewriter into a LatencyEstimator
          // null-deref (SIGSEGV right after "Finished scheduling" —
          // aggregation_of_nested_structs / maxi_cache_conditional /
          // auto_burst_inference_failure).
          Depth = AT->getNumElements();
          if (Depth == 0)
            continue;
          // Write the derived count back into the marker: this pass strips
          // the array type, so injectMAxiSideeffect's own pointee-based
          // derivation would only see the decayed scalar (depth 1).
          CI->setArgOperand(
              1, ConstantInt::get(DepthC->getType(), Depth));
        }
      }
      Cand.push_back({Arg->getParent(), Arg->getArgNo()});
    }
  }
  if (Cand.empty())
    return false;
  std::sort(Cand.begin(), Cand.end());
  Cand.erase(std::unique(Cand.begin(), Cand.end()), Cand.end());

  DenseMap<Function *, SmallVector<unsigned, 4>> ByFn;
  for (auto &P : Cand)
    ByFn[P.first].push_back(P.second);

  bool Changed = false;
  for (auto &KV : ByFn) {
    Function *F = KV.first;
    if (F->isDeclaration())
      continue;
    SmallVector<unsigned, 4> &Idxs = KV.second;

    LLVMContext &Ctx = F->getContext();
    FunctionType *OldFT = F->getFunctionType();
    unsigned NParams = OldFT->getNumParams();

    // Pre-collect which (to-be-)decayed args have __vxx_array_partition
    // markers — those need `fpga.decayed.dim.hint` so Vitis HLS can split
    // into per-element ports. m_axi-only args skip this;
    // their dim.hint is added later by `injectMAxiSideeffect`.
    //
    // MUST run BEFORE the BasicBlock splice below: after splice the marker
    // calls' `getFunction()` returns NewF (BBs were moved), so a `!= F` check
    // would skip every marker call → PartitionArgs ends empty → dim.hint
    // never stamped → Vitis HLS can't recover the original dim count for
    // complete-partition into per-element ports.
    SmallSet<unsigned, 4> PartitionArgs;
    if (Function *APart = M.getFunction("__vxx_array_partition")) {
      for (User *U : APart->users()) {
        auto *PCI = dyn_cast<CallInst>(U);
        if (!PCI || PCI->getFunction() != F) continue;
        Argument *A = resolveMarkerArg(PCI->getArgOperand(0));
        if (A && A->getParent() == F) PartitionArgs.insert(A->getArgNo());
      }
    }
    // AXIS disagg markers: we want the args decayed (so HLS sees scalar
    // stream ptr, not BRAM array ptr) but WITHOUT `fpga.decayed.dim.hint`
    // — the hint triggers HLS's ap_memory (BRAM) inference, conflicting
    // with our explicit SpecInterface("axis") and causing XFORM 203-801.
    // C++ AXIS args have no dim.hint either; trans count is inferred
    // from the body's pop/push intrinsic count.
    // → Track AXIS args separately and use them to SUPPRESS the dim.hint
    //   path in the per-arg decay loop below.

    SmallVector<Type *, 8> NewParamTys;
    SmallVector<bool, 8> Decayed(NParams, false);
    SmallVector<bool, 8> WrappedAS128(NParams, false);
    // Track per-arg target address space (default = old AS, 128 for AXIS top).
    SmallVector<unsigned, 8> TargetAS(NParams, 0);
    {
      auto AB = F->arg_begin();
      for (unsigned i = 0; i < NParams; ++i, ++AB) {
        if (AxisTopArgs.count(&*AB)) TargetAS[i] = 128;
      }
    }
    for (unsigned i = 0; i < NParams; ++i) {
      Type *Old = OldFT->getParamType(i);
      bool DecayThis = std::find(Idxs.begin(), Idxs.end(), i) != Idxs.end();
      if (!DecayThis) {
        NewParamTys.push_back(Old);
        continue;
      }
      auto *PT = cast<PointerType>(Old);
      auto *AT = cast<ArrayType>(PT->getElementType());
      Type *NewElem = AT->getElementType();
      unsigned NewAS = (TargetAS[i] != 0) ? TargetAS[i] : PT->getAddressSpace();
      NewParamTys.push_back(PointerType::get(NewElem, NewAS));
      Decayed[i] = true;
      if (NewAS == 128) WrappedAS128[i] = true;
    }

    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParamTys, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       F->getName() + ".decay_tmp",
                                       F->getParent());
    NewF->copyAttributesFrom(F);
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());

    // Carry over function-level metadata (notably `!fpga.function.pragma`,
    // which the HLS backend uses to identify per-function pragmas
    // — without it, the m_axi rewriter ignores the function's body).
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs)
      NewF->setMetadata(P.first, P.second);

    // Carry over per-arg attributes for the params we did NOT decay.
    // For decayed params, drop arg attrs entirely (alignment / dereferenceable
    // were sized for the old pointee and would be wrong post-decay; noalias
    // / nocapture would be safe but we keep things minimal).
    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (!Decayed[i]) {
        AttributeSet AS = F->getAttributes().getParamAttributes(i);
        for (Attribute A : AS)
          NewA->addAttr(A);
      } else if (PartitionArgs.count(i)) {
        // Partition'd decayed arg: stamp dim.hint so Vitis HLS recovers
        // original dim count and splits into per-element ports.
        //
        // First drop stale attrs inherited via copyAttributesFrom (the
        // OLD `[N x T]*` had dereferenceable=N*sizeof(T), align=..., etc.
        // — after decay each per-element port is sizeof(T), so the old
        // bounds are wrong and Vitis HLS's DetachIRWrapper pass crashes
        // when it tries to honour them post-partition).
        NewA->removeAttr(Attribute::NoAlias);
        NewA->removeAttr(Attribute::ReadOnly);
        NewA->removeAttr(Attribute::WriteOnly);
        NewA->removeAttr(Attribute::ReadNone);
        NewA->removeAttr(Attribute::NoCapture);
        NewA->removeAttr(Attribute::Alignment);
        NewA->removeAttr(Attribute::Dereferenceable);
        NewA->removeAttr(Attribute::DereferenceableOrNull);
        auto *OldPT = cast<PointerType>(OldFT->getParamType(i));
        auto *OldAT = cast<ArrayType>(OldPT->getElementType());
        uint64_t OuterDim = OldAT->getNumElements();
        NewA->addAttr(Attribute::get(M.getContext(),
            "fpga.decayed.dim.hint", std::to_string(OuterDim)));
      } else {
        // Plain decayed arg (no partition): strip stale dereferenceable/align
        // inherited from OLD `[N x T]*` (causes HLS ap_auto add).
        // Also stamp `fpga.signal_name` attr — the HLS backend's
        // default-interface pass checks for this; if present, skips ap_auto auto-add.
        // SKIP signal_name for addrspace(128) AXIS args — AXIS args carry no
        // signal_name attr, just `noalias`.
        NewA->removeAttr(Attribute::Alignment);
        NewA->removeAttr(Attribute::Dereferenceable);
        NewA->removeAttr(Attribute::DereferenceableOrNull);
        NewA->removeAttr(Attribute::ReadOnly);
        NewA->removeAttr(Attribute::WriteOnly);
        NewA->removeAttr(Attribute::ReadNone);
        NewA->removeAttr(Attribute::NoCapture);
        if (!WrappedAS128[i]) {
          NewA->addAttr(Attribute::get(M.getContext(),
              "fpga.signal_name", NewA->getName()));
        } else {
          // AXIS addrspace(128) args carry a `noalias` attr;
          // set it so HLS's stream detection has the same hint.
          NewA->addAttr(Attribute::NoAlias);
        }
      }
    }

    // Bridge old uses to new args.
    BasicBlock &Entry = NewF->getEntryBlock();
    Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
    IRBuilder<> B(InsertBefore);
    OldA = F->arg_begin();
    NewA = NewF->arg_begin();
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      if (!Decayed[i]) {
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Bitcast the new (decayed) arg back to the old type. This is a
      // catch-all so any use that we don't optimise (op-bundle args,
      // stray bitcasts, helper calls) sees the original type and stays
      // type-correct. For addrspace-128 (AXIS) decayed args, addrspacecast
      // first then bitcast (LLVM disallows direct bitcast across AS).
      Value *Cast;
      if (WrappedAS128[i]) {
        // addrspacecast T addrspace(128)* → T addrspace(0)*, then bitcast
        // to OldA's full pointee type.
        auto *DecayedNoAS = cast<PointerType>(NewA->getType());
        Type *DecayedElem = DecayedNoAS->getElementType();
        Type *AS0Ptr = PointerType::get(DecayedElem, 0);
        Value *AsAS0 = B.CreateAddrSpaceCast(&*NewA, AS0Ptr,
                                              OldA->getName() + ".as0");
        Cast = B.CreateBitCast(AsAS0, OldA->getType(),
                                OldA->getName() + ".decayed_cast");
      } else {
        Cast = B.CreateBitCast(&*NewA, OldA->getType(),
                                OldA->getName() + ".decayed_cast");
      }
      OldA->replaceAllUsesWith(Cast);

      // Optimise the common case: GEPs that start with a constant 0
      // index. These are the canonical Rust pointer-to-fixed-array
      // dereference (`gep [N x T], [N x T]* %p, i64 0, i64 idx, ...`)
      // and they map cleanly to the C-decayed `gep T, T* %p_decayed,
      // i64 idx, ...` form. Rewriting these is what actually lets
      // Vitis HLS' m_axi rewriter find the access pattern.
      //
      // SKIP for addrspace-128 (AXIS): NewA is addrspace 128, body
      // accesses are addrspace 0 via Cast — rewriting GEPs to use
      // NewA directly would mix address spaces (body load/store on
      // addrspace 0 wouldn't accept addrspace 128 GEP result).
      if (WrappedAS128[i]) continue;
      SmallVector<User *, 16> CastUsers(Cast->users());
      for (User *U : CastUsers) {
        auto *GEP = dyn_cast<GetElementPtrInst>(U);
        if (!GEP)
          continue;
        if (GEP->getPointerOperand() != Cast)
          continue;
        if (GEP->getNumOperands() < 2)
          continue;
        auto *Idx0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
        if (!Idx0 || !Idx0->isZero())
          continue;
        SmallVector<Value *, 4> NewIdxs;
        for (unsigned j = 2; j < GEP->getNumOperands(); ++j)
          NewIdxs.push_back(GEP->getOperand(j));
        Type *NewSrcElem = cast<PointerType>(NewA->getType())->getElementType();
        if (NewIdxs.empty()) {
          // GEP was just stripping the leading 0 — equivalent to NewA
          // itself when types match (they do, by construction).
          if (GEP->getType() == NewA->getType()) {
            GEP->replaceAllUsesWith(&*NewA);
            GEP->eraseFromParent();
          }
        } else {
          auto *NewGEP = GetElementPtrInst::Create(
              NewSrcElem, &*NewA, NewIdxs, GEP->getName(), GEP);
          NewGEP->setIsInBounds(GEP->isInBounds());
          GEP->replaceAllUsesWith(NewGEP);
          GEP->eraseFromParent();
        }
      }
      // The bitcast may now be dead — leave for the late DCE sweep at
      // the end of VXXPrep to clean up.
      (void)Ctx;
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: decayed C-array kernel param(s)\n";
  return Changed;
}

// Helper: recursively walk struct's leaf fields (non-padding primitive types)
// and collect their (compact-byte offset, type, gep_path) info.
namespace {
struct LeafField {
  Type *Ty;
  uint64_t ByteOffset;
  SmallVector<unsigned, 4> GepPath;
  // Bit-compact layout (compact=bit, the pragma's default): fields are packed
  // at their EFFECTIVE widths with no padding. EffBits differs from the LLVM
  // type width for Rust `bool` (stored as i8, logically 1 bit — clang packs a
  // C++ `bool` member to 1 bit here, e.g. nested_structs' 32+32+32+1 = i97).
  uint64_t BitOffset = 0;
  uint64_t EffBits = 0;
};

bool isPaddingArray(Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    if (auto *ET = dyn_cast<IntegerType>(AT->getElementType()))
      return ET->getBitWidth() == 8;
  return false;
}

// Returns false when the struct holds a field shape this pass cannot pack
// (e.g. a non-padding array member) — the caller must then leave the arg
// unpacked rather than emit a layout that silently drops fields.
bool collectLeafFields(StructType *ST, uint64_t &CurOffsetBytes,
                              SmallVector<unsigned, 4> &Path,
                              SmallVectorImpl<LeafField> &Out) {
  for (unsigned i = 0; i < ST->getNumElements(); ++i) {
    Type *FT = ST->getElementType(i);
    Path.push_back(i);
    bool Ok = true;
    if (auto *Sub = dyn_cast<StructType>(FT)) {
      Ok = collectLeafFields(Sub, CurOffsetBytes, Path, Out);
    } else if (isPaddingArray(FT)) {
      // skip padding
    } else if (FT->isIntegerTy() || FT->isFloatingPointTy()) {
      uint64_t bits = FT->getPrimitiveSizeInBits();
      uint64_t bytes = (bits + 7) / 8;
      LeafField L{FT, CurOffsetBytes, SmallVector<unsigned, 4>(Path.begin(), Path.end())};
      Out.push_back(L);
      CurOffsetBytes += bytes;
    } else if (auto *AT = dyn_cast<ArrayType>(FT)) {
      Ok = AT->getNumElements() == 0; // zero-sized markers are ignorable
    } else {
      Ok = false;
    }
    Path.pop_back();
    if (!Ok) return false;
  }
  return true;
}

// A Rust `bool` struct member arrives as i8; C++'s bit-compact aggregate
// packs bool to 1 bit. Confirm the i8 leaf is bool-LIKE from its uses: every
// load through this arg+field path must feed only `and X, 1` / `trunc to i1`
// / icmp — the exact shapes rustc emits for bool reads. Unconfirmed i8 (a
// real u8, or a field never read) keeps its full 8 bits.
bool i8LeafIsBoolLike(Argument *Arg, const LeafField &L) {
  bool SawLoad = false;
  for (User *U : Arg->users()) {
    auto *GEP = dyn_cast<GetElementPtrInst>(U);
    if (!GEP) continue;
    // GEP shape over [N x %S]*: (0, elemIdx, <field path...>).
    if (GEP->getNumOperands() != 3 + L.GepPath.size()) continue;
    bool PathMatch = true;
    for (size_t i = 0; i < L.GepPath.size(); ++i) {
      auto *C = dyn_cast<ConstantInt>(GEP->getOperand(3 + i));
      if (!C || C->getZExtValue() != L.GepPath[i]) { PathMatch = false; break; }
    }
    if (!PathMatch) continue;
    for (User *GU : GEP->users()) {
      auto *LD = dyn_cast<LoadInst>(GU);
      if (!LD) return false;
      SawLoad = true;
      for (User *LU : LD->users()) {
        if (auto *BO = dyn_cast<BinaryOperator>(LU)) {
          auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (BO->getOpcode() == Instruction::And && C && C->getZExtValue() == 1)
            continue;
          return false;
        }
        if (auto *TR = dyn_cast<TruncInst>(LU)) {
          if (TR->getDestTy()->isIntegerTy(1)) continue;
          return false;
        }
        if (isa<ICmpInst>(LU)) continue;
        return false;
      }
    }
  }
  return SawLoad;
}
} // anon namespace

// Pack `[N x %S]*` kernel args to `[N x iK]*` (compact-byte) when the arg
// has `__vxx_aggregate(ptr, compact=1)` marker. Required for HLS to see
// int-typed array (not struct array) which:
// 1. Avoids HLS's struct-arg auto ap_memory injection (203-801 conflict
//    with explicit bram marker on the same arg)
// 2. Produces single-port (vs dual-port) memory ports matching C++
//
// C++ source `#pragma HLS aggregate compact=byte` → OSS Clang packs struct
// to int at frontend. Rust IR keeps `{ T1, T2, ... }` struct; we do the
// pack here at VXXPrep level.
//
// Method: function clone + temp alloca bridge (same pattern as
// disaggCompletePartitionKernelSig). At entry: load each packed iK,
// unpack into local struct fields. Body: VMap[old_arg] = local_alloca.
// At returns: pack local struct back to iK, store to new arg.
bool packAggregateByteKernelSig(Module &M) {
  Function *AggMarker = M.getFunction("__vxx_aggregate");
  if (!AggMarker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);

  // Pre-scan: which args have bram mode (preserve mode info across
  // function-clone in packing). `injectMemoryInterface` runs BEFORE this
  // pass and consumes the `__vxx_bram` marker, but emits
  // `SpecInterface(arg, "bram", ...)` call which we can detect.
  DenseSet<Argument *> BramArgs;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (Instruction &I : F.getEntryBlock()) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->getCalledFunction()) continue;
      if (CI->getCalledFunction()->getName() != "_ssdm_op_SpecInterface") continue;
      if (CI->arg_size() < 2) continue;
      // arg 1 is mode name (i8* to global "bram"\0 or "ap_memory"\0 etc.)
      Value *ModeArg = CI->getArgOperand(1);
      while (auto *BC = dyn_cast<BitCastOperator>(ModeArg)) ModeArg = BC->getOperand(0);
      if (auto *CE = dyn_cast<ConstantExpr>(ModeArg))
        if (CE->getOpcode() == Instruction::GetElementPtr)
          ModeArg = CE->getOperand(0);
      auto *GV = dyn_cast<GlobalVariable>(ModeArg);
      if (!GV || !GV->hasInitializer()) continue;
      auto *Init = dyn_cast<ConstantDataArray>(GV->getInitializer());
      if (!Init || !Init->isString()) continue;
      StringRef ModeStr = Init->getAsCString();
      if (ModeStr != "bram") continue;
      // arg 0 is the kernel arg
      Value *V = CI->getArgOperand(0);
      while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
      if (auto *A = dyn_cast<Argument>(V)) BramArgs.insert(A);
    }
  }

  // Collect marker args: compact=byte (compact == 1) packs at byte offsets,
  // anything else keeps the pragma default = bit compaction.
  DenseSet<Argument *> Targets;
  DenseSet<Argument *> BitModeArgs;
  SmallVector<CallInst *, 8> MarkersToDrop;
  for (User *U : AggMarker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 2) continue;
    auto *CompactC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!CompactC) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Function *F = Arg->getParent();
    if (!F->hasFnAttribute("fpga.top.func")) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    auto *AT = dyn_cast<ArrayType>(PT->getElementType());
    if (!AT) continue;
    auto *ST = dyn_cast<StructType>(AT->getElementType());
    if (!ST) continue;
    // compact=auto (2)/none (3) keep the struct shape — HLS decides.
    if (CompactC->getZExtValue() >= 2) continue;
    Targets.insert(Arg);
    if (CompactC->getZExtValue() == 0) BitModeArgs.insert(Arg);
    MarkersToDrop.push_back(CI);
  }
  // C++ default-aggregates every struct port even WITHOUT a pragma (bit
  // compaction): nested_structs' un-pragma'd `b` is [8 x i97] at the pp
  // stage. Mirror that for sibling struct-array args of functions that
  // already aggregate something, when the arg has no interface spec of its
  // own (no SpecInterface reference, no op-bundle) — i.e. pure default
  // ap_memory ports. Scoping to aggregate-using functions keeps every other
  // example's struct ports untouched.
  {
    SmallPtrSet<Function *, 4> AggFns;
    for (Argument *A : Targets) AggFns.insert(A->getParent());
    for (Function *F : AggFns) {
      for (Argument &A : F->args()) {
        if (Targets.count(&A)) continue;
        auto *PT = dyn_cast<PointerType>(A.getType());
        if (!PT) continue;
        auto *AT = dyn_cast<ArrayType>(PT->getElementType());
        if (!AT) continue;
        if (!isa<StructType>(AT->getElementType())) continue;
        // This pass runs AFTER the interface-spec passes, so a plain default
        // port already carries an auto-emitted SpecInterface (ap_auto /
        // ap_memory). Those are exactly the ports C++ default-aggregates;
        // only a non-memory interface (m_axi / axis / fifo bundle or spec)
        // blocks the default pack.
        bool Blocking = false;
        for (User *U : A.users()) {
          auto *CI = dyn_cast<CallInst>(U);
          if (!CI) continue;
          if (CI->getNumOperandBundles() > 0) {
            StringRef BN = CI->getOperandBundleAt(0).getTagName();
            if (BN != "xlx_ap_memory" && BN != "xlx_bram") { Blocking = true; break; }
            continue;
          }
          Function *CF = CI->getCalledFunction();
          if (!CF || CF->getName() != "_ssdm_op_SpecInterface") continue;
          if (CI->arg_size() < 2) continue;
          Value *ModeArg = CI->getArgOperand(1);
          while (auto *BC = dyn_cast<BitCastOperator>(ModeArg)) ModeArg = BC->getOperand(0);
          if (auto *CE = dyn_cast<ConstantExpr>(ModeArg))
            if (CE->getOpcode() == Instruction::GetElementPtr)
              ModeArg = CE->getOperand(0);
          auto *GV = dyn_cast<GlobalVariable>(ModeArg);
          if (!GV || !GV->hasInitializer()) continue;
          auto *Init = dyn_cast<ConstantDataArray>(GV->getInitializer());
          if (!Init || !Init->isString()) continue;
          StringRef Mode = Init->getAsCString();
          if (Mode != "ap_memory" && Mode != "ap_auto" && Mode != "bram") {
            Blocking = true;
            break;
          }
        }
        if (Blocking) continue;
        Targets.insert(&A);
        BitModeArgs.insert(&A);
      }
    }
  }
  if (Targets.empty()) return false;

  // Group by parent function
  DenseMap<Function *, SmallVector<Argument *, 4>> ByFn;
  for (Argument *A : Targets) ByFn[A->getParent()].push_back(A);

  bool Changed = false;
  for (auto &Pair : ByFn) {
    Function *Old = Pair.first;
    auto &PackArgs = Pair.second;
    std::sort(PackArgs.begin(), PackArgs.end(),
              [](Argument *A, Argument *B) { return A->getArgNo() < B->getArgNo(); });
    DenseSet<Argument *> PackSet(PackArgs.begin(), PackArgs.end());

    // Compute new param types + per-arg layout
    struct PackInfo {
      bool IsPack;
      bool BitMode;
      uint64_t N;
      StructType *OrigST;
      uint64_t PackedBits;
      IntegerType *PackedTy;
      SmallVector<LeafField, 8> Leaves;
    };
    SmallVector<PackInfo, 8> Infos;
    SmallVector<Type *, 16> NewParams;
    for (Argument &A : Old->args()) {
      PackInfo PI{};
      PI.IsPack = false;
      PI.BitMode = BitModeArgs.count(&A) != 0;
      if (PackSet.count(&A)) {
        auto *PT = cast<PointerType>(A.getType());
        auto *AT = cast<ArrayType>(PT->getElementType());
        auto *ST = cast<StructType>(AT->getElementType());
        PI.N = AT->getNumElements();
        PI.OrigST = ST;
        uint64_t Offset = 0;
        SmallVector<unsigned, 4> Path;
        bool Complete = collectLeafFields(ST, Offset, Path, PI.Leaves);
        uint64_t TotalBits = Complete ? Offset * 8 : 0;
        if (Complete && PI.BitMode) {
          // Bit compaction: pack each leaf at its effective width.
          uint64_t Bit = 0;
          for (LeafField &L : PI.Leaves) {
            uint64_t W = L.Ty->getPrimitiveSizeInBits();
            if (W == 8 && L.Ty->isIntegerTy() && i8LeafIsBoolLike(&A, L))
              W = 1;
            L.EffBits = W;
            L.BitOffset = Bit;
            Bit += W;
          }
          TotalBits = Bit;
        }
        if (TotalBits > 0 && TotalBits <= 8192) {  // sanity
          PI.IsPack = true;
          PI.PackedBits = TotalBits;
          PI.PackedTy = IntegerType::get(Ctx, PI.PackedBits);
          NewParams.push_back(PointerType::get(
              ArrayType::get(PI.PackedTy, PI.N), 0));
        } else {
          NewParams.push_back(A.getType());
        }
      } else {
        NewParams.push_back(A.getType());
      }
      Infos.push_back(PI);
    }

    FunctionType *NewFT = FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF = Function::Create(NewFT, Old->getLinkage(),
                                       Old->getAddressSpace(),
                                       Old->getName() + ".agg_pack", &M);
    NewF->copyAttributesFrom(Old);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "agg.entry", NewF);
    IRBuilder<> EB(EntryBB);

    ValueToValueMapTy VMap;
    auto NewArgIt = NewF->arg_begin();
    // For each packed arg: alloca [N x %S], pre-unpack from packed iK,
    // and store back to packed iK at returns. VMap old arg → alloca.
    struct AllocaInfo {
      AllocaInst *Local;
      Argument *NewArg;
      PackInfo *PI;
      bool IsBram;
    };
    SmallVector<AllocaInfo, 4> Allocas;
    for (size_t idx = 0; idx < Infos.size(); ++idx) {
      PackInfo &PI = Infos[idx];
      Argument &OldA = *(Old->arg_begin() + idx);
      if (!PI.IsPack) {
        NewArgIt->setName(OldA.getName());
        if (OldA.hasAttribute(Attribute::NoAlias))
          NewArgIt->addAttr(Attribute::NoAlias);
        VMap[&OldA] = &*NewArgIt;
        ++NewArgIt;
        continue;
      }
      // Packed: new arg = [N x iK]*
      NewArgIt->setName(OldA.getName());
      if (OldA.hasAttribute(Attribute::NoAlias))
        NewArgIt->addAttr(Attribute::NoAlias);
      // Explicitly STRIP nocapture (copyAttributesFrom(Old) above copied it
      // from the old [N x %S]* arg). Rust's nocapture on packed-int array args
      // causes the HLS backend to auto-add ap_memory SpecInterface even when
      // an explicit bram/ap_memory marker is present (203-801 conflict on
      // aggregation_of_nested_structs).
      NewF->removeParamAttr(NewArgIt->getArgNo(), Attribute::NoCapture);
      Argument *PackedArg = &*NewArgIt;
      ++NewArgIt;
      // Local alloca = [N x %S]
      ArrayType *LocalArrTy = ArrayType::get(PI.OrigST, PI.N);
      AllocaInst *Local = EB.CreateAlloca(LocalArrTy, nullptr,
                                          OldA.getName() + ".unpack");
      VMap[&OldA] = Local;
      Allocas.push_back({Local, PackedArg, &PI, BramArgs.count(&OldA) != 0});
      // Pre-unpack: for i in 0..N, load packed[i], extract each leaf field, store into local[i].field_path
      for (uint64_t i = 0; i < PI.N; ++i) {
        Value *PackedGEP = EB.CreateInBoundsGEP(
            cast<ArrayType>(PackedArg->getType()->getPointerElementType()),
            PackedArg,
            {ConstantInt::get(I64, 0), ConstantInt::get(I64, i)});
        LoadInst *PackedVal = EB.CreateLoad(PI.PackedTy, PackedGEP);
        for (const LeafField &L : PI.Leaves) {
          uint64_t ShiftBits = PI.BitMode ? L.BitOffset : L.ByteOffset * 8;
          Value *Shifted = ShiftBits
              ? EB.CreateLShr(PackedVal, ConstantInt::get(PI.PackedTy, ShiftBits))
              : (Value*)PackedVal;
          // Trunc to the packed effective width, then widen back to the LLVM
          // field width (bool: 1 packed bit zext'd into the i8 store type).
          uint64_t LBits = L.Ty->getPrimitiveSizeInBits();
          uint64_t EBits = PI.BitMode ? L.EffBits : LBits;
          Value *FieldVal;
          if (EBits == PI.PackedBits) {
            FieldVal = Shifted;
          } else {
            FieldVal = EB.CreateTrunc(Shifted, IntegerType::get(Ctx, EBits));
            if (EBits < LBits)
              FieldVal = EB.CreateZExt(FieldVal, IntegerType::get(Ctx, LBits));
            // If L.Ty is float/double, bitcast
            if (L.Ty->isFloatingPointTy()) {
              FieldVal = EB.CreateBitCast(FieldVal, L.Ty);
            }
          }
          // GEP into local[i].field_path
          SmallVector<Value*, 6> GepIdx;
          GepIdx.push_back(ConstantInt::get(I64, 0));      // array of [N x S]
          GepIdx.push_back(ConstantInt::get(I64, i));       // S index
          for (unsigned p : L.GepPath) GepIdx.push_back(ConstantInt::get(I32, p));
          Value *LocalGEP = EB.CreateInBoundsGEP(LocalArrTy, Local, GepIdx);
          EB.CreateStore(FieldVal, LocalGEP);
        }
      }
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap, /*ModuleLevelChanges=*/false, Returns);

    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF) {
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    }
    if (!ClonedEntry) { NewF->eraseFromParent(); continue; }
    EB.CreateBr(ClonedEntry);

    // At each return: re-pack local → packed iK → store to new arg
    for (ReturnInst *RI : Returns) {
      IRBuilder<> RB(RI);
      for (auto &AI : Allocas) {
        Argument *PackedArg = AI.NewArg;
        AllocaInst *Local = AI.Local;
        PackInfo *PI = AI.PI;
        ArrayType *LocalArrTy = ArrayType::get(PI->OrigST, PI->N);
        for (uint64_t i = 0; i < PI->N; ++i) {
          Value *Packed = ConstantInt::get(PI->PackedTy, 0);
          for (const LeafField &L : PI->Leaves) {
            // GEP local[i].field
            SmallVector<Value*, 6> GepIdx;
            GepIdx.push_back(ConstantInt::get(I64, 0));
            GepIdx.push_back(ConstantInt::get(I64, i));
            for (unsigned p : L.GepPath) GepIdx.push_back(ConstantInt::get(I32, p));
            Value *LocalGEP = RB.CreateInBoundsGEP(LocalArrTy, Local, GepIdx);
            Value *FieldVal = RB.CreateLoad(L.Ty, LocalGEP);
            // bitcast float → int if needed
            uint64_t LBits = L.Ty->getPrimitiveSizeInBits();
            if (L.Ty->isFloatingPointTy()) {
              FieldVal = RB.CreateBitCast(FieldVal,
                  IntegerType::get(Ctx, LBits));
            }
            uint64_t EBits = PI->BitMode ? L.EffBits : LBits;
            if (EBits < LBits)
              FieldVal = RB.CreateTrunc(FieldVal, IntegerType::get(Ctx, EBits));
            Value *Extended = (EBits == PI->PackedBits)
                ? FieldVal
                : RB.CreateZExt(FieldVal, PI->PackedTy);
            uint64_t ShiftBits = PI->BitMode ? L.BitOffset : L.ByteOffset * 8;
            if (ShiftBits)
              Extended = RB.CreateShl(Extended,
                  ConstantInt::get(PI->PackedTy, ShiftBits));
            Packed = RB.CreateOr(Packed, Extended);
          }
          // Store packed → new_arg[i]
          Value *PackedGEP = RB.CreateInBoundsGEP(
              cast<ArrayType>(PackedArg->getType()->getPointerElementType()),
              PackedArg,
              {ConstantInt::get(I64, 0), ConstantInt::get(I64, i)});
          RB.CreateStore(Packed, PackedGEP);
        }
      }
    }

    // Strip SpecInterface/SpecBitsMap + llvm.sideeffect op-bundle calls that
    // reference the local allocas (they were referencing the old struct-array
    // arg via VMap remap and are now useless on a local).
    SmallVector<Instruction *, 8> Dead;
    for (BasicBlock &BB : *NewF) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        bool IsSpec = (N == "_ssdm_op_SpecInterface" ||
                       N == "_ssdm_op_SpecBitsMap");
        bool IsSideEffect = (CI->getIntrinsicID() == Intrinsic::sideeffect);
        if (!IsSpec && !IsSideEffect) continue;
        // For SpecInterface/SpecBitsMap, the arg is at position 0.
        // For sideeffect, the values referencing allocas live inside the
        // op-bundle operands.
        bool RefsLocal = false;
        if (IsSpec && CI->arg_size() >= 1) {
          Value *V = CI->getArgOperand(0);
          while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
          for (auto &AI : Allocas) {
            if (V == AI.Local) { RefsLocal = true; break; }
          }
        }
        if (IsSideEffect) {
          for (unsigned bi = 0, be = CI->getNumOperandBundles(); bi != be; ++bi) {
            auto OBU = CI->getOperandBundleAt(bi);
            for (const Use &Op : OBU.Inputs) {
              Value *V = Op.get();
              while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
              for (auto &AI : Allocas) {
                if (V == AI.Local) { RefsLocal = true; break; }
              }
              if (RefsLocal) break;
            }
            if (RefsLocal) break;
          }
        }
        if (RefsLocal) Dead.push_back(CI);
      }
    }
    for (Instruction *I : Dead) I->eraseFromParent();

    // Re-emit SpecInterface for the new packed args (mode based on existing
    // markers on the original args — for simplicity, default to ap_memory
    // unless the original had bram marker).
    // CRITICAL: emit SpecInterface BEFORE the unpack code (at the very start
    // of EntryBB). The HLS backend's "this kernel is canonical, skip auto-add"
    // detection looks for SpecInterface calls at the top of the function
    // body — if they come after non-spec code (allocas, loads, etc.) it
    // doesn't recognize them and re-adds SpecInterface(ap_memory) for every
    // arg (203-801 conflict with our explicit bram for c).
    {
      FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), true);
      FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
      FunctionCallee SBMFn  = M.getOrInsertFunction("_ssdm_op_SpecBitsMap",   SpecTy);
      if (auto *FF = dyn_cast<Function>(SpecFn.getCallee())) FF->addFnAttr(Attribute::NoUnwind);
      if (auto *FF = dyn_cast<Function>(SBMFn.getCallee()))  FF->addFnAttr(Attribute::NoUnwind);
      GlobalVariable *ApMemStr = getOrCreateCStrGlobal(M, "ap_memory");
      GlobalVariable *BramStr  = getOrCreateCStrGlobal(M, "bram");
      GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
      MDNode *MapMD = MDNode::get(Ctx, {MDNode::get(Ctx, {})});
      // Insert at the very TOP of EntryBB (before any alloca/unpack code).
      Instruction *InsertPt = &*EntryBB->getFirstInsertionPt();
      IRBuilder<> SB(InsertPt);
      Constant *Zero = ConstantInt::get(I32, 0);
      Constant *NegOne = ConstantInt::get(I32, -1);

      // First, collect ClonedEntry SpecTopModule/SpecInterface/SpecBitsMap
      // calls keyed by the arg they reference (for arg-order emit below).
      DenseMap<Argument*, SmallVector<CallInst*, 2>> ClonedSpecByArg;
      SmallVector<CallInst*, 4> ClonedSpecTopModule;
      for (Instruction &I : *ClonedEntry) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N == "_ssdm_op_SpecTopModule") {
          ClonedSpecTopModule.push_back(CI);
          continue;
        }
        if (N != "_ssdm_op_SpecInterface" && N != "_ssdm_op_SpecBitsMap")
          continue;
        if (CI->arg_size() < 1) continue;
        Value *V = CI->getArgOperand(0);
        while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
        if (auto *A = dyn_cast<Argument>(V)) ClonedSpecByArg[A].push_back(CI);
      }

      // Move SpecTopModule first (one expected; if dup, all to top).
      for (CallInst *CI : ClonedSpecTopModule) CI->moveBefore(InsertPt);

      // Per-arg emit in fn arg order
      // (the HLS backend expects SpecInterface in declaration order; out-of-order
      // emit looks like "user-added pragmas but not canonical" and triggers
      // blanket auto-add).
      DenseMap<Argument*, AllocaInfo*> AllocaByNewArg;
      for (auto &AI : Allocas) AllocaByNewArg[AI.NewArg] = &AI;
      Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
      Type *I64Ty = Type::getInt64Ty(Ctx);
      ArrayType *ZeroByteArrTy = ArrayType::get(Type::getInt8Ty(Ctx), 0);
      Constant *ZeroByteArr = ConstantAggregateZero::get(ZeroByteArrTy);
      Constant *NegOne64 = ConstantInt::get(I64Ty, -1);
      Constant *Zero64 = ConstantInt::get(I64Ty, 0);
      Constant *One64 = ConstantInt::get(I64Ty, 1);
      Type *I1Ty = Type::getInt1Ty(Ctx);
      FunctionCallee SpecResFn = M.getOrInsertFunction("_ssdm_op_SpecResource", FunctionType::get(Type::getVoidTy(Ctx), true));
      for (Argument &NewA : NewF->args()) {
        if (auto *AI = AllocaByNewArg.lookup(&NewA)) {
          // Packed arg — re-emit op-bundle + SpecInterface (both needed:
          // op-bundle for HLS backend recognition, SpecInterface for csim
          // top fn discovery).
          StringRef BundleName = AI->IsBram ? "xlx_bram" : "xlx_ap_memory";
          Constant *Arg7 = AI->IsBram ? Zero64 : One64;
          Value *BundleArgs[] = {
              (Value*)&NewA, NegOne64, NegOne64, NegOne64,
              (Value*)ZeroByteArr, NegOne64, (Value*)ZeroByteArr,
              Arg7, Zero};
          OperandBundleDef OBD(BundleName.str(),
                               ArrayRef<Value*>(BundleArgs, 9));
          CallInst *SECall = SB.CreateCall(SEFn, None,
                                           ArrayRef<OperandBundleDef>{OBD});
          SECall->setOnlyAccessesInaccessibleMemory();
          SECall->setDoesNotThrow();
          // Force single-port RAM via SpecResource(666 BIND_STORAGE, 18
          // Ram1pBram, -1 latency). C++ default for bram pragma w/ in-loop
          // access produces single-port; Rust's pre-unpack pattern triggers
          // HLS dual-port inference. Force RAM_1P explicitly to match C++.
          SB.CreateCall(SpecResFn,
                        {(Value*)&NewA,
                         ConstantInt::get(I64Ty, 666),
                         ConstantInt::get(I64Ty, 18),
                         ConstantInt::get(I64Ty, -1, true),
                         ConstantInt::get(I1Ty, 0)});
          GlobalVariable *ModeStr = AI->IsBram ? BramStr : ApMemStr;
          Value *SIArgs[] = {
              (Value*)&NewA, (Value*)ModeStr,
              Zero, Zero, (Value*)EmptyStr,
              NegOne, Zero,
              (Value*)EmptyStr, (Value*)EmptyStr, (Value*)EmptyStr,
              Zero, Zero, Zero, Zero,
              (Value*)EmptyStr, (Value*)EmptyStr,
              NegOne, Zero, Zero, Zero};
          CallInst *SI = SB.CreateCall(SpecFn, SIArgs);
          (void)SI;
          CallInst *SBM = SB.CreateCall(SBMFn, ArrayRef<Value*>{(Value*)&NewA});
          SBM->setMetadata("map", MapMD);
        } else if (auto It = ClonedSpecByArg.find(&NewA); It != ClonedSpecByArg.end()) {
          // Non-packed arg — move existing SpecInterface/SpecBitsMap calls
          // (from autoEmit etc.) to this position
          for (CallInst *CI : It->second) CI->moveBefore(InsertPt);
        }
      }
    }

    // Replace old function with new
    std::string OldName = Old->getName().str();
    if (!Old->use_empty())
      Old->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, Old->getType()));
    Old->eraseFromParent();
    NewF->setName(OldName);
    Changed = true;
  }

  for (CallInst *CI : MarkersToDrop) {
    if (CI->getParent()) CI->eraseFromParent();
  }
  if (AggMarker->use_empty()) AggMarker->eraseFromParent();
  return Changed;
}

// Rewrite an `ap_fifo` + `aggregate` struct-array port to the exact form the
// C++ frontend hands the closed flow (verified against aggregation_of_struct
// a.g.ld.5): the pp stage then does the packing (i119) and the stream
// lowering (IfRead) itself — we only shape the input (0-F: feed reflow, don't
// reimplement it).
//
//   [10 x %A]* %arr + SpecInterface(ap_fifo) + __vxx_aggregate marker
//     with __vxx_agg_leaf(w, signed) width notes (from aggregate! /
//     #[derive(AggregateLayout)]; ApInt<23> is i128 in the Rust layout, so
//     the logical width can only come from the source)
// becomes
//   %"struct.A"* "fpga.decayed.dim.hint"="10" %arr
//     where %"struct.A" = { [3 x i32], %"struct.ap_int<23>" } (the ap_int
//     named-struct nesting carries the width to the pp packer)
//   [ "xlx_ap_fifo"(%"struct.A"* %arr, i32 -1, [0 x i8] zeroinitializer,
//     i64 -1) ] + [ "aggregate"(ptr, i64 compact) ] op-bundles
//   per-element access as ONE whole-struct load + extractvalue chain — the
//   pp FIFO legality check (214-244) requires each array element be read
//   whole, in order, exactly once; rustc's SROA'd per-field loads fail it.
//
// Conservative: every use of the arg must be a per-field GEP (+load) with a
// loop-uniform element index; any other shape leaves the port unrewritten.
bool retypeAggregateFifoStructPort(Module &M) {
  Function *AggMarker = M.getFunction("__vxx_aggregate");
  Function *LeafMarker = M.getFunction("__vxx_agg_leaf");
  if (!AggMarker || !LeafMarker) return false;
  LLVMContext &Ctx = M.getContext();

  // Leaf width notes per argument, in PROGRAM order — users() iterates in an
  // arbitrary order, so walk the containing functions' instructions instead
  // (same lesson as the DSP-cascade marker collection).
  DenseMap<Argument *, SmallVector<std::pair<uint64_t, bool>, 8>> LeafW;
  SmallVector<CallInst *, 8> LeafCalls;
  {
    SmallPtrSet<CallInst *, 8> CallSet;
    SmallPtrSet<Function *, 4> Fns;
    for (User *U : LeafMarker->users())
      if (auto *CI = dyn_cast<CallInst>(U)) {
        CallSet.insert(CI);
        Fns.insert(CI->getFunction());
      }
    for (Function &F : M) {
      if (!Fns.count(&F)) continue;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CallSet.count(CI) || CI->arg_size() < 3) continue;
          LeafCalls.push_back(CI);
          Argument *A = resolveMarkerArg(CI->getArgOperand(0));
          auto *W = dyn_cast<ConstantInt>(CI->getArgOperand(1));
          auto *S = dyn_cast<ConstantInt>(CI->getArgOperand(2));
          if (!A || !W || !S) continue;
          LeafW[A].push_back({W->getZExtValue(), S->getZExtValue() != 0});
        }
    }
  }
  if (LeafW.empty()) {
    for (auto *CI : LeafCalls) CI->eraseFromParent();
    if (LeafMarker->use_empty()) LeafMarker->eraseFromParent();
    return false;
  }

  // ap_fifo detection: the RAW `__vxx_ap_fifo` marker — this pass runs
  // BEFORE injectApFifo (which skips struct-array args entirely, emitting no
  // SpecInterface for them).
  Function *FifoMarker = M.getFunction("__vxx_ap_fifo");
  auto findSpecFifo = [&](Argument *A) -> CallInst * {
    if (!FifoMarker) return nullptr;
    for (User *U : FifoMarker->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() < 1) continue;
      if (resolveMarkerArg(CI->getArgOperand(0)) == A) return CI;
    }
    return nullptr;
  };

  bool Changed = false;
  SmallVector<User *, 8> AggUsers(AggMarker->user_begin(), AggMarker->user_end());
  for (User *U : AggUsers) {
    auto *AggCI = dyn_cast<CallInst>(U);
    if (!AggCI || AggCI->arg_size() < 2) continue;
    Argument *Arg = resolveMarkerArg(AggCI->getArgOperand(0));
    auto *CompactC = dyn_cast<ConstantInt>(AggCI->getArgOperand(1));
    if (!Arg || !CompactC) continue;
    if (!LeafW.count(Arg)) continue;
    Function *Old = Arg->getParent();
    if (!Old->hasFnAttribute("fpga.top.func") || !Old->use_empty()) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    auto *AT = dyn_cast<ArrayType>(PT->getElementType());
    if (!AT) continue;
    auto *ST = dyn_cast<StructType>(AT->getElementType());
    if (!ST) continue;
    CallInst *FifoSpec = findSpecFifo(Arg);
    if (!FifoSpec) continue; // ap_memory family goes to the pack pass

    // --- Build the C++-shaped element type: keep native-width fields, wrap
    // width-noted narrow ints in the ap_int/ap_uint named nesting.
    auto &Notes = LeafW[Arg];
    auto namedTy = [&](const std::string &Nm, ArrayRef<Type *> Fl) -> StructType * {
      StructType *T = M.getTypeByName(Nm);
      if (!T) T = StructType::create(Ctx, Fl, Nm);
      return T;
    };
    SmallVector<Type *, 8> NewFields;   // new element struct fields
    // Old GEP field index -> (new field index, extra nesting for ap_int).
    DenseMap<unsigned, std::pair<unsigned, bool>> FieldMap;
    size_t NoteIdx = 0;
    bool Bad = false;
    for (unsigned fi = 0; fi < ST->getNumElements() && !Bad; ++fi) {
      Type *FT = ST->getElementType(fi);
      if (isPaddingArray(FT)) continue;
      if (auto *FAT = dyn_cast<ArrayType>(FT)) {
        if (FAT->getNumElements() == 0) continue;
        // Int/float array field: consumes one note per element (widths must
        // equal the native element width — no narrow arrays supported yet).
        // Rust repr(C) inserts UNNAMED padding arrays whose element type is
        // not always i8 (e.g. [1 x i32] before a 16-aligned i128); they carry
        // no AggregateLayout note, so an array whose element width does NOT
        // match the pending note is treated as padding and skipped. A real
        // field wrongly skipped here leaves notes unconsumed and fails the
        // final count check.
        uint64_t EW = FAT->getElementType()->getPrimitiveSizeInBits();
        if (NoteIdx >= Notes.size() || Notes[NoteIdx].first != EW) continue;
        for (uint64_t k = 0; k < FAT->getNumElements(); ++k, ++NoteIdx)
          if (NoteIdx >= Notes.size() || Notes[NoteIdx].first != EW) Bad = true;
        FieldMap[fi] = {(unsigned)NewFields.size(), false};
        NewFields.push_back(FT);
        continue;
      }
      if (NoteIdx >= Notes.size()) { Bad = true; break; }
      uint64_t W = Notes[NoteIdx].first;
      bool Signed = Notes[NoteIdx].second;
      ++NoteIdx;
      uint64_t LBits = FT->getPrimitiveSizeInBits();
      if (W == LBits) {
        FieldMap[fi] = {(unsigned)NewFields.size(), false};
        NewFields.push_back(FT);
      } else if (FT->isIntegerTy() && W < LBits) {
        std::string ws = std::to_string(W);
        std::string sg = Signed ? "true" : "false";
        StructType *Ssdm = namedTy("struct.ssdm_int<" + ws + ", " + sg + ">",
                                   {IntegerType::get(Ctx, W)});
        StructType *Base =
            namedTy("struct.ap_int_base<" + ws + ", " + sg + ">", {Ssdm});
        StructType *Ap = namedTy(
            (Signed ? "struct.ap_int<" : "struct.ap_uint<") + ws + ">", {Base});
        FieldMap[fi] = {(unsigned)NewFields.size(), true};
        NewFields.push_back(Ap);
      } else {
        Bad = true;
      }
    }
    if (Bad || NoteIdx != Notes.size()) continue;
    StructType *NewST = StructType::create(Ctx, NewFields, "struct.AggFifoElem");

    // --- Validate + collect the arg's accesses: per-field GEPs feeding one
    // load each, all with the same (loop-carried) element index per BB.
    struct Acc {
      GetElementPtrInst *GEP;
      LoadInst *LD;
      unsigned Field;                  // old struct field index
      SmallVector<uint64_t, 2> Inner;  // constant indices inside the field
      Value *ElemIdx;
    };
    SmallVector<Acc, 8> Accs;
    bool Bail = false;
    for (User *AU : Arg->users()) {
      if (auto *CI = dyn_cast<CallInst>(AU)) {
        if (CI == AggCI || CI == FifoSpec) continue;
        if (Function *CF = CI->getCalledFunction()) {
          StringRef N = CF->getName();
          if (N == "_ssdm_op_SpecBitsMap" || N == "__vxx_agg_leaf") continue;
        }
        Bail = true;
        break;
      }
      // Marker calls take the port through a `bitcast ... to i8*`; accept a
      // bitcast whose every user is a call (they are erased with the arg).
      if (auto *BC = dyn_cast<BitCastInst>(AU)) {
        bool AllCalls = true;
        for (User *BU : BC->users())
          if (!isa<CallInst>(BU)) { AllCalls = false; break; }
        if (AllCalls) continue;
        Bail = true;
        break;
      }
      auto *GEP = dyn_cast<GetElementPtrInst>(AU);
      if (!GEP || GEP->getNumOperands() < 4) { Bail = true; break; }
      auto *FieldC = dyn_cast<ConstantInt>(GEP->getOperand(3));
      if (!FieldC) { Bail = true; break; }
      Acc A;
      A.GEP = GEP;
      A.Field = FieldC->getZExtValue();
      A.ElemIdx = GEP->getOperand(2);
      for (unsigned oi = 4; oi < GEP->getNumOperands(); ++oi) {
        auto *C = dyn_cast<ConstantInt>(GEP->getOperand(oi));
        if (!C) { Bail = true; break; }
        A.Inner.push_back(C->getZExtValue());
      }
      if (Bail) break;
      if (!GEP->hasOneUse()) { Bail = true; break; }
      A.LD = dyn_cast<LoadInst>(*GEP->user_begin());
      if (!A.LD) { Bail = true; break; }
      // rustc SROA may address the first real field THROUGH a leading
      // zero-length array member ([0 x i32] at offset 0): remap such a GEP
      // to the following real field's first element.
      if (!FieldMap.count(A.Field)) {
        auto *ZA = dyn_cast<ArrayType>(ST->getElementType(A.Field));
        bool ZeroAlias = ZA && ZA->getNumElements() == 0 &&
                         A.Inner.size() == 1 && A.Inner[0] == 0;
        unsigned Next = A.Field + 1;
        while (ZeroAlias && Next < ST->getNumElements() && !FieldMap.count(Next))
          ++Next;
        if (!ZeroAlias || Next >= ST->getNumElements()) { Bail = true; break; }
        A.Field = Next;
        A.Inner.clear();
        A.Inner.push_back(0);
      }
      Accs.push_back(A);
    }
    if (Bail || Accs.empty()) continue;

    // --- Clone into the retyped signature.
    SmallVector<Type *, 8> NewParams;
    for (Argument &A : Old->args()) NewParams.push_back(A.getType());
    NewParams[Arg->getArgNo()] = PointerType::get(NewST, 0);
    FunctionType *NewFT =
        FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF = Function::Create(NewFT, Old->getLinkage(), "", &M);
    NewF->copyAttributesFrom(Old);
    NewF->copyMetadata(Old, 0);
    NewF->getBasicBlockList().splice(NewF->begin(), Old->getBasicBlockList());
    Argument *NewArg = nullptr;
    {
      auto OI = Old->arg_begin();
      auto NI = NewF->arg_begin();
      for (; OI != Old->arg_end(); ++OI, ++NI) {
        NI->setName(OI->getName());
        if (&*OI == Arg) NewArg = &*NI;
        else OI->replaceAllUsesWith(&*NI);
      }
    }
    NewF->removeParamAttr(NewArg->getArgNo(), Attribute::Dereferenceable);
    NewF->removeParamAttr(NewArg->getArgNo(), Attribute::Alignment);
    NewF->addParamAttr(NewArg->getArgNo(),
        Attribute::get(Ctx, "fpga.decayed.dim.hint",
                       std::to_string(AT->getNumElements())));

    // --- Rewrite accesses: one whole-element load per (BB, elem idx), then
    // extractvalue per field. Process in PROGRAM order so the shared load is
    // created at the EARLIEST access and dominates every extractvalue.
    llvm::sort(Accs, [](const Acc &X, const Acc &Y) {
      if (X.LD->getParent() != Y.LD->getParent())
        return X.LD->getParent() < Y.LD->getParent();
      return X.LD->comesBefore(Y.LD);
    });
    DenseMap<std::pair<BasicBlock *, Value *>, LoadInst *> ElemLoad;
    for (Acc &A : Accs) {
      IRBuilder<> B(A.LD);
      auto Key = std::make_pair(A.LD->getParent(), A.ElemIdx);
      LoadInst *Whole = ElemLoad.lookup(Key);
      if (!Whole) {
        Value *EPtr = B.CreateInBoundsGEP(NewST, NewArg, A.ElemIdx);
        Whole = B.CreateLoad(NewST, EPtr);
        Whole->setAlignment(Align(4));
        ElemLoad[Key] = Whole;
      }
      unsigned NewField = FieldMap[A.Field].first;
      bool ApNested = FieldMap[A.Field].second;
      SmallVector<unsigned, 4> Idx{NewField};
      if (ApNested) {
        Idx.append({0, 0, 0}); // ap_int -> ap_int_base -> ssdm_int -> iW
      } else {
        for (uint64_t v : A.Inner) Idx.push_back((unsigned)v);
      }
      Value *FV = B.CreateExtractValue(Whole, Idx);
      if (FV->getType() != A.LD->getType()) {
        auto &Note = Notes[0]; (void)Note;
        bool Sgn = false;
        // Recover signedness from the wrapped type name.
        if (auto *WST = dyn_cast<StructType>(NewST->getElementType(NewField)))
          Sgn = WST->getName().contains("ap_int<");
        FV = Sgn ? B.CreateSExt(FV, A.LD->getType())
                 : B.CreateZExt(FV, A.LD->getType());
      }
      A.LD->replaceAllUsesWith(FV);
      A.LD->eraseFromParent();
      if (A.GEP->use_empty()) A.GEP->eraseFromParent();
    }

    // --- Emit the C++-shaped op-bundles at the top; drop the ap_fifo
    // SpecInterface (pp re-derives it from the bundle).
    {
      Instruction *At = &*NewF->getEntryBlock().getFirstInsertionPt();
      IRBuilder<> B(At);
      Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
      ArrayType *ZB = ArrayType::get(Type::getInt8Ty(Ctx), 0);
      Value *FifoArgs[] = {(Value *)NewArg,
                           ConstantInt::get(Type::getInt32Ty(Ctx), -1, true),
                           ConstantAggregateZero::get(ZB),
                           ConstantInt::get(Type::getInt64Ty(Ctx), -1, true)};
      OperandBundleDef FifoOB("xlx_ap_fifo", ArrayRef<Value *>(FifoArgs, 4));
      CallInst *C1 = B.CreateCall(SE, None, {FifoOB});
      C1->setOnlyAccessesInaccessibleMemory();
      C1->setDoesNotThrow();
      Value *AggArgs[] = {(Value *)NewArg,
                          ConstantInt::get(Type::getInt64Ty(Ctx),
                                           CompactC->getZExtValue())};
      OperandBundleDef AggOB("aggregate", ArrayRef<Value *>(AggArgs, 2));
      CallInst *C2 = B.CreateCall(SE, None, {AggOB});
      C2->setOnlyAccessesInaccessibleMemory();
      C2->setDoesNotThrow();
    }
    FifoSpec->eraseFromParent();
    AggCI->eraseFromParent();
    // Drop stale marker calls / bitcasts on the old arg (now unused).
    {
      SmallVector<Instruction *, 8> Stale;
      for (User *AU : Arg->users()) {
        if (auto *CI = dyn_cast<CallInst>(AU)) Stale.push_back(CI);
        else if (auto *BC = dyn_cast<BitCastInst>(AU)) {
          for (User *BU : BC->users())
            if (auto *CI2 = dyn_cast<CallInst>(BU)) Stale.push_back(CI2);
          Stale.push_back(BC);
        }
      }
      for (Instruction *I : Stale)
        if (I->getParent()) I->eraseFromParent();
    }

    std::string Name = Old->getName().str();
    Old->eraseFromParent();
    NewF->setName(Name);
    vxxDbg() << "vxx: retyped aggregate ap_fifo struct port in " << Name
             << "\n";
    Changed = true;
  }

  for (auto *CI : LeafCalls)
    if (CI->getParent()) CI->eraseFromParent();
  if (LeafMarker->use_empty()) LeafMarker->eraseFromParent();
  return Changed;
}

bool injectAggregate(Module &M) {
  bool A = packAggregateByteKernelSig(M);
  bool B = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_aggregate");
  bool L = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_agg_leaf");
  return A || B || L;
}

// Disaggregate kernel sig for `[N x T]*` args that have a complete-partition
// marker (`__vxx_array_partition(ptr, kind=2, factor=0, dim)` in Rust convention).
// C++ source `#pragma HLS array_partition variable=a complete` becomes 4
// scalar `i64* %a_0, %a_1, %a_2, %a_3` args at a.g.0.bc — OSS Clang does
// the disagg from the source pragma. Rust IR has the array form + a marker;
// VXXPrep must do the disagg in-place.
//
// Body remapping: alloca temp of `[N x T]`, at entry pre-load N new args
// into temp fields, body's `GEP [N x T], %a, 0, %k` → `GEP temp, 0, %k`
// (untouched, just remapped via VMap). HLS unroll later will propagate
// scalar loads.
//
// Trigger: top fn with `[N x T]*` arg + `__vxx_array_partition` marker
// with kind==2 (Rust complete) factor==0.
//
// Without sig disagg, kernel emits single BRAM port `a_address0/ce0/q0`
// (HLS 200-70 "Failed building synthesis data model" if we try
// SpecArrayPartition kind=3 on array-typed arg).
// ===========================================================================
// `#pragma HLS disaggregate variable=<arg>` on a struct-ARRAY port. The C++
// frontend consumes this pragma BEFORE reflow: disaggregation_of_axis_port's
// a.pp.bc is already `@dut(i8 as(128)* %in_c, i32 as(128)* %in_i, ...)` — one
// per-field array arg named `<arg>_<field>`. Reproduce exactly that from the
// `__vxx_disaggregate(ptr, names, names_len)` marker (names = comma-separated
// field names from the #[derive(HlsStruct)] — LLVM IR has no field names).
//
// The split runs BEFORE the Phase 6 interface emitters, so a per-port
// `__vxx_axis` marker is re-pointed at each new per-field arg and the
// existing AXIS array machinery (decay + addrspace(128) + SpecInterface)
// treats the result exactly like a hand-split kernel.
//
// Body rewrite: rustc's -O1 body accesses the port ONLY through constant-
// field GEPs (`gep [N x {..}]* %arg, 0, %k, F` — SROA splits whole-struct
// copies), so each GEP is re-pointed at the matching per-field arg. A bridge
// alloca would break AXIS semantics (the port would see no reads). Any
// access shape we can't re-point aborts the whole function's transform.
//
// Gate: uncalled top kernels (the cpp_proxy build calls the kernel from the
// generated adapter with the struct-array ABI and must keep it).
bool disaggStructArrayKernelSig(Module &M) {
  Function *Marker = M.getFunction("__vxx_disaggregate");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // arg → field names
  DenseMap<Argument *, SmallVector<std::string, 8>> Targets;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 3) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Function *F = Arg->getParent();
    if (!F->hasFnAttribute("fpga.top.func") || !F->use_empty()) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    auto *AT = PT ? dyn_cast<ArrayType>(PT->getElementType()) : nullptr;
    auto *ST = AT ? dyn_cast<StructType>(AT->getElementType()) : nullptr;
    if (!ST) continue;
    std::string Names = hlsrs::vxx::extractByteSlice(CI->getArgOperand(1),
                                                     CI->getArgOperand(2));
    SmallVector<std::string, 8> Fields;
    size_t Pos = 0;
    while (Pos <= Names.size()) {
      size_t Comma = Names.find(',', Pos);
      if (Comma == std::string::npos) { Fields.push_back(Names.substr(Pos)); break; }
      Fields.push_back(Names.substr(Pos, Comma - Pos));
      Pos = Comma + 1;
    }
    if (Fields.size() != ST->getNumElements()) continue;
    Targets[Arg] = Fields;
  }
  if (Targets.empty()) return false;

  DenseMap<Function *, SmallVector<Argument *, 4>> ByFn;
  for (auto &E : Targets) ByFn[E.first->getParent()].push_back(E.first);

  bool Changed = false;
  for (auto &Pair : ByFn) {
    Function *Old = Pair.first;
    SmallVector<Argument *, 8> OldArgs;
    for (Argument &A : Old->args()) OldArgs.push_back(&A);

    // New param list: target [N x {f0,f1,..}]* → per-field [N x fI]*.
    SmallVector<Type *, 16> NewParams;
    for (Argument *A : OldArgs) {
      auto It = Targets.find(A);
      if (It == Targets.end()) { NewParams.push_back(A->getType()); continue; }
      auto *AT = cast<ArrayType>(
          cast<PointerType>(A->getType())->getElementType());
      auto *ST = cast<StructType>(AT->getElementType());
      for (unsigned f = 0; f < ST->getNumElements(); ++f)
        NewParams.push_back(PointerType::get(
            ArrayType::get(ST->getElementType(f), AT->getNumElements()), 0));
    }
    FunctionType *NewFT =
        FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF =
        Function::Create(NewFT, Old->getLinkage(), Old->getAddressSpace(),
                         Old->getName() + ".fld_disagg", &M);
    AttributeList OldAL = Old->getAttributes();
    NewF->setAttributes(AttributeList::get(Ctx, OldAL.getFnAttributes(),
                                           OldAL.getRetAttributes(), {}));

    // Map each old arg; targets go through a placeholder alloca whose GEPs
    // are re-pointed after the clone.
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "fld.entry", NewF);
    IRBuilder<> EB(EntryBB);
    ValueToValueMapTy VMap;
    struct Placeholder {
      AllocaInst *AI;
      SmallVector<Argument *, 8> FieldArgs;
    };
    SmallVector<Placeholder, 4> Placeholders;
    auto NewArgIt = NewF->arg_begin();
    for (Argument *A : OldArgs) {
      auto It = Targets.find(A);
      if (It == Targets.end()) {
        NewArgIt->setName(A->getName());
        if (A->hasAttribute(Attribute::NoAlias))
          NewArgIt->addAttr(Attribute::NoAlias);
        VMap[A] = &*NewArgIt;
        ++NewArgIt;
        continue;
      }
      auto *AT = cast<ArrayType>(
          cast<PointerType>(A->getType())->getElementType());
      auto *ST = cast<StructType>(AT->getElementType());
      AllocaInst *AI = EB.CreateAlloca(AT, nullptr, A->getName() + ".fld_tmp");
      VMap[A] = AI;
      Placeholder P{AI, {}};
      for (unsigned f = 0; f < ST->getNumElements(); ++f) {
        NewArgIt->setName(A->getName().str() + "_" + It->second[f]);
        NewArgIt->addAttr(Attribute::NoAlias);
        P.FieldArgs.push_back(&*NewArgIt);
        ++NewArgIt;
      }
      Placeholders.push_back(P);
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap, /*ModuleLevelChanges=*/false, Returns);
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF)
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    if (!ClonedEntry) { NewF->eraseFromParent(); continue; }
    EB.CreateBr(ClonedEntry);

    // Re-point every use of each placeholder alloca:
    //  - GEP with constant field index → GEP on the per-field arg.
    //  - bitcast feeding ONLY `__vxx_*` marker calls → per-field markers for
    //    `__vxx_axis`; the (cloned) `__vxx_disaggregate` is just erased.
    bool Aborted = false;
    for (Placeholder &P : Placeholders) {
      SmallVector<User *, 16> Users;
      {
        // users() yields one entry PER USE — dedupe so an instruction
        // taking the alloca in two operands is visited (and erased) once
        SmallPtrSet<User *, 16> Seen;
        for (User *UU : P.AI->users())
          if (Seen.insert(UU).second) Users.push_back(UU);
      }
      for (User *U : Users) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // Expect gep [N x ST], AI, 0, <elem>, <constant field>.
          if (GEP->getNumIndices() != 3) { Aborted = true; break; }
          auto IdxIt = GEP->idx_begin();
          auto *Zero = dyn_cast<ConstantInt>(IdxIt->get());
          Value *Elem = (IdxIt + 1)->get();
          auto *FieldC = dyn_cast<ConstantInt>((IdxIt + 2)->get());
          if (!Zero || !Zero->isZero() || !FieldC) { Aborted = true; break; }
          unsigned FIdx = (unsigned)FieldC->getZExtValue();
          if (FIdx >= P.FieldArgs.size()) { Aborted = true; break; }
          Argument *FArg = P.FieldArgs[FIdx];
          IRBuilder<> B(GEP);
          auto *FAT = cast<ArrayType>(
              cast<PointerType>(FArg->getType())->getElementType());
          Value *NewGEP = B.CreateInBoundsGEP(
              FAT, FArg, {ConstantInt::get(I64, 0), Elem},
              GEP->getName());
          GEP->replaceAllUsesWith(NewGEP);
          GEP->eraseFromParent();
          continue;
        }
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          SmallVector<CallInst *, 4> MarkerUses;
          bool OnlyMarkers = true;
          for (User *BU : BC->users()) {
            auto *CI = dyn_cast<CallInst>(BU);
            Function *CF = CI ? CI->getCalledFunction() : nullptr;
            if (!CF || !CF->getName().startswith("__vxx_")) {
              OnlyMarkers = false;
              break;
            }
            MarkerUses.push_back(CI);
          }
          if (!OnlyMarkers) { Aborted = true; break; }
          for (CallInst *CI : MarkerUses) {
            StringRef MN = CI->getCalledFunction()->getName();
            if (MN == "__vxx_axis") {
              IRBuilder<> B(CI);
              for (Argument *FArg : P.FieldArgs) {
                Value *Cast = B.CreateBitCast(FArg, BC->getType());
                B.CreateCall(CI->getCalledFunction(), {Cast});
              }
            }
            // __vxx_disaggregate (and the consumed __vxx_axis) just go away.
            CI->eraseFromParent();
          }
          BC->eraseFromParent();
          continue;
        }
        Aborted = true;
        break;
      }
      if (Aborted) break;
    }
    bool AllocasDead = !Aborted;
    for (Placeholder &P : Placeholders)
      if (!P.AI->use_empty()) AllocasDead = false;
    if (!AllocasDead) {
      NewF->eraseFromParent();
      vxxDbg() << "vxx: disaggStructArrayKernelSig aborted on "
               << Old->getName() << " (unrecognised port access shape)\n";
      continue;
    }
    for (Placeholder &P : Placeholders) P.AI->eraseFromParent();

    std::string OldName = Old->getName().str();
    Old->eraseFromParent(); // use_empty by the gate above
    NewF->setName(OldName);
    vxxDbg() << "vxx: disaggregated struct-array port(s) on top kernel "
             << OldName << "\n";
    Changed = true;
  }
  return Changed;
}

// ===========================================================================
// Dissolve named single-scalar-field wrapper structs (C++ `template<T>
// struct dat_t { T data; }`) into their field type, module-wide. The C++
// frontend flattens such wrappers BEFORE reflow (aliasing a.pp.bc: the
// `hls::stream<dat_t<u32>>` ports arrive as `i32 addrspace(128)*`, the
// m_axi pointers as `i32 addrspace(1)*`), so every downstream machinery
// (stream class renames, fifo intrinsics, AXIS, m_axi bursts) must see the
// scalar too. Gated to structs whose name looks like a user wrapper (not
// the flow's own `class.`/`struct.`/`lifted` types).
bool dissolveSingleFieldStructs(Module &M) {
  vxxDbg() << "vxx: dissolve scan " << M.getName() << " ntypes="
           << M.getIdentifiedStructTypes().size() << "\n";
  for (StructType *ST : M.getIdentifiedStructTypes())
    vxxDbg() << "vxx:   type '" << (ST->hasName() ? ST->getName() : "<anon>")
             << "' nelem=" << ST->getNumElements() << "\n";
  DenseMap<Type *, Type *> TypeMap;
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (!ST->hasName()) continue;
    StringRef N = ST->getName();
    if (N.startswith("class.") || N.startswith("struct.") ||
        N.startswith("lifted") || N.startswith("barista_hls::"))
      continue; // flow/library types keep their shapes (AxisDisabled etc.)
    if (ST->getNumElements() != 1) continue;
    Type *F0 = ST->getElementType(0);
    if (!F0->isIntegerTy() && !F0->isFloatTy() && !F0->isDoubleTy()) continue;
    TypeMap[ST] = F0;
    vxxDbg() << "vxx: dissolving single-field struct " << N << " -> ";
    F0->print(vxxDbg());
    vxxDbg() << "\n";
  }
  if (TypeMap.empty()) return false;
  // Names of the dissolved structs — the second round below must touch ONLY
  // wrappers that contained one (other padded wrappers belong to the
  // existing strip/AXIS machinery and must keep their shapes).
  SmallVector<std::string, 4> DissolvedNames;
  for (auto &KV : TypeMap)
    DissolvedNames.push_back(cast<StructType>(KV.first)->getName().str());
  hlsrs::vxx::remapStructsInModule(M, TypeMap, /*FieldMaps=*/nullptr);

  // Second round: the dissolve leaves the enclosing Stream wrapper as a
  // rustc ZSA sandwich around the (now scalar) element —
  // `{[0 x i32], i32, [0 x i32]}` — which the earlier wrapper strips no
  // longer match (their inner-struct patterns assume a struct element).
  // Canonicalize any such sandwich-around-one-scalar to `{ scalar }`,
  // keeping the struct's NAME (the stream class rename keys on it).
  {
    const DataLayout &DL = M.getDataLayout();
    DenseMap<Type *, Type *> WrapMap;
    DenseMap<StructType *, std::vector<int>> FieldMaps;
    for (StructType *ST : M.getIdentifiedStructTypes()) {
      if (!ST->hasName() || ST->getNumElements() < 2) continue;
      // Only wrappers whose NAME references a dissolved struct (e.g.
      // `barista_hls::Stream<dat_t<u32>>`).
      bool Related = false;
      for (const std::string &DN : DissolvedNames)
        if (!DN.empty() && ST->getName().contains(DN)) { Related = true; break; }
      if (!Related) continue;
      int RealIdx = -1;
      bool Ok = true;
      for (unsigned i = 0; i < ST->getNumElements(); ++i) {
        Type *FT = ST->getElementType(i);
        if (DL.getTypeAllocSize(FT) == 0) continue; // zero-size pad
        if (!FT->isIntegerTy() && !FT->isFloatTy() && !FT->isDoubleTy()) {
          Ok = false;
          break;
        }
        if (RealIdx != -1) { Ok = false; break; }
        RealIdx = (int)i;
      }
      if (!Ok || RealIdx == -1) continue;
      Type *Scalar = ST->getElementType((unsigned)RealIdx);
      std::string N = ST->getName().str();
      // A dissolved-element Stream wrapper takes the SCALAR's spelling
      // (`barista_hls::Stream<i32>`) so the downstream class renamer maps
      // it to the same `class.hls::stream<int>` the C++ proxy names —
      // LLVM type-name equality is what clears HLS 214-136 at the
      // cosim boundary. Merge with an existing same-name type if any.
      if (StringRef(N).startswith("barista_hls::Stream<")) {
        std::string Spelled;
        if (auto *IT = dyn_cast<IntegerType>(Scalar))
          Spelled = "i" + std::to_string(IT->getBitWidth());
        else if (Scalar->isFloatTy())
          Spelled = "f32";
        else if (Scalar->isDoubleTy())
          Spelled = "f64";
        if (!Spelled.empty())
          N = "barista_hls::Stream<" + Spelled + ">";
      }
      StructType *NewST = M.getTypeByName(N);
      if (NewST && (NewST->getNumElements() != 1 ||
                    NewST->getElementType(0) != Scalar))
        NewST = nullptr; // same name, different body — make a fresh one
      if (!NewST) {
        ST->setName("");
        NewST = StructType::create(M.getContext(), {Scalar}, N,
                                   ST->isPacked());
      } else {
        ST->setName("");
      }
      WrapMap[ST] = NewST;
      std::vector<int> FMv(ST->getNumElements(), -1);
      FMv[(unsigned)RealIdx] = 0;
      FieldMaps[ST] = FMv;
      vxxDbg() << "vxx: canonicalized ZSA wrapper " << N << " -> { scalar }\n";
    }
    if (!WrapMap.empty())
      hlsrs::vxx::remapStructsInModule(M, WrapMap, &FieldMaps);
  }
  return true;
}

// ===========================================================================
// AUTOMATIC struct-port disaggregation (`auto_disaggregation_of_struct`):
// C++ `long dut(struct A& d)` with `A { hls::stream<int> s_in; long arr[N]; }`
// carries NO pragma — the frontend auto-splits the struct-ref port. Its
// a.pp.bc is `@dut(i32* %d_s_in, [10 x i64]* %d_arr)`: one arg per field,
// named `<arg>_<field>`, the stream member decayed to its ELEMENT pointer.
//
// The Rust `#[top]` wrapper probes every struct-ref param; when the struct
// derives HlsStruct the probe emits `__vxx_auto_disagg(ptr, names, names_len,
// mask, mask_len)` (mask = 1 byte per field, 1 for Stream members). This pass
// reproduces the frontend split:
//   - non-stream field F of type Ti     → new arg `Ti* <arg>_<field>`
//     (`gep %d, 0, F` re-pointed at the arg directly)
//   - stream field ({T} single-field)   → new arg `T* <arg>_<field>` plus a
//     `__vxx_top_stream_param(new_idx)` at entry so renameTopStreamArgs
//     retypes it to class.hls::stream<T>* like a hand-written Stream param.
//     (`gep %d, 0, F` gives {T}*; its inner `gep {T}*, 0, 0` re-points at
//     the arg, bitcasts re-cast.)
// Gate: uncalled top kernels, same as the explicit-pragma pass above.
bool autoDisaggStructRefKernelSig(Module &M) {
  Function *Marker = M.getFunction("__vxx_auto_disagg");
  if (!Marker) {
    vxxDbg() << "vxx: auto_disagg pass — no marker in " << M.getName() << "\n";
    return false;
  }
  vxxDbg() << "vxx: auto_disagg pass — marker users=" << Marker->getNumUses()
           << " in " << M.getName() << "\n";
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  const DataLayout &DL = M.getDataLayout();

  struct Info {
    SmallVector<std::string, 8> Names;
    SmallVector<bool, 8> IsStream;
    // Declared field index -> IR struct field index (rustc interleaves
    // explicit padding fields; mapped by repr(C) offset + size).
    SmallVector<unsigned, 8> IRIdx;
    // repr(C) byte offset of each declared field.
    SmallVector<uint64_t, 8> DeclOffs;
  };

  // Per-field layout markers (`__vxx_auto_disagg_f(ptr, idx, size, align,
  // is_stream)`) — scalar immediates keyed by the resolved argument.
  struct FieldInfo { uint64_t Size; uint64_t Align; bool IsStream; };
  DenseMap<Argument *, SmallVector<FieldInfo, 8>> FieldsByArg;
  if (Function *FMark = M.getFunction("__vxx_auto_disagg_f")) {
    for (User *U : FMark->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() < 5) continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg) continue;
      auto *IdxC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      auto *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      auto *AlignC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
      auto *StreamC = dyn_cast<ConstantInt>(CI->getArgOperand(4));
      if (!IdxC || !SizeC || !AlignC || !StreamC) continue;
      auto &V = FieldsByArg[Arg];
      unsigned Idx = (unsigned)IdxC->getZExtValue();
      if (V.size() <= Idx) V.resize(Idx + 1);
      V[Idx] = {SizeC->getZExtValue(), AlignC->getZExtValue(),
                StreamC->getZExtValue() != 0};
    }
  }

  DenseMap<Argument *, Info> Targets;   // uncalled-top args (design path)
  DenseMap<Argument *, Info> ImplInfos; // non-top args (split-adapter path)
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 4) {
      vxxDbg() << "vxx: auto_disagg marker user skipped (arity)\n";
      continue;
    }
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) {
      vxxDbg() << "vxx: auto_disagg marker skipped (arg not resolved)\n";
      continue;
    }
    Function *F = Arg->getParent();
    bool IsTop = F->hasFnAttribute("fpga.top.func") && F->use_empty();
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    auto *ST = PT ? dyn_cast<StructType>(PT->getElementType()) : nullptr;
    if (!ST) {
      vxxDbg() << "vxx: auto_disagg skipped (arg not struct*)\n";
      continue;
    }
    std::string Names = hlsrs::vxx::extractByteSlice(CI->getArgOperand(1),
                                                     CI->getArgOperand(2));
    Info I;
    size_t Pos = 0;
    while (Pos <= Names.size()) {
      size_t Comma = Names.find(',', Pos);
      if (Comma == std::string::npos) { I.Names.push_back(Names.substr(Pos)); break; }
      I.Names.push_back(Names.substr(Pos, Comma - Pos));
      Pos = Comma + 1;
    }
    unsigned NF = I.Names.size();
    auto FIt = FieldsByArg.find(Arg);
    if (FIt == FieldsByArg.end() || FIt->second.size() != NF) {
      vxxDbg() << "vxx: auto_disagg skipped on " << F->getName()
               << " (field markers " 
               << (FIt == FieldsByArg.end() ? 0 : FIt->second.size())
               << " != fields " << NF << ")\n";
      continue;
    }
    SmallVector<uint64_t, 8> Sizes, Aligns;
    for (auto &FI : FIt->second) {
      Sizes.push_back(FI.Size);
      Aligns.push_back(FI.Align);
      I.IsStream.push_back(FI.IsStream);
    }
    // repr(C) offsets of the DECLARED fields.
    SmallVector<uint64_t, 8> DeclOff;
    uint64_t Off = 0;
    for (unsigned f = 0; f < NF; ++f) {
      uint64_t A = Aligns[f] ? Aligns[f] : 1;
      Off = (Off + A - 1) / A * A;
      DeclOff.push_back(Off);
      Off += Sizes[f];
    }
    // Map each declared field to the IR field at the same offset with the
    // same allocation size (rustc's zero-size / pad fields share offsets
    // but differ in size).
    const StructLayout *SL = DL.getStructLayout(ST);
    bool Ok = true;
    for (unsigned f = 0; f < NF && Ok; ++f) {
      bool Found = false;
      for (unsigned j = 0; j < ST->getNumElements(); ++j) {
        if (SL->getElementOffset(j) != DeclOff[f]) continue;
        if (DL.getTypeAllocSize(ST->getElementType(j)) != Sizes[f]) continue;
        I.IRIdx.push_back(j);
        Found = true;
        break;
      }
      if (!Found) Ok = false;
    }
    if (!Ok) {
      vxxDbg() << "vxx: auto_disagg skipped on " << F->getName()
               << " (offset map failed)\n";
      continue;
    }
    I.DeclOffs = DeclOff;
    if (IsTop)
      Targets[Arg] = I;
    else
      ImplInfos[Arg] = I;
  }
  if (Targets.empty() && ImplInfos.empty()) {
    hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_auto_disagg");
    hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_auto_disagg_f");
    hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_split_adapter");
    return false;
  }

  bool Changed = false;
  // Impl functions split for the cpp_proxy adapter (old -> split clone):
  // a placeholder used as a CALL operand is rewritten through this map.
  DenseMap<Function *, Function *> CallRewrites;
  // Split every ArgMap-listed struct-ref arg of Old into per-field pointer
  // args. Returns the clone (taking Old's name when ReplaceOld) or null on
  // abort. Shared by the design-top loop and the cpp_proxy adapter path.
  auto SplitOne = [&](Function *Old, DenseMap<Argument *, Info> &ArgMap,
                      bool EmitTopMarkers, bool ReplaceOld) -> Function * {
    SmallVector<Argument *, 8> OldArgs;
    for (Argument &A : Old->args()) OldArgs.push_back(&A);

    // New param type per DECLARED field: pointer to the IR field type; a
    // stream field still in `{T}` wrapper form is unwrapped to `T*` (rustc
    // may also have flattened it to the scalar already).
    auto FieldPtrTy = [&](StructType *ST, const Info &I, unsigned f) -> Type * {
      Type *FT = ST->getElementType(I.IRIdx[f]);
      if (I.IsStream[f])
        if (auto *FS = dyn_cast<StructType>(FT))
          if (FS->getNumElements() == 1) FT = FS->getElementType(0);
      return PointerType::get(FT, 0);
    };

    SmallVector<Type *, 16> NewParams;
    for (Argument *A : OldArgs) {
      auto It = ArgMap.find(A);
      if (It == ArgMap.end()) { NewParams.push_back(A->getType()); continue; }
      auto *ST = cast<StructType>(
          cast<PointerType>(A->getType())->getElementType());
      for (unsigned f = 0; f < It->second.Names.size(); ++f)
        NewParams.push_back(FieldPtrTy(ST, It->second, f));
    }
    FunctionType *NewFT =
        FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF =
        Function::Create(NewFT, Old->getLinkage(), Old->getAddressSpace(),
                         Old->getName() + ".auto_disagg", &M);
    AttributeList OldAL = Old->getAttributes();
    NewF->setAttributes(AttributeList::get(Ctx, OldAL.getFnAttributes(),
                                           OldAL.getRetAttributes(), {}));

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "adis.entry", NewF);
    IRBuilder<> EB(EntryBB);
    ValueToValueMapTy VMap;
    struct Placeholder {
      AllocaInst *AI;
      SmallVector<Argument *, 8> FieldArgs; // declared order
      const Info *I;
      const StructLayout *SL;
      StructType *ST;
    };
    SmallVector<Placeholder, 4> Placeholders;
    SmallVector<unsigned, 4> StreamArgIdxs;
    auto NewArgIt = NewF->arg_begin();
    unsigned NewIdx = 0;
    for (Argument *A : OldArgs) {
      auto It = ArgMap.find(A);
      if (It == ArgMap.end()) {
        NewArgIt->setName(A->getName());
        if (A->hasAttribute(Attribute::NoAlias))
          NewArgIt->addAttr(Attribute::NoAlias);
        VMap[A] = &*NewArgIt;
        ++NewArgIt; ++NewIdx;
        continue;
      }
      auto *ST = cast<StructType>(
          cast<PointerType>(A->getType())->getElementType());
      AllocaInst *AI = EB.CreateAlloca(ST, nullptr, A->getName() + ".adis_tmp");
      VMap[A] = AI;
      Placeholder P{AI, {}, &It->second, DL.getStructLayout(ST), ST};
      for (unsigned f = 0; f < It->second.Names.size(); ++f) {
        NewArgIt->setName(A->getName().str() + "_" + It->second.Names[f]);
        NewArgIt->addAttr(Attribute::NoAlias);
        P.FieldArgs.push_back(&*NewArgIt);
        if (It->second.IsStream[f]) StreamArgIdxs.push_back(NewIdx);
        ++NewArgIt; ++NewIdx;
      }
      Placeholders.push_back(P);
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap, /*ModuleLevelChanges=*/false, Returns);
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF)
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    if (!ClonedEntry) { NewF->eraseFromParent(); return nullptr; }
    // Emit the stream-param markers renameTopStreamArgs consumes, then
    // branch into the cloned body. Top kernels only — a split IMPL keeps
    // plain element pointers (it is not an interface function).
    if (EmitTopMarkers && !StreamArgIdxs.empty()) {
      FunctionType *SPTy =
          FunctionType::get(Type::getVoidTy(Ctx), {I32}, false);
      FunctionCallee SPFn =
          M.getOrInsertFunction("__vxx_top_stream_param", SPTy);
      for (unsigned Idx : StreamArgIdxs)
        EB.CreateCall(SPFn, {ConstantInt::get(I32, Idx)});
    }
    EB.CreateBr(ClonedEntry);

    // Re-point placeholder uses.
    bool Aborted = false;
    for (Placeholder &P : Placeholders) {
      // declared index for an IR field index
      auto DeclFor = [&](unsigned IRIdx, unsigned &DOut) -> bool {
        for (unsigned f = 0; f < P.I->IRIdx.size(); ++f)
          if (P.I->IRIdx[f] == IRIdx) { DOut = f; return true; }
        return false;
      };
      // declared field whose repr(C) offset is 0 (offset-0 bitcasts fold
      // away their GEP)
      unsigned DAtZero = ~0u;
      for (unsigned f = 0; f < P.I->IRIdx.size(); ++f)
        if (P.SL->getElementOffset(P.I->IRIdx[f]) == 0) { DAtZero = f; break; }

      SmallVector<User *, 16> Users;
      {
        // users() yields one entry PER USE — dedupe so an instruction
        // taking the alloca in two operands is visited (and erased) once
        SmallPtrSet<User *, 16> Seen;
        for (User *UU : P.AI->users())
          if (Seen.insert(UU).second) Users.push_back(UU);
      }
      for (User *U : Users) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          if (GEP->getNumIndices() < 2) {
            vxxDbg() << "vxx: adis abort gep<2idx: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
            Aborted = true; break; }
          auto IdxIt = GEP->idx_begin();
          auto *Zero = dyn_cast<ConstantInt>(IdxIt->get());
          auto *FieldC = dyn_cast<ConstantInt>((IdxIt + 1)->get());
          if (!Zero || !Zero->isZero() || !FieldC) {
            vxxDbg() << "vxx: adis abort gep-idx-shape: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
            Aborted = true; break; }
          unsigned D = 0;
          if (!DeclFor((unsigned)FieldC->getZExtValue(), D)) {
            // rustc also spells field accesses through the ZERO-SIZE `[0 x T]`
            // pad fields (`&d.s_in.value` = gep 0, <zsa>, 0 — an i32* at the
            // pad's offset). Resolve by offset: the declared field at the
            // same offset, provided every remaining index is zero and the
            // GEP's result type equals that field's arg type.
            unsigned IRF = (unsigned)FieldC->getZExtValue();
            Type *IRFieldTy = P.ST->getElementType(IRF);
            bool AllZeroRest = true;
            for (auto It2 = IdxIt + 2; It2 != GEP->idx_end(); ++It2) {
              auto *ZC = dyn_cast<ConstantInt>(It2->get());
              if (!ZC || !ZC->isZero()) { AllZeroRest = false; break; }
            }
            const DataLayout &DL2 =
                GEP->getModule()->getDataLayout();
            if (DL2.getTypeAllocSize(IRFieldTy) == 0 && AllZeroRest) {
              uint64_t OffZ = P.SL->getElementOffset(IRF);
              unsigned D2 = ~0u;
              for (unsigned f = 0; f < P.I->DeclOffs.size(); ++f)
                if (P.I->DeclOffs[f] == OffZ) { D2 = f; break; }
              if (D2 != ~0u &&
                  GEP->getType() == P.FieldArgs[D2]->getType()) {
                GEP->replaceAllUsesWith(P.FieldArgs[D2]);
                GEP->eraseFromParent();
                continue;
              }
            }
            vxxDbg() << "vxx: adis abort pad-field gep: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
            Aborted = true;
            break;
          }
          Argument *FArg = P.FieldArgs[D];
          SmallVector<Value *, 4> Rest;
          for (auto It2 = IdxIt + 2; It2 != GEP->idx_end(); ++It2)
            Rest.push_back(It2->get());

          if (Rest.empty() && GEP->getType() == FArg->getType()) {
            GEP->replaceAllUsesWith(FArg);
            GEP->eraseFromParent();
            continue;
          }
          if (P.I->IsStream[D]) {
            // `&d.s_in.value` (rest == [0]) or the `{T}*` intermediate.
            if (Rest.size() == 1) {
              auto *RZ = dyn_cast<ConstantInt>(Rest[0]);
              if (!RZ || !RZ->isZero() ||
                  GEP->getType() != FArg->getType()) {
                vxxDbg() << "vxx: adis abort stream-rest: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
                Aborted = true; break; }
              GEP->replaceAllUsesWith(FArg);
              GEP->eraseFromParent();
              continue;
            }
            if (!Rest.empty()) {
              vxxDbg() << "vxx: adis abort stream-deep-rest: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
              Aborted = true; break; }
            SmallVector<User *, 8> Inner(GEP->user_begin(), GEP->user_end());
            for (User *IU : Inner) {
              if (auto *IG = dyn_cast<GetElementPtrInst>(IU)) {
                if (IG->getNumIndices() == 2 && IG->hasAllZeroIndices() &&
                    IG->getType() == FArg->getType()) {
                  IG->replaceAllUsesWith(FArg);
                  IG->eraseFromParent();
                  continue;
                }
                Aborted = true;
                break;
              }
              if (auto *IB = dyn_cast<BitCastInst>(IU)) {
                IRBuilder<> B(IB);
                Value *Re = B.CreateBitCast(FArg, IB->getType());
                IB->replaceAllUsesWith(Re);
                IB->eraseFromParent();
                continue;
              }
              Aborted = true;
              break;
            }
            if (Aborted) break;
            if (!GEP->use_empty()) {
              vxxDbg() << "vxx: adis abort stream-gep-leftover: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
              Aborted = true; break; }
            GEP->eraseFromParent();
            continue;
          }
          // Non-stream with deeper indexing: gep(FArg, 0, rest...).
          IRBuilder<> B(GEP);
          SmallVector<Value *, 4> Idx;
          Idx.push_back(ConstantInt::get(I64, 0));
          Idx.append(Rest.begin(), Rest.end());
          auto *FPT = cast<PointerType>(FArg->getType());
          Value *NewGEP = B.CreateInBoundsGEP(FPT->getElementType(), FArg,
                                              Idx, GEP->getName());
          if (NewGEP->getType() != GEP->getType()) {
            vxxDbg() << "vxx: adis abort gep-type: "; GEP->print(vxxDbg()); vxxDbg() << "\n";
            Aborted = true; break; }
          GEP->replaceAllUsesWith(NewGEP);
          GEP->eraseFromParent();
          continue;
        }
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          // InstCombine folds OFFSET-0 field GEPs into plain bitcasts, so a
          // cast of the struct pointer is an access to the field at offset 0.
          if (DAtZero == ~0u) {
            vxxDbg() << "vxx: adis abort no-zero-field bitcast: "; BC->print(vxxDbg()); vxxDbg() << "\n";
            Aborted = true; break; }
          Argument *F0 = P.FieldArgs[DAtZero];
          SmallVector<CallInst *, 4> ProbeUses;
          for (User *BU : BC->users()) {
            auto *CI = dyn_cast<CallInst>(BU);
            Function *CF = CI ? CI->getCalledFunction() : nullptr;
            if (CF && CF->getName().startswith("__vxx_auto_disagg"))
              ProbeUses.push_back(CI);
          }
          for (CallInst *CI : ProbeUses) CI->eraseFromParent();
          if (BC->use_empty()) {
            BC->eraseFromParent();
            continue;
          }
          // Only a folded FIELD-0 access may retarget to F0. A WHOLE-STRUCT
          // byte access through the cast (memcpy of sizeof(struct), or a
          // wide load/store) would silently write out of bounds through the
          // field-0 port — abort on any access whose extent exceeds field 0.
          // Other users (fifo-op calls on the folded stream pointer, marker
          // casts) access via the element type and stay legal.
          {
            const DataLayout &DLb = M.getDataLayout();
            uint64_t F0Bytes = DLb.getTypeAllocSize(
                cast<PointerType>(F0->getType())->getElementType());
            bool ExtentOk = true;
            for (User *BU : BC->users()) {
              if (auto *MI = dyn_cast<MemIntrinsic>(BU)) {
                auto *Len = dyn_cast<ConstantInt>(MI->getLength());
                if (!Len || Len->getZExtValue() > F0Bytes) {
                  ExtentOk = false;
                  break;
                }
              } else if (auto *LI = dyn_cast<LoadInst>(BU)) {
                if (DLb.getTypeAllocSize(LI->getType()) > F0Bytes) {
                  ExtentOk = false;
                  break;
                }
              } else if (auto *SI = dyn_cast<StoreInst>(BU)) {
                if (SI->getPointerOperand() == BC &&
                    DLb.getTypeAllocSize(
                        SI->getValueOperand()->getType()) > F0Bytes) {
                  ExtentOk = false;
                  break;
                }
              }
            }
            if (!ExtentOk) {
              vxxDbg() << "vxx: adis abort bitcast-extent on "
                       << Old->getName() << "\n";
              Aborted = true;
              break;
            }
          }
          IRBuilder<> B(BC);
          Value *Re = B.CreateBitCast(F0, BC->getType());
          BC->replaceAllUsesWith(Re);
          BC->eraseFromParent();
          continue;
        }
        if (auto *CI = dyn_cast<CallInst>(U)) {
          // Placeholder passed to a call: only legal when the callee was
          // split for the adapter path (CallRewrites) — swap the whole-struct
          // operand for the per-field args (identical FieldPtrTy types).
          Function *Callee = CI->getCalledFunction();
          auto RIt = Callee ? CallRewrites.find(Callee) : CallRewrites.end();
          if (RIt != CallRewrites.end()) {
            SmallVector<Value *, 8> NewArgs;
            for (unsigned oi = 0; oi < CI->arg_size(); ++oi) {
              Value *Op = CI->getArgOperand(oi);
              if (Op == P.AI) {
                for (Argument *FA : P.FieldArgs) NewArgs.push_back(FA);
              } else {
                NewArgs.push_back(Op);
              }
            }
            FunctionType *NFT = RIt->second->getFunctionType();
            bool BadShape = NewArgs.size() != NFT->getNumParams();
            for (unsigned oi = 0; !BadShape && oi < NewArgs.size(); ++oi)
              if (NewArgs[oi]->getType() != NFT->getParamType(oi))
                BadShape = true;
            if (!BadShape) {
              IRBuilder<> B(CI);
              CallInst *NC = B.CreateCall(RIt->second, NewArgs);
              NC->setCallingConv(RIt->second->getCallingConv());
              if (!CI->getType()->isVoidTy()) CI->replaceAllUsesWith(NC);
              CI->eraseFromParent();
              continue;
            }
          }
          vxxDbg() << "vxx: adis abort call-user: "; U->print(vxxDbg()); vxxDbg() << "\n";
          Aborted = true;
          break;
        }
        vxxDbg() << "vxx: adis abort other-user: "; U->print(vxxDbg()); vxxDbg() << "\n";
        Aborted = true;
        break;
      }
      if (Aborted) break;
    }
    bool AllocasDead = !Aborted;
    for (Placeholder &P : Placeholders)
      if (!P.AI->use_empty()) AllocasDead = false;
    if (!AllocasDead) {
      NewF->eraseFromParent();
      vxxDbg() << "vxx: autoDisaggStructRefKernelSig aborted on "
               << Old->getName() << " (unrecognised port access shape)\n";
      return nullptr;
    }
    for (Placeholder &P : Placeholders) P.AI->eraseFromParent();

    if (ReplaceOld) {
      std::string OldName = Old->getName().str();
      Old->eraseFromParent();
      NewF->setName(OldName);
      vxxDbg() << "vxx: auto-disaggregated struct-ref port(s) on "
               << OldName << "\n";
    } else {
      NewF->setName(Old->getName().str() + ".split");
      vxxDbg() << "vxx: auto-disaggregated struct-ref arg on called fn "
               << NewF->getName() << "\n";
    }
    return NewF;
  };

  // ── cpp_proxy struct_ports: split the OUTLINED `__<top>_impl` FIRST so
  // the design-top loop below can rewrite its call-shaped body through
  // CallRewrites; the marked `rust_<top>` adapter is regenerated afterwards
  // as a direct split call (its Rust-level assemble/drain body — whose
  // struct local trips HLS 214-319 and whose local-stream drain would
  // deadlock in RTL — is discarded wholesale).
  struct AdapterJob {
    Function *Adapter;
    Function *OldImpl;
    Function *NewImpl;
    unsigned ImplStructArgIdx;
    const Info *I;
  };
  SmallVector<AdapterJob, 2> AdapterJobs;
  if (Function *AdMark = M.getFunction("__vxx_split_adapter")) {
    SmallPtrSet<Function *, 4> MarkedAdapters;
    for (User *U : AdMark->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        MarkedAdapters.insert(CI->getParent()->getParent());
    for (auto &E : ImplInfos) {
      Argument *ImplArg = E.first;
      Function *ImplF = ImplArg->getParent();
      Function *Adapter = nullptr;
      for (User *U : ImplF->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          Function *Caller = CI->getParent()->getParent();
          if (MarkedAdapters.count(Caller)) {
            Adapter = Caller;
            break;
          }
        }
      }
      if (!Adapter) {
        vxxDbg() << "vxx: split_adapter — impl " << ImplF->getName()
                 << " has probe info but no marked caller\n";
        continue;
      }
      // PRECHECK adapter viability before splitting: the split impl will
      // have arg_size-1+NF params; committing the split (CallRewrites)
      // and then bailing on the adapter would leave a hybrid module.
      unsigned WantParams = ImplF->arg_size() - 1 + E.second.Names.size();
      if (Adapter->arg_size() != WantParams) {
        vxxDbg() << "vxx: split_adapter arg-count precheck failed on "
                 << Adapter->getName() << " (" << Adapter->arg_size()
                 << " vs " << WantParams << ")\n";
        continue;
      }
      DenseMap<Argument *, Info> ImplMap;
      ImplMap[ImplArg] = E.second;
      Function *NewImpl = SplitOne(ImplF, ImplMap, /*EmitTopMarkers=*/false,
                                   /*ReplaceOld=*/false);
      if (!NewImpl) continue;
      CallRewrites[ImplF] = NewImpl;
      AdapterJobs.push_back(
          {Adapter, ImplF, NewImpl, ImplArg->getArgNo(), &E.second});
      Changed = true;
    }
  }

  DenseMap<Function *, SmallVector<Argument *, 4>> ByFn;
  for (auto &E : Targets) ByFn[E.first->getParent()].push_back(E.first);
  for (auto &Pair : ByFn) {
    Function *Old = Pair.first;
    DenseMap<Argument *, Info> ArgMap;
    for (Argument *A : Pair.second) ArgMap[A] = Targets[A];
    if (SplitOne(Old, ArgMap, /*EmitTopMarkers=*/true, /*ReplaceOld=*/true))
      Changed = true;
  }

  // Regenerate each marked adapter: per-field TYPED args — stream fields
  // get the `class.hls::stream<elem>*` the C++ `hls::stream<T>&` decl
  // matches (an int*/void* erasure at this boundary makes the apatb
  // classify the port as A2Stream and the RTL sim starves), array fields
  // get the C-decayed ELEMENT pointer. Body = one direct split-impl call.
  for (AdapterJob &J : AdapterJobs) {
    Function *Old = J.Adapter;
    Function *NewImpl = J.NewImpl;
    unsigned NF = J.I->Names.size();
    FunctionType *ImplFT = NewImpl->getFunctionType();
    if (Old->arg_size() != ImplFT->getNumParams()) {
      vxxDbg() << "vxx: split_adapter arg-count mismatch on "
               << Old->getName() << " (" << Old->arg_size() << " vs "
               << ImplFT->getNumParams() << ")\n";
      continue;
    }
    SmallVector<Type *, 8> ATys;
    for (unsigned i = 0; i < ImplFT->getNumParams(); ++i) {
      Type *PT = ImplFT->getParamType(i);
      bool InFields = i >= J.ImplStructArgIdx && i < J.ImplStructArgIdx + NF;
      bool IsStreamField = InFields && J.I->IsStream[i - J.ImplStructArgIdx];
      if (IsStreamField) {
        Type *ElemT = cast<PointerType>(PT)->getElementType();
        std::string TName;
        if (auto *IT = dyn_cast<IntegerType>(ElemT)) {
          switch (IT->getBitWidth()) {
            case 8:  TName = "char"; break;
            case 16: TName = "short"; break;
            case 32: TName = "int"; break;
            case 64: TName = "long"; break;
            default: TName = "i" + std::to_string(IT->getBitWidth()); break;
          }
        } else if (ElemT->isFloatTy()) {
          TName = "float";
        } else {
          TName = "double";
        }
        std::string SName = "class.hls::stream<" + TName + ">";
        StructType *SST = M.getTypeByName(SName);
        if (!SST) SST = StructType::create(Ctx, {ElemT}, SName, false);
        ATys.push_back(PointerType::get(SST, 0));
      } else if (InFields) {
        Type *PointeeT = cast<PointerType>(PT)->getElementType();
        if (auto *AT = dyn_cast<ArrayType>(PointeeT))
          ATys.push_back(PointerType::get(AT->getElementType(), 0));
        else
          ATys.push_back(PT);
      } else {
        ATys.push_back(Old->getFunctionType()->getParamType(i));
      }
    }
    FunctionType *NewFT = FunctionType::get(Old->getReturnType(), ATys, false);
    Function *NewAd =
        Function::Create(NewFT, Old->getLinkage(), Old->getAddressSpace(),
                         Old->getName() + ".regen", &M);
    AttributeList OldAL = Old->getAttributes();
    NewAd->setAttributes(AttributeList::get(Ctx, OldAL.getFnAttributes(),
                                            OldAL.getRetAttributes(), {}));
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", NewAd);
    IRBuilder<> B(BB);
    SmallVector<Value *, 8> CallArgs;
    unsigned ai = 0;
    for (Argument &A : NewAd->args()) {
      A.setName(Old->getArg(ai)->getName());
      Type *Want = ImplFT->getParamType(ai);
      Value *V = &A;
      if (V->getType() != Want) {
        auto *APT = dyn_cast<PointerType>(V->getType());
        auto *SST = APT ? dyn_cast<StructType>(APT->getElementType()) : nullptr;
        if (SST && SST->getName().startswith("class.hls::stream<")) {
          Value *Idx[] = {ConstantInt::get(I64, 0), ConstantInt::get(I32, 0)};
          V = B.CreateInBoundsGEP(SST, V, Idx, A.getName() + ".inner");
          if (V->getType() != Want) V = B.CreateBitCast(V, Want);
        } else {
          V = B.CreateBitCast(V, Want);
        }
      }
      CallArgs.push_back(V);
      ++ai;
    }
    CallInst *Call = B.CreateCall(NewImpl, CallArgs);
    Call->setCallingConv(NewImpl->getCallingConv());
    if (Old->getReturnType()->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(Call);
    if (!Old->use_empty()) {
      vxxDbg() << "vxx: split_adapter " << Old->getName()
               << " unexpectedly has callers — keeping both\n";
      NewAd->setName(Old->getName().str() + ".split_regen");
      continue;
    }
    std::string OldName = Old->getName().str();
    Old->eraseFromParent(); // verified: the adapter has no bc callers
    NewAd->setName(OldName);
    vxxDbg() << "vxx: split_adapter regenerated " << OldName << " -> call "
             << NewImpl->getName() << "\n";
    if (J.OldImpl->use_empty()) J.OldImpl->eraseFromParent();
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_split_adapter");

  // Consume any remaining probe markers (aborted transforms, non-top
  // callers) so they never reach the HLS backend.
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_auto_disagg");
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_auto_disagg_f");
  return Changed;
}

// ===========================================================================
// `__vxx_stream_array(ptr, m)` — array-of-streams kernel param
// (`&mut [Stream<T>; M]`, LLVM `[M x T]*`). The C++ frontend flattens
// `hls::stream<T> p[M]` args into M per-stream args `p_0..p_{M-1}`
// (a.pp.bc of using_array_of_streams) — reproduce that:
//   * marker on an UNCALLED TOP arg: split the arg in place (clone with M
//     elem-ptr args, re-point the constant-index GEPs, replicate any
//     `__vxx_axis` marker per split arg).
//   * marker on a CALLED impl arg (cpp_proxy; `__vxx_split_adapter` caller):
//     split the impl the same way and REGENERATE the adapter as a direct
//     per-stream call (its host-side alias/assemble body is discarded).
bool splitStreamArrayKernelSig(Module &M) {
  Function *Marker = M.getFunction("__vxx_stream_array");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  bool Changed = false;

  struct Target {
    Argument *Arg;
    unsigned M;
  };
  SmallVector<Target, 4> Tops, Impls;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    MarkerCalls.push_back(CI);
    Argument *A = resolveMarkerArg(CI->getArgOperand(0));
    auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!A || !MC) continue;
    auto *PT = dyn_cast<PointerType>(A->getType());
    auto *AT = PT ? dyn_cast<ArrayType>(PT->getElementType()) : nullptr;
    if (!AT || AT->getNumElements() != MC->getZExtValue()) {
      vxxDbg() << "vxx: stream_array skipped (arg shape) on "
               << A->getParent()->getName() << "\n";
      continue;
    }
    Function *F = A->getParent();
    if (F->hasFnAttribute("fpga.top.func") && F->use_empty())
      Tops.push_back({A, (unsigned)MC->getZExtValue()});
    else
      Impls.push_back({A, (unsigned)MC->getZExtValue()});
  }
  for (CallInst *CI : MarkerCalls) CI->eraseFromParent();
  if (Tops.empty() && Impls.empty()) {
    hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_stream_array");
    return false;
  }

  // Split every listed arg of `Old` into per-stream elem-ptr args. Returns
  // the clone or null on abort.
  DenseMap<Function *, Function *> CallRewrites;
  auto SplitOne = [&](Function *Old, ArrayRef<Target> Args,
                      bool ReplaceOld) -> Function * {
    DenseMap<Argument *, unsigned> Ms;
    for (auto &T : Args)
      if (T.Arg->getParent() == Old) Ms[T.Arg] = T.M;
    if (Ms.empty()) return nullptr;

    SmallVector<Type *, 8> NewParams;
    for (Argument &A : Old->args()) {
      auto It = Ms.find(&A);
      if (It == Ms.end()) {
        NewParams.push_back(A.getType());
        continue;
      }
      auto *AT = cast<ArrayType>(
          cast<PointerType>(A.getType())->getElementType());
      for (unsigned k = 0; k < It->second; ++k)
        NewParams.push_back(PointerType::get(AT->getElementType(), 0));
    }
    Function *NewF = Function::Create(
        FunctionType::get(Old->getReturnType(), NewParams, false),
        Old->getLinkage(), Old->getAddressSpace(),
        Old->getName() + ".sa_split", &M);
    AttributeList OldAL = Old->getAttributes();
    NewF->setAttributes(AttributeList::get(Ctx, OldAL.getFnAttributes(),
                                           OldAL.getRetAttributes(), {}));

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "sa.entry", NewF);
    IRBuilder<> EB(EntryBB);
    ValueToValueMapTy VMap;
    struct PH {
      AllocaInst *AI;
      SmallVector<Argument *, 4> Elems;
    };
    SmallVector<PH, 2> PHs;
    SmallVector<SmallVector<Argument *, 4>, 2> AllSplitArgs;
    auto NIt = NewF->arg_begin();
    for (Argument &A : Old->args()) {
      auto It = Ms.find(&A);
      if (It == Ms.end()) {
        NIt->setName(A.getName()); // copy, not takeName: Old must keep its
                                   // arg names in case of a later abort
        VMap[&A] = &*NIt;
        ++NIt;
        continue;
      }
      auto *AT = cast<ArrayType>(
          cast<PointerType>(A.getType())->getElementType());
      AllocaInst *AI = EB.CreateAlloca(AT, nullptr, A.getName() + ".sa_tmp");
      VMap[&A] = AI;
      PH P{AI, {}};
      for (unsigned k = 0; k < It->second; ++k) {
        NIt->setName(A.getName().str() + "_" + std::to_string(k));
        NIt->addAttr(Attribute::NoAlias);
        P.Elems.push_back(&*NIt);
        ++NIt;
      }
      AllSplitArgs.push_back(P.Elems);
      PHs.push_back(P);
    }
    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap, false, Returns);
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF)
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    if (!ClonedEntry) { NewF->eraseFromParent(); return nullptr; }
    EB.CreateBr(ClonedEntry);

    bool Aborted = false;
    for (PH &P : PHs) {
      // classify first, mutate after
      SmallVector<std::pair<Instruction *, Value *>, 8> Repls;
      SmallVector<std::pair<CallInst *, unsigned>, 4> AxisCalls;
      SmallVector<User *, 16> Users;
      {
        // users() yields one entry PER USE — dedupe so an instruction
        // taking the alloca in two operands is visited (and erased) once
        SmallPtrSet<User *, 16> Seen;
        for (User *UU : P.AI->users())
          if (Seen.insert(UU).second) Users.push_back(UU);
      }
      for (User *U : Users) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // gep [0, k, (trailing zeros)]
          if (GEP->getNumIndices() < 2) { Aborted = true; break; }
          auto It2 = GEP->idx_begin();
          auto *Z = dyn_cast<ConstantInt>(It2->get());
          auto *K = dyn_cast<ConstantInt>((It2 + 1)->get());
          bool RestZero = true;
          for (auto It3 = It2 + 2; It3 != GEP->idx_end(); ++It3) {
            auto *ZC = dyn_cast<ConstantInt>(It3->get());
            if (!ZC || !ZC->isZero()) { RestZero = false; break; }
          }
          if (Z && Z->isZero() && !K && RestZero && GEP->getNumIndices() == 2) {
            // DYNAMIC element index (`s[j]` in a rolled loop — rustc does
            // not unroll it the way the C++ frontend does): lower to a
            // SELECT chain over the split args; the downstream
            // splitSelectPtrFifoOps pass re-expands fifo ops on selects
            // into per-case branches (the free_running recipe), which is
            // the C++ pre-reflow shape for dynamic stream selection.
            Value *Idx = (It2 + 1)->get();
            IRBuilder<> B(GEP);
            Type *IdxT = Idx->getType();
            Value *Sel = P.Elems[P.Elems.size() - 1];
            for (int k = (int)P.Elems.size() - 2; k >= 0; --k) {
              Value *Cmp = B.CreateICmpEQ(
                  Idx, ConstantInt::get(IdxT, (uint64_t)k));
              Sel = B.CreateSelect(Cmp, P.Elems[(unsigned)k], Sel);
            }
            if (Sel->getType() != GEP->getType()) {
              vxxDbg() << "vxx: stream_array abort dyn-gep-type\n";
              Aborted = true;
              break;
            }
            Repls.push_back({GEP, Sel});
            continue;
          }
          if (!Z || !Z->isZero() || !K || !RestZero ||
              K->getZExtValue() >= P.Elems.size()) {
            vxxDbg() << "vxx: stream_array abort gep: ";
            GEP->print(vxxDbg());
            vxxDbg() << "\n";
            Aborted = true;
            break;
          }
          Argument *EA = P.Elems[(unsigned)K->getZExtValue()];
          if (GEP->getType() != EA->getType()) {
            vxxDbg() << "vxx: stream_array abort gep-type: ";
            GEP->print(vxxDbg());
            vxxDbg() << "\n";
            Aborted = true;
            break;
          }
          Repls.push_back({GEP, EA});
          continue;
        }
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          // the array-level `__vxx_axis` marker reaches through a cast —
          // replicate it per split arg
          bool CastOk = true;
          for (User *BU : BC->users()) {
            auto *CI = dyn_cast<CallInst>(BU);
            Function *CF = CI ? CI->getCalledFunction() : nullptr;
            if (CF && CF->getName() == "__vxx_axis") {
              AxisCalls.push_back({CI, 0});
              continue;
            }
            CastOk = false;
            break;
          }
          if (!CastOk) {
            vxxDbg() << "vxx: stream_array abort cast-user on "
                     << Old->getName() << "\n";
            Aborted = true;
            break;
          }
          continue;
        }
        if (auto *CI = dyn_cast<CallInst>(U)) {
          Function *Callee = CI->getCalledFunction();
          auto RIt = Callee ? CallRewrites.find(Callee) : CallRewrites.end();
          if (RIt != CallRewrites.end()) {
            // whole-array operand into a split callee: expand to elems
            SmallVector<Value *, 8> NewArgs;
            for (unsigned oi = 0; oi < CI->arg_size(); ++oi) {
              Value *Op = CI->getArgOperand(oi);
              if (Op == P.AI)
                for (Argument *EA : P.Elems) NewArgs.push_back(EA);
              else
                NewArgs.push_back(Op);
            }
            FunctionType *NFT = RIt->second->getFunctionType();
            bool Bad = NewArgs.size() != NFT->getNumParams();
            for (unsigned oi = 0; !Bad && oi < NewArgs.size(); ++oi)
              if (NewArgs[oi]->getType() != NFT->getParamType(oi)) Bad = true;
            if (!Bad) {
              IRBuilder<> B(CI);
              CallInst *NC = B.CreateCall(RIt->second, NewArgs);
              NC->setCallingConv(RIt->second->getCallingConv());
              if (!CI->getType()->isVoidTy()) CI->replaceAllUsesWith(NC);
              Repls.push_back({CI, nullptr});
              continue;
            }
          }
          vxxDbg() << "vxx: stream_array abort call-user on "
                   << Old->getName() << "\n";
          Aborted = true;
          break;
        }
        vxxDbg() << "vxx: stream_array abort other-user on "
                 << Old->getName() << "\n";
        Aborted = true;
        break;
      }
      if (Aborted) break;
      for (auto &RP : Repls) {
        if (RP.second) RP.first->replaceAllUsesWith(RP.second);
        RP.first->eraseFromParent();
      }
      // replicate the axis marker per split arg, then drop the original
      for (auto &AC : AxisCalls) {
        CallInst *CI = AC.first;
        IRBuilder<> B(CI);
        Function *AxisF = CI->getCalledFunction();
        Type *WantT = AxisF->getFunctionType()->getParamType(0);
        for (Argument *EA : P.Elems) {
          Value *Cast = B.CreateBitCast(EA, WantT);
          B.CreateCall(AxisF, {Cast});
        }
        CI->eraseFromParent();
      }
      // remaining bitcast users of the alloca are now dead marker casts
      SmallVector<Instruction *, 4> DeadCasts;
      for (User *U : P.AI->users())
        if (auto *BC = dyn_cast<BitCastInst>(U))
          if (BC->use_empty()) DeadCasts.push_back(BC);
      for (Instruction *I : DeadCasts) I->eraseFromParent();
    }
    bool AllocasDead = !Aborted;
    for (PH &P : PHs)
      if (!P.AI->use_empty()) AllocasDead = false;
    if (!AllocasDead) {
      NewF->eraseFromParent();
      vxxDbg() << "vxx: splitStreamArrayKernelSig aborted on "
               << Old->getName() << "\n";
      return nullptr;
    }
    for (PH &P : PHs) P.AI->eraseFromParent();
    if (ReplaceOld) {
      std::string Nm = Old->getName().str();
      Old->eraseFromParent();
      NewF->setName(Nm);
      vxxDbg() << "vxx: stream-array split on top " << Nm << "\n";
    } else {
      vxxDbg() << "vxx: stream-array split on impl " << NewF->getName()
               << "\n";
    }
    (void)AllSplitArgs;
    return NewF;
  };

  // impls first (adapter path), so the design-top loop can rewrite calls
  DenseMap<Function *, SmallVector<Target, 2>> ByImpl;
  for (auto &T : Impls) ByImpl[T.Arg->getParent()].push_back(T);
  struct AJob {
    Function *OldImpl;
    Function *NewImpl;
    Function *Adapter; // marked caller (may be null: no adapter to regen)
  };
  SmallVector<AJob, 2> AJobs;
  SmallPtrSet<Function *, 4> MarkedAdapters;
  if (Function *AdM = M.getFunction("__vxx_split_adapter"))
    for (User *U : AdM->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        MarkedAdapters.insert(CI->getParent()->getParent());
  for (auto &KV : ByImpl) {
    Function *ImplF = KV.first;
    Function *Adapter = nullptr;
    for (User *U : ImplF->users())
      if (auto *CI = dyn_cast<CallInst>(U)) {
        Function *Caller = CI->getParent()->getParent();
        if (MarkedAdapters.count(Caller)) { Adapter = Caller; break; }
      }
    // PRECHECK adapter viability before splitting (a post-split bail would
    // leave a half-committed module: split impl live, adapter unrewritten)
    if (Adapter) {
      unsigned WantParams = ImplF->arg_size();
      for (auto &T : KV.second) WantParams += T.M - 1;
      if (Adapter->arg_size() != WantParams) {
        vxxDbg() << "vxx: stream_array adapter arg-count precheck failed on "
                 << Adapter->getName() << " (" << Adapter->arg_size()
                 << " vs " << WantParams << ")\n";
        continue;
      }
    }
    Function *NewImpl = SplitOne(ImplF, KV.second, /*ReplaceOld=*/false);
    if (!NewImpl) continue;
    CallRewrites[ImplF] = NewImpl;
    AJobs.push_back({ImplF, NewImpl, Adapter});
    Changed = true;
  }

  DenseMap<Function *, SmallVector<Target, 2>> ByTop;
  for (auto &T : Tops) ByTop[T.Arg->getParent()].push_back(T);
  for (auto &KV : ByTop) {
    if (SplitOne(KV.first, KV.second, /*ReplaceOld=*/true)) Changed = true;
  }

  // Regenerate marked adapters calling a split impl: per-stream args map 1:1
  // (the macro's adapter emits M params per array in the same positions;
  // the count was prechecked before the impl split).
  {
    for (AJob &J : AJobs) {
      Function *Adapter = J.Adapter;
      if (!Adapter) continue;
      FunctionType *IFT = J.NewImpl->getFunctionType();
      if (Adapter->arg_size() != IFT->getNumParams()) {
        vxxDbg() << "vxx: stream_array adapter arg-count mismatch on "
                 << Adapter->getName() << "\n";
        continue;
      }
      SmallVector<Type *, 8> ATys;
      for (unsigned i = 0; i < IFT->getNumParams(); ++i)
        ATys.push_back(IFT->getParamType(i));
      Function *NewAd = Function::Create(
          FunctionType::get(Adapter->getReturnType(), ATys, false),
          Adapter->getLinkage(), Adapter->getAddressSpace(),
          Adapter->getName() + ".regen", &M);
      AttributeList AL = Adapter->getAttributes();
      NewAd->setAttributes(AttributeList::get(Ctx, AL.getFnAttributes(),
                                              AL.getRetAttributes(), {}));
      BasicBlock *BB = BasicBlock::Create(Ctx, "entry", NewAd);
      IRBuilder<> B(BB);
      SmallVector<Value *, 8> CallArgs;
      unsigned ai = 0;
      for (Argument &A : NewAd->args()) {
        A.setName(Adapter->getArg(ai) ? Adapter->getArg(ai)->getName() : "");
        CallArgs.push_back(&A);
        ++ai;
      }
      CallInst *Call = B.CreateCall(J.NewImpl, CallArgs);
      Call->setCallingConv(J.NewImpl->getCallingConv());
      if (Adapter->getReturnType()->isVoidTy())
        B.CreateRetVoid();
      else
        B.CreateRet(Call);
      if (!Adapter->use_empty()) {
        vxxDbg() << "vxx: stream_array adapter " << Adapter->getName()
                 << " unexpectedly has callers — keeping both\n";
        NewAd->setName(Adapter->getName().str() + ".sa_regen");
        Changed = true;
        continue;
      }
      std::string Nm = Adapter->getName().str();
      Adapter->eraseFromParent();
      NewAd->setName(Nm);
      vxxDbg() << "vxx: stream_array regenerated adapter " << Nm << "\n";
      if (J.OldImpl->use_empty()) J.OldImpl->eraseFromParent();
      Changed = true;
    }
  }

  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_stream_array");
  (void)I32;
  return Changed;
}


// ===========================================================================
// rustc lowers a small two-field by-value struct arg with the ScalarPair ABI:
// `fn top(bias: CmpxC)` with `CmpxC { re: i64, im: i64 }` arrives as TWO
// scalar args `i64 %bias.0, i64 %bias.1`. Xilinx clang instead packs the whole
// struct into ONE by-value integer (systolic_fir_using_complex a.pp.bc:
// `i128 %bias`, field 0 = trunc / field 1 = PartSelect at bit 64), so the
// pre-reflow signature — and hence SpecInterface/SpecBitsMap and the RTL port
// — is a single 2W-bit scalar. Re-pack the rustc pair to that form.
//
// Gate: uncalled top kernels only (same rule as narrowDspCplxPortTypes). The
// cpp_proxy cosim build calls the kernel from the generated `rust_<top>`
// adapter with the ScalarPair ABI; rewriting the callee there would break the
// call. The Stage A / real-kernel build has no internal callers.
bool packScalarPairByValueArgs(Module &M) {
  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  SmallVector<Function *, 4> Tops;
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasFnAttribute("fpga.top.func") && F.use_empty())
      Tops.push_back(&F);
  for (Function *Old : Tops) {
    SmallVector<Argument *, 8> A;
    for (Argument &Ar : Old->args()) A.push_back(&Ar);
    // Pair start index → field width (equal-width integer halves only).
    DenseMap<unsigned, unsigned> PairAt;
    for (unsigned i = 0; i + 1 < A.size(); ++i) {
      auto *T0 = dyn_cast<IntegerType>(A[i]->getType());
      auto *T1 = dyn_cast<IntegerType>(A[i + 1]->getType());
      if (!T0 || !T1 || T0->getBitWidth() != T1->getBitWidth()) continue;
      StringRef N0 = A[i]->getName(), N1 = A[i + 1]->getName();
      if (!N0.endswith(".0") || !N1.endswith(".1")) continue;
      if (N0.drop_back(2) != N1.drop_back(2) || N0.size() <= 2) continue;
      PairAt[i] = T0->getBitWidth();
      ++i;
    }
    if (PairAt.empty()) continue;

    SmallVector<Type *, 8> NewParams;
    for (unsigned i = 0; i < A.size(); ++i) {
      auto It = PairAt.find(i);
      if (It != PairAt.end()) {
        NewParams.push_back(IntegerType::get(Ctx, It->second * 2));
        ++i; // skip the .1 half
      } else {
        NewParams.push_back(A[i]->getType());
      }
    }
    FunctionType *NewFT =
        FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF =
        Function::Create(NewFT, Old->getLinkage(), Old->getAddressSpace(),
                         Old->getName() + ".pairpack", &M);
    // Copy only fn/ret attributes — param attributes shift positions.
    AttributeList OldAL = Old->getAttributes();
    NewF->setAttributes(AttributeList::get(Ctx, OldAL.getFnAttributes(),
                                           OldAL.getRetAttributes(), {}));

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "pairpack.entry", NewF);
    IRBuilder<> EB(EntryBB);
    ValueToValueMapTy VMap;
    auto NewArgIt = NewF->arg_begin();
    for (unsigned i = 0; i < A.size(); ++i, ++NewArgIt) {
      auto It = PairAt.find(i);
      if (It == PairAt.end()) {
        NewArgIt->setName(A[i]->getName());
        VMap[A[i]] = &*NewArgIt;
        continue;
      }
      unsigned W = It->second;
      Type *HalfTy = IntegerType::get(Ctx, W);
      StringRef Base = A[i]->getName().drop_back(2);
      NewArgIt->setName(Base);
      // C++ layout: field 0 in the low half, field 1 at bit W.
      Value *Lo = EB.CreateTrunc(&*NewArgIt, HalfTy, Base + ".0");
      Value *HiShift = EB.CreateLShr(
          &*NewArgIt, ConstantInt::get(NewArgIt->getType(), W));
      Value *Hi = EB.CreateTrunc(HiShift, HalfTy, Base + ".1");
      VMap[A[i]] = Lo;
      VMap[A[i + 1]] = Hi;
      ++i; // consumed the .1 half
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap,
                      /*ModuleLevelChanges=*/false, Returns);
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF)
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    if (!ClonedEntry) {
      NewF->eraseFromParent();
      continue;
    }
    EB.CreateBr(ClonedEntry);

    std::string OldName = Old->getName().str();
    Old->eraseFromParent(); // use_empty by the gate above
    NewF->setName(OldName);
    vxxDbg() << "vxx: packed ScalarPair by-value arg(s) on top kernel "
             << OldName << "\n";
    Changed = true;
  }
  return Changed;
}

bool disaggCompletePartitionKernelSig(Module &M) {
  Function *PartMarker = M.getFunction("__vxx_array_partition");
  if (!PartMarker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);

  // Collect args that need complete-partition disagg.
  // arg → (NumElements, ElemType, dim)
  DenseMap<Argument*, std::tuple<uint64_t, Type*, uint64_t>> Targets;
  SmallVector<CallInst*, 8> MarkersToDrop;
  for (User *U : PartMarker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 4) continue;
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *FactorC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    auto *DimC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
    if (!KindC || !FactorC || !DimC) continue;
    if (KindC->getZExtValue() != 2) continue;  // Rust "complete"
    if (FactorC->getZExtValue() != 0) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Function *F = Arg->getParent();
    if (!F->hasFnAttribute("fpga.top.func")) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    // Accept `[N x T]*` or pre-decayed `T*` + dim.hint
    Type *Pointee = PT->getElementType();
    uint64_t N = 0;
    Type *ET = nullptr;
    if (auto *AT = dyn_cast<ArrayType>(Pointee)) {
      N = AT->getNumElements();
      ET = AT->getElementType();
    } else {
      // Pre-decayed: read fpga.decayed.dim.hint
      AttributeSet PA = F->getAttributes().getParamAttributes(Arg->getArgNo());
      if (!PA.hasAttribute("fpga.decayed.dim.hint")) continue;
      Attribute H = PA.getAttribute("fpga.decayed.dim.hint");
      H.getValueAsString().getAsInteger(10, N);
      ET = Pointee;
    }
    if (N == 0 || !ET) continue;
    // Cap N to avoid runaway disagg (e.g. partitioning a 1024-element array).
    if (N > 64) continue;
    Targets[Arg] = {N, ET, DimC->getZExtValue()};
    MarkersToDrop.push_back(CI);
  }
  if (Targets.empty()) return false;

  // Group targets by parent function — disagg one function at a time.
  DenseMap<Function*, SmallVector<Argument*, 4>> ByFn;
  for (auto &E : Targets) ByFn[E.first->getParent()].push_back(E.first);

  bool Changed = false;
  for (auto &Pair : ByFn) {
    Function *Old = Pair.first;
    auto &PartArgs = Pair.second;
    // Sort PartArgs by argno for stable order
    std::sort(PartArgs.begin(), PartArgs.end(),
              [](Argument *A, Argument *B) { return A->getArgNo() < B->getArgNo(); });
    DenseSet<Argument*> PartSet;
    for (Argument *A : PartArgs) PartSet.insert(A);

    SmallVector<Type *, 16> NewParams;
    struct Slot {
      bool IsPart;
      Argument *Old;
      uint64_t N;
      Type *ET;
      uint64_t Dim;
    };
    SmallVector<Slot, 16> Slots;
    for (Argument &A : Old->args()) {
      Slot S{false, &A, 0, nullptr, 0};
      auto It = Targets.find(&A);
      if (It != Targets.end()) {
        S.IsPart = true;
        S.N = std::get<0>(It->second);
        S.ET = std::get<1>(It->second);
        S.Dim = std::get<2>(It->second);
        for (uint64_t i = 0; i < S.N; ++i)
          NewParams.push_back(PointerType::get(S.ET, 0));
      } else {
        NewParams.push_back(A.getType());
      }
      Slots.push_back(S);
    }

    FunctionType *NewFT = FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF = Function::Create(NewFT, Old->getLinkage(),
                                       Old->getAddressSpace(),
                                       Old->getName() + ".part_disagg", &M);
    NewF->copyAttributesFrom(Old);

    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "part.entry", NewF);
    IRBuilder<> EB(EntryBB);

    ValueToValueMapTy VMap;
    auto NewArgIt = NewF->arg_begin();
    SmallVector<std::tuple<AllocaInst*, SmallVector<Argument*, 16>, Type*, uint64_t>, 4> PartAllocas;
    for (Slot &S : Slots) {
      if (!S.IsPart) {
        NewArgIt->setName(S.Old->getName());
        if (S.Old->hasAttribute(Attribute::NoAlias))
          NewArgIt->addAttr(Attribute::NoAlias);
        VMap[S.Old] = &*NewArgIt;
        ++NewArgIt;
      } else {
        // Create alloca of [N x T] as bridge
        ArrayType *AT = ArrayType::get(S.ET, S.N);
        AllocaInst *AL = EB.CreateAlloca(AT, nullptr,
                                         S.Old->getName() + ".part_tmp");
        // VMap old arg to alloca — body GEPs work transparently.
        // But the body might have decayed pre-shape `T*` — we need a bitcast
        // to that type if needed.
        Value *MapVal = AL;
        if (S.Old->getType() != AL->getType()) {
          // Pre-decayed case: cast alloca to T*
          MapVal = EB.CreateBitCast(AL, S.Old->getType());
        }
        VMap[S.Old] = MapVal;
        SmallVector<Argument*, 16> ChanArgs;
        for (uint64_t i = 0; i < S.N; ++i) {
          NewArgIt->setName(S.Old->getName().str() + "_" + std::to_string(i));
          NewArgIt->addAttr(Attribute::NoAlias);
          ChanArgs.push_back(&*NewArgIt);
          ++NewArgIt;
        }
        PartAllocas.push_back({AL, ChanArgs, S.ET, S.N});
        // Pre-load: at entry, load each new arg → store into alloca field i
        for (uint64_t i = 0; i < S.N; ++i) {
          Value *FieldPtr = EB.CreateInBoundsGEP(AT, AL,
              {ConstantInt::get(I64, 0), ConstantInt::get(I64, i)});
          LoadInst *LD = EB.CreateLoad(S.ET, ChanArgs[i]);
          EB.CreateStore(LD, FieldPtr);
        }
      }
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap,
                      /*ModuleLevelChanges=*/false, Returns);

    // Branch from EntryBB to cloned entry
    BasicBlock *ClonedEntry = nullptr;
    for (BasicBlock &BB : *NewF) {
      if (&BB != EntryBB) { ClonedEntry = &BB; break; }
    }
    if (!ClonedEntry) {
      NewF->eraseFromParent();
      continue;
    }
    EB.CreateBr(ClonedEntry);

    // Remove SpecArrayPartition/SpecInterface/SpecBitsMap that reference
    // the alloca (originally referenced the old partition arg).
    SmallVector<Instruction *, 16> Dead;
    for (BasicBlock &BB : *NewF) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N != "_ssdm_op_SpecArrayPartition" &&
            N != "_ssdm_op_SpecInterface" &&
            N != "_ssdm_op_SpecBitsMap") continue;
        if (CI->arg_size() < 1) continue;
        Value *V = CI->getArgOperand(0);
        while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
        for (auto &Tup : PartAllocas) {
          if (V == std::get<0>(Tup)) {
            Dead.push_back(CI);
            break;
          }
        }
      }
    }
    for (auto *I : Dead) I->eraseFromParent();

    // Replace old function with new
    std::string OldName = Old->getName().str();
    if (!Old->use_empty()) {
      Old->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, Old->getType()));
    }
    Old->eraseFromParent();
    NewF->setName(OldName);
    Changed = true;
  }

  // Drop the partition markers we consumed
  for (CallInst *CI : MarkersToDrop) {
    if (CI->getParent()) CI->eraseFromParent();
  }
  if (PartMarker->use_empty()) PartMarker->eraseFromParent();
  return Changed;
}

// __vxx_array_partition(ptr, type, factor, dim): emit SpecArrayPartition.
// type: 1=cyclic, 2=block, 3=complete.
// Factor must be >= 1 except for complete (kind=3) where it's ignored;
// HLS errors with "ssdm call instruction should not happen" if factor=0
// for cyclic/block.
bool injectArrayPartition(Module &M) {
  Function *F = M.getFunction("__vxx_array_partition");
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Value *Ptr = CI->getArgOperand(0);
    Argument *Arg = resolveMarkerArg(Ptr);
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *FactorC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    auto *DimC = dyn_cast<ConstantInt>(CI->getArgOperand(3));

    // Kernel-ARG partition: complete is physically disaggregated by
    // disaggCompletePartitionKernelSig before us → the marker here is redundant
    // (and its cloned copy on the .part_tmp bridge alloca would otherwise survive
    // as a bare _ssdm_op_SpecArrayPartition → HLS 200-70). Drop it.
    // (We keep the legacy SpecArrayPartition emit only for a hypothetical
    //  cyclic/block kernel-arg, which no current example exercises.)
    if (Arg) {
      if (KindC && FactorC && DimC) {
        uint64_t Kind = KindC->getZExtValue();
        uint64_t Factor = FactorC->getZExtValue();
        if (Factor != 0) {
          FunctionCallee SpecPart = hlsrs::vxx::getSsdmOp(M, "_ssdm_op_SpecArrayPartition");
          Value *Args[] = {
            Arg,
            ConstantInt::get(I32, DimC->getZExtValue()),
            ConstantInt::get(I32, Kind + 1),  // Rust 0/1/2 → HLS 1/2/3
            ConstantInt::get(I32, Factor),
            getOrCreateCStrGlobal(M, "")
          };
          B.CreateCall(SpecPart, Args);
        }
      }
      CI->eraseFromParent();
      Changed = true;
      continue;
    }

    // LOCAL-array partition: the marker ptr traces to an AllocaInst. Emit the
    // canonical op-bundle on the INTACT alloca:
    //   llvm.sideeffect() [ "xlx_array_partition"(ptr, i32 type, i64 factor,
    //                                              i64 dim, i1 false) ]
    // type: 0=cyclic, 1=block, 2=complete (= Rust kind directly); factor=0 for
    // complete; dim is 1-indexed. The HLS backend consumes this and physically
    // splits / scalarizes / muxes the array itself. NB:
    // emitting a bare _ssdm_op_SpecArrayPartition on an intact local array
    // instead is NOT consumed → survives → HLS 200-70; and
    // dropping it entirely
    // → the needed-partitioned local array stays BRAM → LatencyEstimator SIGSEGV.
    // The op-bundle is the only form that builds correctly.
    if (KindC && DimC) {
      Value *V = Ptr;
      while (true) {
        if (auto *BC = dyn_cast<BitCastOperator>(V)) { V = BC->getOperand(0); continue; }
        if (auto *AC = dyn_cast<AddrSpaceCastOperator>(V)) { V = AC->getOperand(0); continue; }
        if (auto *GEP = dyn_cast<GEPOperator>(V)) { V = GEP->getPointerOperand(); continue; }
        break;
      }
      if (auto *AI = dyn_cast<AllocaInst>(V)) {
        uint64_t Kind = KindC->getZExtValue();           // 0=cyclic,1=block,2=complete
        uint64_t Factor = FactorC ? FactorC->getZExtValue() : 0;
        if (Kind == 2) Factor = 0;                        // complete ignores factor
        IRBuilder<> EB(CI);
        Value *BundleArgs[] = {
          (Value *)AI,
          ConstantInt::get(I32, Kind),
          ConstantInt::get(I64, Factor),
          ConstantInt::get(I64, DimC->getZExtValue()),
          ConstantInt::get(I1, 0)                         // dynamic=false
        };
        OperandBundleDef OBD("xlx_array_partition", ArrayRef<Value *>(BundleArgs, 5));
        CallInst *SECall = EB.CreateCall(SEFn, None, ArrayRef<OperandBundleDef>{OBD});
        SECall->setOnlyAccessesInaccessibleMemory();
        SECall->setDoesNotThrow();
      }
    }
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}

bool injectArrayReshape(Module &M) {
  Function *F = M.getFunction("__vxx_array_reshape");
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    // Local-array reshape: marker ptr traces to an AllocaInst (not a kernel
    // Argument). Dropping it (old behaviour) leaves a wide local array that the
    // HLS scheduler can't model in a pipelined loop → SIGSEGV (ecc_flags). The
    // reshape is internal storage only (ports unaffected) — emit it so
    // csynth doesn't crash. Mirrors injectArrayPartition's alloca handling.
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *FactorC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    auto *DimC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
    // Local-alloca reshape (e.g. ecc_flags res1): DROP the marker. Emitting
    // SpecArrayReshape OR an xlx_array_reshape op-bundle OR physically packing a
    // *local* alloca makes HLS crash (200-70 / LatencyEstimator SIGSEGV).
    // Dropping it still matches C++ (the uram_ecc SpecResource bind is what
    // produces `ap_ecc_res1_U`). Only emit for a kernel-arg array, where
    // resolveMarkerArg returns a non-null Argument.
    if (Arg && KindC && FactorC && DimC) {
      FunctionCallee SpecRes = hlsrs::vxx::getSsdmOp(M, "_ssdm_op_SpecArrayReshape");
      Value *Args[] = {
        Arg,
        ConstantInt::get(Type::getInt32Ty(M.getContext()), DimC->getZExtValue()),
        ConstantInt::get(Type::getInt32Ty(M.getContext()), KindC->getZExtValue()),
        ConstantInt::get(Type::getInt32Ty(M.getContext()), FactorC->getZExtValue()),
        getOrCreateCStrGlobal(M, "")
      };
      B.CreateCall(SpecRes, Args);
    }
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}

bool injectArrayViewScope(Module &M) {
  bool A = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_array_view_begin");
  bool B = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_array_view_end");
  return A || B;
}

// Collapse memcpy-bridge allocas: when an alloca is only used as
// memcpy-dst (1 call) + memcpy-src (1 call) + lifetime/bitcast helpers,
// and both memcpys copy the same byte count == alloca size, replace
// `memcpy(dst, bridge, N) ; memcpy(bridge, src, N)` with
// `memcpy(dst, src, N)` and erase the bridge.
//
// This eliminates the Rust pattern (intermediate `let` binding for struct
// values) that causes HLS 214-211 OOB on Pkt-typed bridge allocas when
// the SROA pass on the bridge can't reconcile the i96 source memcpy with
// the per-field struct layout.
bool collapseMemcpyBridgeAllocas(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<AllocaInst *, 8> Candidates;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          Candidates.push_back(AI);
    for (AllocaInst *AI : Candidates) {
      // Find memcpy-dst (where this alloca is the dst, via i8* bitcast)
      // and memcpy-src (this alloca is the src). Skip if any other use
      // is not lifetime intrinsic / bitcast-for-memcpy.
      MemCpyInst *DstCpy = nullptr; // alloca is dst (filled)
      MemCpyInst *SrcCpy = nullptr; // alloca is src (drained)
      bool Bad = false;
      SmallVector<BitCastInst *, 4> BCs;
      auto checkBitcastOnly = [&](Value *V, bool *Found, MemCpyInst **Memcpy, bool ExpectDst) {
        for (User *U : V->users()) {
          if (auto *MC = dyn_cast<MemCpyInst>(U)) {
            // Check if our V is the dst or src arg of this memcpy.
            bool IsDst = (MC->getRawDest() == V) || (MC->getDest() == V);
            bool IsSrc = (MC->getRawSource() == V) || (MC->getSource() == V);
            if (IsDst && ExpectDst) {
              if (*Memcpy) { Bad = true; return; }
              *Memcpy = MC; *Found = true;
            } else if (IsSrc && !ExpectDst) {
              if (*Memcpy) { Bad = true; return; }
              *Memcpy = MC; *Found = true;
            }
          }
        }
      };
      for (User *U : AI->users()) {
        if (auto *MC = dyn_cast<MemCpyInst>(U)) {
          if (MC->getDest() == AI) {
            if (DstCpy) { Bad = true; break; }
            DstCpy = MC;
          } else if (MC->getSource() == AI) {
            if (SrcCpy) { Bad = true; break; }
            SrcCpy = MC;
          } else { Bad = true; break; }
        } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
          BCs.push_back(BC);
        } else if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          auto IID = II->getIntrinsicID();
          if (IID != Intrinsic::lifetime_start &&
              IID != Intrinsic::lifetime_end) {
            Bad = true; break;
          }
        } else {
          Bad = true; break;
        }
      }
      if (Bad) continue;
      // Check bitcast uses — each bitcast must only feed memcpy/lifetime.
      for (BitCastInst *BC : BCs) {
        for (User *BU : BC->users()) {
          if (auto *MC = dyn_cast<MemCpyInst>(BU)) {
            if (MC->getRawDest() == BC) {
              if (DstCpy && DstCpy != MC) { Bad = true; break; }
              if (!DstCpy) DstCpy = MC;
            } else if (MC->getRawSource() == BC) {
              if (SrcCpy && SrcCpy != MC) { Bad = true; break; }
              if (!SrcCpy) SrcCpy = MC;
            } else { Bad = true; break; }
          } else if (auto *II = dyn_cast<IntrinsicInst>(BU)) {
            auto IID = II->getIntrinsicID();
            if (IID != Intrinsic::lifetime_start &&
                IID != Intrinsic::lifetime_end) { Bad = true; break; }
          } else { Bad = true; break; }
        }
        if (Bad) break;
      }
      if (Bad || !DstCpy || !SrcCpy) continue;
      // Both memcpys must have same byte length (constant), and the size
      // must equal the alloca's allocated size.
      auto *DstLen = dyn_cast<ConstantInt>(DstCpy->getLength());
      auto *SrcLen = dyn_cast<ConstantInt>(SrcCpy->getLength());
      if (!DstLen || !SrcLen) continue;
      if (DstLen->getZExtValue() != SrcLen->getZExtValue()) continue;
      uint64_t AllocaSize = M.getDataLayout().getTypeAllocSize(
          AI->getAllocatedType());
      if (DstLen->getZExtValue() != AllocaSize) continue;
      // DstCpy must dominate SrcCpy (fill before drain). For simplicity
      // require them in the same basic block with DstCpy before SrcCpy.
      if (DstCpy->getParent() != SrcCpy->getParent()) continue;
      bool SeenDst = false, OrderOK = false;
      for (Instruction &I : *DstCpy->getParent()) {
        if (&I == DstCpy) SeenDst = true;
        else if (&I == SrcCpy && SeenDst) { OrderOK = true; break; }
      }
      if (!OrderOK) continue;
      // Rewrite SrcCpy's source operand to point at DstCpy's source.
      // Both memcpys take i8* args (Rust's normalised form). The new
      // memcpy: memcpy(SrcCpy.dst <- DstCpy.src, N).
      IRBuilder<> B(SrcCpy);
      Value *NewDst = SrcCpy->getRawDest();
      Value *NewSrc = DstCpy->getRawSource();
      // Align min of the two.
      unsigned DstAlign = std::min((unsigned)SrcCpy->getDestAlignment(),
                                    (unsigned)DstCpy->getSourceAlignment());
      unsigned SrcAlign = std::min((unsigned)SrcCpy->getSourceAlignment(),
                                    (unsigned)DstCpy->getSourceAlignment());
      B.CreateMemCpy(NewDst, MaybeAlign(DstAlign), NewSrc, MaybeAlign(SrcAlign),
                     SrcCpy->getLength(), SrcCpy->isVolatile());
      // Erase the two memcpys + bitcasts + lifetime intrinsics + alloca.
      SmallVector<Instruction *, 8> ToErase;
      ToErase.push_back(SrcCpy);
      ToErase.push_back(DstCpy);
      for (BitCastInst *BC : BCs) {
        for (User *BU : BC->users())
          if (auto *II = dyn_cast<IntrinsicInst>(BU))
            ToErase.push_back(II);
        ToErase.push_back(BC);
      }
      for (User *U : AI->users())
        if (auto *II = dyn_cast<IntrinsicInst>(U))
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end)
            ToErase.push_back(II);
      // Dedupe + reverse-order erase
      SmallPtrSet<Instruction *, 8> Seen;
      for (Instruction *I : ToErase) {
        if (!Seen.insert(I).second) continue;
        I->dropAllReferences();
      }
      Seen.clear();
      for (Instruction *I : ToErase) {
        if (!Seen.insert(I).second) continue;
        I->eraseFromParent();
      }
      AI->eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: collapsed memcpy-bridge alloca(s)\n";
  return Changed;
}

// Replace `store iN V, %iN_alloca ; memcpy(%pkt_alloca <- %iN_alloca, N)`
// with field-wise stores into %pkt_alloca, extracting each field from V
// via lshr + trunc. Removes the i96-bridge memcpy that HLS 214-211 flags
// as field-0 OOB on the destination Pkt alloca.
//
// Only fires when:
// 1. The memcpy src is a (bitcast of) iN alloca
// 2. The iN alloca has exactly 1 store (the i96 value) + lifetime intrinsics
// 3. The memcpy dst is a (bitcast of) struct alloca
// 4. Memcpy length == sizeof(struct)
bool inlineIntMemcpyAsFieldStores(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<MemCpyInst *, 8> Candidates;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *MC = dyn_cast<MemCpyInst>(&I))
          Candidates.push_back(MC);
    for (MemCpyInst *MC : Candidates) {
      auto *Len = dyn_cast<ConstantInt>(MC->getLength());
      if (!Len) continue;
      uint64_t NBytes = Len->getZExtValue();
      // Trace dst → struct alloca
      Value *Dst = MC->getRawDest();
      while (auto *BC = dyn_cast<BitCastInst>(Dst)) Dst = BC->getOperand(0);
      auto *PktAI = dyn_cast<AllocaInst>(Dst);
      if (!PktAI) continue;
      auto *PktST = dyn_cast<StructType>(PktAI->getAllocatedType());
      if (!PktST) continue;
      if (DL.getTypeAllocSize(PktST) != NBytes) continue;
      // Trace src → iN alloca with single store iN
      Value *Src = MC->getRawSource();
      while (auto *BC = dyn_cast<BitCastInst>(Src)) Src = BC->getOperand(0);
      auto *IAI = dyn_cast<AllocaInst>(Src);
      if (!IAI) continue;
      auto *IntTy = dyn_cast<IntegerType>(IAI->getAllocatedType());
      if (!IntTy) continue;
      if (IntTy->getBitWidth() != NBytes * 8) continue;
      // Find the single store iN V, %IAI in the same basic block, before MC.
      StoreInst *TheStore = nullptr;
      for (User *U : IAI->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() != IAI) { TheStore = nullptr; break; }
          if (TheStore) { TheStore = nullptr; break; }
          TheStore = SI;
        } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
          // Bitcast → only memcpy/lifetime
          for (User *BU : BC->users()) {
            if (isa<MemCpyInst>(BU) || isa<IntrinsicInst>(BU)) continue;
            TheStore = nullptr; goto next_cand;
          }
        } else if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          auto IID = II->getIntrinsicID();
          if (IID != Intrinsic::lifetime_start &&
              IID != Intrinsic::lifetime_end) { TheStore = nullptr; goto next_cand; }
        } else {
          TheStore = nullptr; goto next_cand;
        }
      }
      next_cand:;
      if (!TheStore) continue;
      if (TheStore->getValueOperand()->getType() != IntTy) continue;
      // Build field-wise stores at the memcpy site
      IRBuilder<> B(MC);
      Value *IntVal = TheStore->getValueOperand();
      bool AnyEmit = false;
      for (unsigned i = 0; i < PktST->getNumElements(); ++i) {
        Type *FT = PktST->getElementType(i);
        uint64_t Off = DL.getStructLayout(PktST)->getElementOffsetInBits(i);
        // Skip ZSA padding [0 x ?] and trailing [N x i8] alignment pads.
        if (auto *AT = dyn_cast<ArrayType>(FT)) {
          if (AT->getNumElements() == 0) continue;
          if (AT->getElementType()->isIntegerTy(8)) continue;
        }
        if (!FT->isIntegerTy() && !FT->isPointerTy() && !FT->isFloatingPointTy())
          continue;
        uint64_t FW = DL.getTypeSizeInBits(FT);
        if (FW == 0) continue;
        // shift = lshr IntVal, Off; trunc to FT
        Value *Shifted = (Off == 0) ? IntVal
                                     : B.CreateLShr(IntVal,
                                                     ConstantInt::get(IntTy, Off));
        Type *NarrowTy = IntegerType::get(M.getContext(), (unsigned)FW);
        Value *Narrow = B.CreateTrunc(Shifted, NarrowTy);
        Value *FieldVal = Narrow;
        if (FT->isPointerTy()) {
          FieldVal = B.CreateIntToPtr(Narrow, FT);
        } else if (FT->isFloatingPointTy()) {
          FieldVal = B.CreateBitCast(Narrow, FT);
        } else if (FT != NarrowTy) {
          // Sign/zero extend, but for integer fields the trunc to FW should match
          FieldVal = B.CreateBitCast(Narrow, FT);
        }
        Value *GEP = B.CreateConstGEP2_32(PktST, PktAI, 0, i);
        B.CreateStore(FieldVal, GEP);
        AnyEmit = true;
      }
      if (!AnyEmit) continue;
      // Erase memcpy + store iN + iN alloca + its lifetime intrinsics
      MC->eraseFromParent();
      TheStore->eraseFromParent();
      SmallVector<Instruction *, 4> ToErase;
      for (User *U : IAI->users()) {
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          for (User *BU : BC->users())
            if (auto *II = dyn_cast<IntrinsicInst>(BU))
              ToErase.push_back(II);
          ToErase.push_back(BC);
        } else if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          ToErase.push_back(II);
        }
      }
      SmallPtrSet<Instruction *, 4> Seen;
      for (Instruction *I : ToErase) {
        if (!Seen.insert(I).second) continue;
        I->dropAllReferences();
      }
      Seen.clear();
      for (Instruction *I : ToErase) {
        if (!Seen.insert(I).second) continue;
        I->eraseFromParent();
      }
      IAI->eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: inlined int-memcpy → field-wise store(s)\n";
  return Changed;
}

} } // namespace hlsrs::vxx
