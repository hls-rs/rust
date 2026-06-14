//===----------------------------------------------------------------------===//
//
// maxi.cpp — m_axi (master AXI) interface / cache / burst passes.
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

// generic private-string-constant helper. Moved here from VXXPrep.cpp's
// outer anonymous namespace (where it could not be re-namespaced cleanly).
// Used by both the moved domain passes and many passes still in VXXPrep.cpp;
// declared in vxx_passes.h.
GlobalVariable *getOrCreateCStrGlobal(Module &M, StringRef Text) {
  LLVMContext &Ctx = M.getContext();
  Constant *Init = ConstantDataArray::getString(Ctx, Text, /*AddNull=*/true);
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasPrivateLinkage() || !G.isConstant()) continue;
    if (G.getValueType() != Init->getType()) continue;
    if (G.getInitializer() == Init) return &G;
  }
  auto *GV = new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, Init, ".str");
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

// Resolve a marker's first arg (a *const T pointer) back to the function
// argument it ultimately points to. Strips bitcasts / addrspacecasts.
Argument *resolveMarkerArg(Value *V) {
  while (true) {
    if (auto *BC = dyn_cast<BitCastInst>(V)) {
      V = BC->getOperand(0);
      continue;
    }
    if (auto *AS = dyn_cast<AddrSpaceCastInst>(V)) {
      V = AS->getOperand(0);
      continue;
    }
    if (auto *EV = dyn_cast<ExtractValueInst>(V)) {
      // For BurstMaxi<T> by-value kernel args, `m_axi_named` calls go
      // through `extractvalue %class.hls::burst_maxi<T> %argN, 0` to get
      // at the inner ptr. Walk through the extract back to the aggregate
      // operand so the m_axi op-bundle attaches to the kernel arg itself.
      V = EV->getAggregateOperand();
      continue;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      if (CE->getOpcode() == Instruction::BitCast ||
          CE->getOpcode() == Instruction::AddrSpaceCast) {
        V = CE->getOperand(0);
        continue;
      }
    }
    // For by-value scalar kernel args, Rust spills the arg to a local
    // alloca on entry and `&arg` resolves to that alloca pointer. Detect
    // the canonical "alloca + single store-from-Argument" pattern and
    // skip back to the Argument.
    if (auto *AI = dyn_cast<AllocaInst>(V)) {
      Argument *FromArg = nullptr;
      bool Ok = true;
      for (User *U : AI->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() != AI) continue;
          if (auto *A = dyn_cast<Argument>(SI->getValueOperand())) {
            if (FromArg && FromArg != A) { Ok = false; break; }
            FromArg = A;
            continue;
          }
          // Stored value isn't an arg: not the pattern we want.
          Ok = false; break;
        }
        // Loads / lifetime / debug uses are fine. Anything else means
        // the alloca is mutated beyond the entry spill — bail.
        if (isa<LoadInst>(U) || isa<BitCastInst>(U) ||
            isa<DbgInfoIntrinsic>(U) || isa<IntrinsicInst>(U))
          continue;
        Ok = false; break;
      }
      if (Ok && FromArg) { V = FromArg; continue; }
    }
    break;
  }
  return dyn_cast<Argument>(V);
}

// For each `__vxx_m_axi` marker, emit
// `call void @llvm.sideeffect() [ "xlx_m_axi"(ptr, ...) ]` at the marker
// site. This is the lowering for `#pragma HLS INTERFACE m_axi`.
// `llvm.sideeffect` is a standard LLVM intrinsic; the `"xlx_m_axi"` operand
// bundle name is the interop point with the HLS backend.
//
// The marker is unified: `__vxx_m_axi(ptr, depth, name_ptr, name_len,
// bundle_idx, widen, direct)`. Every optional `#pragma HLS INTERFACE m_axi`
// clause is one arg with a sentinel for "unset"; all variants fold into the
// same op-bundle below.
//
// Bundle arg layout:
//   ptr, "" (signal_name), i64 depth, "" (?), "" (offset_mode), i64*6 (-1=default), "" (?)
bool injectMAxiSideeffect(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  StringMap<unsigned> NameToIdx; // bundle string -> assigned idx
  unsigned NextNamedIdx = 0;

  struct Spec {
    Argument *Arg;
    std::string IfaceVal;   // m_axi.<idx> or m_axi.<name>
    std::string BundleName; // user-visible bundle string for xlx_m_axi slot 1
    uint64_t Depth = 0;
    int64_t MaxWidenBitwidth = -1; // -1 = auto/default
    bool OffsetDirect = false; // true → offset=direct (no s_axi_control)
    // Optional performance clauses (xlx_m_axi slots 5-9); -1 = default.
    int64_t NumReadOutstanding = -1;
    int64_t NumWriteOutstanding = -1;
    int64_t MaxReadBurst = -1;
    int64_t MaxWriteBurst = -1;
    int64_t Latency = -1;
  };
  SmallVector<Spec, 8> Specs;
  SmallPtrSet<Function *, 4> KernelsWithSAxi;

  SmallVector<CallInst *, 16> DeadCalls;
  // Single unified marker `__vxx_m_axi`. Read its 7 args and build the same
  // Spec the old per-variant markers built, so the op-bundle below is
  // byte-identical for every kind of m_axi interface.
  if (Function *F = M.getFunction("__vxx_m_axi")) {
    SmallVector<CallInst *, 4> Calls;
    for (User *U : F->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        Calls.push_back(CI);
    // Marker->users() is LIFO; reverse to source order.
    std::reverse(Calls.begin(), Calls.end());
    for (CallInst *CI : Calls) {
      DeadCalls.push_back(CI);
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg)
        continue;
      Spec S;
      S.Arg = Arg;
      // arg1: depth. u32::MAX (= 0xFFFFFFFF) is the Rust API's "omitted
      // depth" sentinel — the builder has no depth clause because the element
      // count is carried by the argument's TYPE. Derive here the same
      // explicit N the C++ `depth=N` clause spells out: the flattened element
      // count of the pointee array ([H x [W x T]]* → H*W), a decayed pointer's
      // `fpga.decayed.dim.hint` (already the flat scalar count), or 1 for a
      // scalar pointee (&mut i32 / BitInt wrappers — C++ uses depth=1 there).
      // Emitting the explicit N (not the backend's `-1` auto sentinel) keeps
      // the xlx_m_axi bundle byte-identical to the C++ pragma's; `-1` was
      // tried and mis-classifies ports (maxi_cache 214-399) and SIGSEGVs
      // csynth on scalar BitInt ports (ram_uram).
      auto *DC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      uint64_t Depth = DC ? DC->getZExtValue() : 0;
      if (Depth == 0xFFFFFFFFull) {
        Depth = (uint64_t)-1; // fall back to auto if the type is opaque
        if (auto *PT = dyn_cast<PointerType>(Arg->getType())) {
          Type *T = PT->getElementType();
          uint64_t Elems = 1;
          while (auto *AT = dyn_cast<ArrayType>(T)) {
            Elems *= AT->getNumElements();
            T = AT->getElementType();
          }
          AttributeSet AA =
              Arg->getParent()->getAttributes().getParamAttributes(
                  Arg->getArgNo());
          if (AA.hasAttribute("fpga.decayed.dim.hint")) {
            uint64_t HintN = 0;
            AA.getAttribute("fpga.decayed.dim.hint")
                .getValueAsString()
                .getAsInteger(10, HintN);
            if (HintN)
              Depth = HintN;
          } else if (T->isSized()) {
            Depth = Elems;
          }
        }
      }
      S.Depth = Depth;
      // arg5: widen. u32::MAX → -1 (native element width / auto).
      if (auto *WC = dyn_cast<ConstantInt>(CI->getArgOperand(5))) {
        uint64_t W = WC->getZExtValue();
        S.MaxWidenBitwidth = (W == 0xFFFFFFFFull) ? -1 : (int64_t)W;
      }
      // arg6: direct (0/1) → offset=direct (no s_axi_control).
      if (auto *OC = dyn_cast<ConstantInt>(CI->getArgOperand(6)))
        S.OffsetDirect = OC->getZExtValue() != 0;
      // args 7-11 (optional; u32::MAX = unset): num_read_outstanding,
      // num_write_outstanding, max_read_burst_length, max_write_burst_length,
      // latency — the pragma's remaining performance clauses.
      auto ReadOpt = [&](unsigned Idx, int64_t &Out) {
        if (CI->arg_size() > Idx)
          if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(Idx))) {
            uint64_t V = C->getZExtValue();
            if (V != 0xFFFFFFFFull)
              Out = (int64_t)V;
          }
      };
      ReadOpt(7, S.NumReadOutstanding);
      ReadOpt(8, S.NumWriteOutstanding);
      ReadOpt(9, S.MaxReadBurst);
      ReadOpt(10, S.MaxWriteBurst);
      ReadOpt(11, S.Latency);
      // Bundle string: an explicit name (arg2 ptr + arg3 len) wins; else
      // gmem<idx> (arg4 != u32::MAX); else the default adaptor index.
      uint64_t NameLen = 0;
      if (auto *LC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
        NameLen = LC->getZExtValue();
      uint64_t BundleIdx = 0xFFFFFFFFull;
      if (auto *BC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        BundleIdx = BC->getZExtValue();
      std::string Bundle;
      std::string UserBundleName;
      if (NameLen > 0) {
        Bundle = hlsrs::vxx::extractByteSlice(CI->getArgOperand(2),
                                              CI->getArgOperand(3));
        UserBundleName = Bundle;
      } else if (BundleIdx != 0xFFFFFFFFull) {
        Bundle = std::to_string(BundleIdx);
        UserBundleName = std::string("gmem") + Bundle;
      }
      if (Bundle.empty()) {
        Bundle = "0"; // default adaptor index
        // UserBundleName stays empty → xlx_m_axi emits empty bundle slot,
        // matching the default-bundle behaviour ("gmem").
      }
      // Lookup or assign a deterministic int idx for the bundle string so
      // multiple ports sharing the same bundle land on the same adaptor.
      auto It = NameToIdx.find(Bundle);
      unsigned BIdx;
      if (It == NameToIdx.end()) {
        BIdx = NextNamedIdx++;
        NameToIdx[Bundle] = BIdx;
      } else {
        BIdx = It->second;
      }
      // OSS HLS clang's `SynthesisAttr` only appends `.<offset_mode>`
      // when offset is explicit and != Default. For our default markers
      // we leave it implicit, matching `m_axi.<idx>` exactly.
      S.IfaceVal = std::string("m_axi.") + std::to_string(BIdx);
      S.BundleName = UserBundleName;
      Specs.push_back(S);
      KernelsWithSAxi.insert(Arg->getParent());
    }
  }
  if (Specs.empty())
    return false;

  // Emit `llvm.sideeffect [ "xlx_m_axi"(ptr, ...) ]` at the entry of each
  // kernel that has m_axi ports. The bundle layout is the
  // `#pragma HLS INTERFACE m_axi` lowering.
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *I8 = IntegerType::get(Ctx, 8);
  Type *I64 = IntegerType::get(Ctx, 64);
  Constant *EmptyArr = ConstantAggregateZero::get(ArrayType::get(I8, 0));
  Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  // Group specs by kernel so we emit at the entry of each.
  StringMap<SmallVector<Spec *, 4>> ByKernel;
  for (Spec &S : Specs)
    ByKernel[S.Arg->getParent()->getName()].push_back(&S);
  for (auto &KV : ByKernel) {
    Function *F = M.getFunction(KV.first());
    if (!F)
      continue;
    IRBuilder<> B(&F->getEntryBlock(),
                  F->getEntryBlock().getFirstInsertionPt());
    for (Spec *S : KV.second) {
      Value *Ptr = S->Arg;
      // For BurstMaxi/burst_maxi struct-by-value args (e.g. after
      // renameAndStripBurstMaxi which makes the arg type `{ T* }`), the
      // operand bundle wants the inner pointer, not the wrapper struct.
      // Passing a by-value struct as an operand bundle value confuses
      // later LLVM passes (segfault in rustc downstream LLVM opts).
      // Extract field 0 at function entry to get the i32*/T* and use that.
      if (auto *ST = dyn_cast<StructType>(Ptr->getType())) {
        if (ST->getNumElements() >= 1 &&
            ST->getElementType(0)->isPointerTy()) {
          Ptr = B.CreateExtractValue(S->Arg, {0}, S->Arg->getName() + ".bm.ptr");
        }
      }
      // xlx_m_axi: (ptr, "<bundle>", i64 depth, "", "", i64*6, "")
      // 12 args total. -1 sentinels for unset numeric fields.
      Value *MinusOne = ConstantInt::get(I64, (uint64_t)-1, /*isSigned=*/true);
      // 2nd xlx_m_axi arg: bundle-name byte array (or empty array for default).
      // The bundle expects `[N x i8] c"<name>"` directly as a value-typed
      // arg (not a pointer to a global).
      Value *BundleNameC = EmptyArr;
      if (!S->BundleName.empty())
        BundleNameC = ConstantDataArray::getString(
            Ctx, S->BundleName, /*AddNull=*/false);
      // xlx_m_axi arg layout (`EmitMAxiInterfaceIntrinsic`
      // at HLS/llvm/clang/lib/CodeGen/CGXlxAttr.cpp:1741):
      //   0: ptr, 1: bundle name, 2: depth, 3: offset_mode, 4: signal_name,
      //   5: num_read_outstanding, 6: num_write_outstanding,
      //   7: max_read_burst, 8: max_write_burst,
      //   9: latency, 10: max_widen_bitwidth
      // The 12th slot (`EmptyArr`) is internal padding kept for
      // backward-compat (e.g. on lmem_2rw).
      Value *MaxWidenC =
          (S->MaxWidenBitwidth < 0) ? MinusOne
                                    : ConstantInt::get(I64, S->MaxWidenBitwidth);
      // Slot 3 = offset_mode. Empty for default ("slave"), or "direct"
      // when m_axi_direct marker was used.
      Value *OffsetModeC = EmptyArr;
      if (S->OffsetDirect)
        OffsetModeC = ConstantDataArray::getString(Ctx, "direct", /*AddNull=*/false);
      Value *Args[] = {
          Ptr,
          BundleNameC,
          ConstantInt::get(I64, S->Depth),
          OffsetModeC,
          EmptyArr,
          (S->NumReadOutstanding < 0) ? MinusOne
              : ConstantInt::get(I64, S->NumReadOutstanding),
          (S->NumWriteOutstanding < 0) ? MinusOne
              : ConstantInt::get(I64, S->NumWriteOutstanding),
          (S->MaxReadBurst < 0) ? MinusOne
              : ConstantInt::get(I64, S->MaxReadBurst),
          (S->MaxWriteBurst < 0) ? MinusOne
              : ConstantInt::get(I64, S->MaxWriteBurst),
          (S->Latency < 0) ? MinusOne : ConstantInt::get(I64, S->Latency),
          MaxWidenC,
          EmptyArr,
      };
      OperandBundleDef OB("xlx_m_axi",
                           SmallVector<Value *, 12>(std::begin(Args), std::end(Args)));
      CallInst *NewCI = B.CreateCall(FunctionCallee(SideEffect), {}, {OB});
      // Stamp `xlx.port.bitwidth` + `xlx.source="user"` call-site attrs.
      // The HLS backend's early m_axi rewrite uses these to identify the bundle's
      // total port size in bits (= depth_bytes * 8) and that the pragma
      // came from user code (vs synthesised). Without them the rewriter
      // can't size the gmem bundle and silently drops the body.
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::InaccessibleMemOnly);
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      // `xlx.port.bitwidth` is the total m_axi gmem bundle size in bits
      // (= depth_in_elements × elem_size_bytes × 8). The Rust marker passes
      // `depth` as element count (matching C++ `#pragma HLS INTERFACE m_axi
      // depth=N`), so multiply by the pointee element size to get bytes,
      // then by 8 to get bits.
      uint64_t PortBits = 0;
      auto *PT = dyn_cast<PointerType>(S->Arg->getType());
      // Two element sizes: SCALAR (innermost) for the explicit-depth branch,
      // and the immediate POINTEE for the auto-depth branch.
      // - explicit depth: the Rust marker passes `depth` as the FLAT scalar
      //   count (e.g. W*H for a 2-D image), so a row-array pointee `[W x i8]`
      //   would 1000× over-count. The opaque-ptr port size = depth × scalar
      //   × 8 (stencil src/dst = "240000" = 30000×1×8, NOT "240000000"). An
      //   over-sized xlx.port.bitwidth makes the backend model the m_axi as an
      //   array-of-rows (i8P1A per-element, no burst → II=1 unmet →
      //   getEstimatedSchedLatency SIGSEGV) instead of a flat byte burst (p1i8).
      // - auto depth (-1): size is inferred from the pointee TYPE itself
      //   (e.g. coeff `[256 x i8]*` → 256 bytes → 2048 bits), so the full
      //   pointee size is correct there; unwrapping to scalar would give 8.
      uint64_t ScalarBytes = 0, PointeeBytes = 0;
      if (PT) {
        Type *Pointee = PT->getElementType();
        if (Pointee->isSized())
          PointeeBytes = M.getDataLayout().getTypeAllocSize(Pointee);
        Type *Scalar = Pointee;
        while (auto *AT = dyn_cast<ArrayType>(Scalar))
          Scalar = AT->getElementType();
        if (Scalar->isSized())
          ScalarBytes = M.getDataLayout().getTypeAllocSize(Scalar);
      }
      if (S->Depth > 0 && S->Depth != (uint64_t)-1) {
        // Explicit depth → total bytes = depth × scalar_size. Fall back to
        // (depth × 8) only when we can't determine the element size
        // (preserves old behavior for opaque pointer pointees).
        PortBits = ScalarBytes > 0 ? (S->Depth * ScalarBytes * 8)
                                   : (S->Depth * 8);
      } else if (PT) {
        Type *Pointee = PT->getElementType();
        if (Pointee->isSized()) {
          uint64_t Bytes = PointeeBytes;
          AttributeSet ArgAttrs =
              S->Arg->getParent()->getAttributes().getParamAttributes(
                  S->Arg->getArgNo());
          if (ArgAttrs.hasAttribute("fpga.decayed.dim.hint")) {
            Attribute Hint = ArgAttrs.getAttribute("fpga.decayed.dim.hint");
            uint64_t HintN = 0;
            Hint.getValueAsString().getAsInteger(10, HintN);
            // dim.hint is the element COUNT of the decayed dimension, so the
            // region size is scalar_size × count — NOT pointee_size × count.
            // For coeff `[256 x i8]*` (1-D, pointee = the full [256 x i8]):
            // ScalarBytes(1) × 256 = 256 B → 2048 bits (the opaque-ptr coeff
            // port = "2048"). The old `PointeeBytes × HintN` double-
            // counted (256 × 256 × 8 = 524288, 256× too large) which poisoned
            // the SHARED gmem1 bundle port width → the backend could not size the
            // burst → getEstimatedSchedLatency null-deref (stencil_2d crash).
            if (ScalarBytes > 0)
              Bytes = ScalarBytes * HintN;
          }
          PortBits = Bytes * 8;
        }
      }
      NewCI->addAttribute(AttributeList::FunctionIndex,
          Attribute::get(Ctx, "xlx.port.bitwidth", std::to_string(PortBits)));
      NewCI->addAttribute(AttributeList::FunctionIndex,
          Attribute::get(Ctx, "xlx.source", "user"));
      Changed = true;

      // Stamp `fpga.decayed.dim.hint = "<outer_dim>"` on the m_axi arg.
      // This is emitted ONLY for 2-D row-pointer params
      // (e.g. `char arr[R][W]` decays to `[W x i8]*` with hint=R). the cosim TB's
      // `_ir` wrapper uses it to size the input buffer correctly —
      // without it, cosim TB mistakes `depth` for "number of pointee-typed
      // elements" and over-allocates (e.g. `[1000 x i8]*` with depth
      // 30000 → `[30000 x [1000 x i8]]` = 30 MB instead of 30 rows).
      //
      // For scalar pointees (`i32*`, `i64*`), DON'T emit the hint:
      // cosim TB correctly infers nbytes = depth * sizeof(elem) from
      // depth alone, and emitting hint=depth/sizeof(elem) would make
      // cosim TB compute nbytes = (depth/sizeof) * sizeof = depth bytes
      // (4× too small for i32, 8× for i64).
      // Only emit for sized array pointees `[W x T]`.
      // Skip when depth is the `-1` sentinel ("default, auto-infer") —
      // `dim.hint` is not emitted when `depth=` is omitted.
      if (S->Depth > 0 && S->Depth != (uint64_t)-1) {
        if (auto *PtrTy = dyn_cast<PointerType>(S->Arg->getType())) {
          Type *Pointee = PtrTy->getElementType();
          if (Pointee->isSized() && isa<ArrayType>(Pointee)) {
            uint64_t InnerBytes = M.getDataLayout().getTypeAllocSize(Pointee);
            if (InnerBytes > 0 && (uint64_t)S->Depth >= InnerBytes
                && ((uint64_t)S->Depth % InnerBytes) == 0) {
              uint64_t OuterDim = (uint64_t)S->Depth / InnerBytes;
              S->Arg->addAttr(Attribute::get(
                  Ctx, "fpga.decayed.dim.hint", std::to_string(OuterDim)));
            }
          }
        }
      }
    }
  }

  // NOTE: tried emitting `_ssdm_op_SpecInterface(ptr, "m_axi", ..., depth, ...)`
  // here to feed the cosim TB generator the `port_props.nbytes = depth × elem_size`
  // it needs. But the expected shape uses `i32 addrspace(1)*` for
  // the first arg (a per-port addrspace-1 adaptor pointer, not the
  // user's addrspace-0 kernel arg). Passing the Rust addrspace-0 arg
  // crashes csynth with `[HLS 200-70] Failed building synthesis data
  // model`. Leaving the m_axi cluster on the xlx_m_axi sideeffect path
  // for now; full fix requires synthesising the addrspace-1 adaptor
  // pointer ourselves.
  // Stamp `fpga.top.func` and `fpga.demangled.name` on the kernel
  // function so the HLS backend recognises it as the top.
  // `fpga.demangled.name` is what `set_top` / `set_directive_*` from
  // TCL key off when looking up functions by user-visible name.
  for (Function *F : KernelsWithSAxi) {
    if (!F->hasFnAttribute("fpga.top.func"))
      F->addFnAttr("fpga.top.func", F->getName());
    if (!F->hasFnAttribute("fpga.demangled.name"))
      F->addFnAttr("fpga.demangled.name", F->getName());
  }
  // Erase the marker calls so the marker function definitions can be
  // dropped at the end of the pass (otherwise the HLS backend sees undefined
  // references to `__vxx_m_axi*`).
  for (CallInst *CI : DeadCalls)
    CI->eraseFromParent();
  return Changed;
}



// For each m_axi arg of each kernel, emit
// `_ssdm_op_SpecInterface` calls. the cosim TB generator reads these to size its
// helper functions (`arraycpy_hls.p0aN i32` for bulk copy, with N from
// the depth field). Without these, cosim TB only emits 1-element
// `onebyonecpy_hls.p0i32` helpers → cosim reads only 1 element per
// port → kernel sees 0 for the rest → wrong outputs / FAIL.
//
// The SpecInterface calls reference
// a fresh null `i32 addrspace(1)* null` for the gmem adaptor. Passing the user's
// addrspace(0) arg as first arg crashes csynth; null addrspace(1)
// avoids that. The cosim TB generator still gets the depth (args 6+7 carry
// it), so we get bulk copy.
//
// Reference shape:
//   @_ssdm_op_SpecInterface(
//     i32 addrspace(1)* %gmem,        ; 0: adaptor (we use null)
//     [6 x i8]* @"m_axi",             ; 1: interface type
//     i32 0,                          ; 2: mode (0=master)
//     i32 0,                          ; 3
//     [1 x i8]* @"",                  ; 4: signal name (empty)
//     i32 0,                          ; 5
// i32 256, ; 6: depth ★ critical for cosim TB
//     [5 x i8]* @"gmem",              ; 7: bundle name
//     [6 x i8]* @"slave",             ; 8: offset mode
//     [1 x i8]* @"",                  ; 9
//     i32 16, i32 16,                 ; 10,11: num_read/write_outstanding
//     i32 16, i32 16,                 ; 12,13: max_read/write_burst
//     [1 x i8]* @"",                  ; 14
//     [1 x i8]* @"",                  ; 15
//     i32 -1,                         ; 16: latency
//     i32 0,                          ; 17: max_widen_bitwidth
//     i32 0,                          ; 18
//     i32 4)                          ; 19: alignment
// Lower `__vxx_maxi_cache(ptr, lines, depth)` markers (barista_hls::m_axi_cache)
// to the `#pragma HLS cache` form: an op-bundle on
// `llvm.sideeffect`, `[ "xlx_cache"(ptr, i64 lines, i64 depth, i64 1, i64 0,
// i64 0, i64 0) ]` (7 args, NO trailing string). The HLS backend converts THIS
// to its internal `_ssdm_op_SpecMAXICache`. Pre-emitting `_ssdm_op_SpecMAXICache`
// directly (the previous form) made the backend downgrade the i32 m_axi channel
// to an i8 generic pointer (`%.addr`), so the cache config no longer matched the
// i32 data port → RTGEN 206-100 SIGSEGV at interface bundling. The op-bundle
// form keeps the port as the i32 `aximm` channel.
// Runs after injectMAxiSpecInterface and before eraseUnimplementedMarkers.
bool injectMAxiCache(Module &M) {
  Function *Mk = M.getFunction("__vxx_maxi_cache");
  if (!Mk) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  SmallVector<CallInst *, 4> ToErase;
  bool Changed = false;
  for (User *U : Mk->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    ToErase.push_back(CI);
    if (CI->arg_size() < 3) continue;
    Argument *A = resolveMarkerArg(CI->getArgOperand(0));
    auto *LinesC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *DepthC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!A || !LinesC || !DepthC) continue;
    Value *Lines = ConstantInt::get(I64, LinesC->getZExtValue());
    Value *Depth = ConstantInt::get(I64, DepthC->getZExtValue());
    Value *One = ConstantInt::get(I64, 1);
    Value *Zero = ConstantInt::get(I64, 0);
    IRBuilder<> B(CI);
    // xlx_cache op-bundle: (ptr, lines, depth, 1, 0, 0, 0).
    OperandBundleDef OB("xlx_cache",
                        SmallVector<Value *, 7>{(Value *)A, Lines, Depth, One,
                                                Zero, Zero, Zero});
    CallInst *NewCI = B.CreateCall(FunctionCallee(SideEffect), {}, {OB});
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    NewCI->addAttribute(AttributeList::FunctionIndex,
                        Attribute::InaccessibleMemOnly);
    Changed = true;
  }
  for (auto *CI : ToErase) CI->eraseFromParent();
  if (Mk->use_empty()) Mk->eraseFromParent();
  return Changed;
}



// Lower the token-chain `cache()` builder to the same `xlx_cache` op-bundle as
// injectMAxiCache above. The builder emits one marker per clause, threaded by
// an SSA token: `__vxx_cache_begin()` -> `__vxx_cache_port(tok, ptr)` ->
// `__vxx_cache_lines(tok, L)` -> `__vxx_cache_depth(tok, D)` (any clause order).
// Follow the token def-use chain from each begin to collect (port, lines,
// depth), then emit the bundle. The port pointer arrives DIRECTLY at its clause
// marker (SROA-safe), so no Drop-builder field can degenerate it.
// Runs alongside injectMAxiCache (both marker forms accepted during migration).
bool injectCacheChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_cache_begin");
  if (!Begin) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  Function *PortMk = M.getFunction("__vxx_cache_port");
  Function *LinesMk = M.getFunction("__vxx_cache_lines");
  Function *DepthMk = M.getFunction("__vxx_cache_depth");
  SmallVector<CallInst *, 8> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI) continue;
    ToErase.push_back(BeginCI);
    Value *Tok = BeginCI;
    Argument *Port = nullptr;
    uint64_t Lines = 0, Depth = 0;
    bool HaveLines = false, HaveDepth = false;
    // Walk the linear token chain: each clause marker takes the running token
    // as arg 0 and returns it.
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == PortMk || Callee == LinesMk || Callee == DepthMk) {
          Next = CI;
          break;
        }
      }
      if (!Next) break;
      Function *Callee = Next->getCalledFunction();
      if (Callee == PortMk && Next->arg_size() >= 2) {
        Port = resolveMarkerArg(Next->getArgOperand(1));
      } else if (Callee == LinesMk && Next->arg_size() >= 2) {
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1))) {
          Lines = C->getZExtValue();
          HaveLines = true;
        }
      } else if (Callee == DepthMk && Next->arg_size() >= 2) {
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1))) {
          Depth = C->getZExtValue();
          HaveDepth = true;
        }
      }
      ToErase.push_back(Next);
      Tok = Next;
    }
    if (!Port || !HaveLines || !HaveDepth) continue;
    IRBuilder<> B(BeginCI);
    OperandBundleDef OB("xlx_cache",
                        SmallVector<Value *, 7>{
                            (Value *)Port, ConstantInt::get(I64, Lines),
                            ConstantInt::get(I64, Depth), ConstantInt::get(I64, 1),
                            ConstantInt::get(I64, 0), ConstantInt::get(I64, 0),
                            ConstantInt::get(I64, 0)});
    CallInst *NewCI = B.CreateCall(FunctionCallee(SideEffect), {}, {OB});
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    NewCI->addAttribute(AttributeList::FunctionIndex,
                        Attribute::InaccessibleMemOnly);
    Changed = true;
  }
  // Erase deepest-token-first so no still-referenced marker is freed early:
  // reverse the collection order (begin, port, lines, depth) -> (depth ... begin).
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty()) Begin->eraseFromParent();
  if (PortMk && PortMk->use_empty()) PortMk->eraseFromParent();
  if (LinesMk && LinesMk->use_empty()) LinesMk->eraseFromParent();
  if (DepthMk && DepthMk->use_empty()) DepthMk->eraseFromParent();
  return Changed;
}

bool injectMAxiSpecInterface(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();

  // Find existing xlx_m_axi sideeffect calls (emitted by injectMAxiSideeffect).
  // Each gives us (kernel function, m_axi arg index, depth, bundle name).
  struct Site {
    Function *Kernel;
    Value *Ptr;
    uint64_t Depth;
    std::string Bundle;
    bool OffsetDirect = false;
  };
  SmallVector<Site, 8> Sites;
  Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  if (!SE)
    return false;
  for (User *U : SE->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    // Look for `xlx_m_axi` operand bundle.
    for (unsigned BI = 0; BI < CI->getNumOperandBundles(); ++BI) {
      OperandBundleUse OBU = CI->getOperandBundleAt(BI);
      if (OBU.getTagName() != "xlx_m_axi")
        continue;
      // Bundle args: 0=ptr, 1=bundle name (i8 array constant or empty),
      // 2=i64 depth, 3-15=others.
      if (OBU.Inputs.size() < 3)
        continue;
      Site S;
      S.Kernel = CI->getFunction();
      S.Ptr = OBU.Inputs[0];
      auto *DepthC = dyn_cast<ConstantInt>(OBU.Inputs[2]);
      S.Depth = DepthC ? DepthC->getZExtValue() : 0;
      // Skip the -1 sentinel (auto-infer); cosim TB won't get useful info.
      if (S.Depth == 0 || S.Depth == (uint64_t)-1)
        continue;
      // Bundle name: ConstantDataArray of i8, or zero-init [0 x i8].
      if (auto *CDA = dyn_cast<ConstantDataArray>(OBU.Inputs[1])) {
        if (CDA->isString())
          S.Bundle = CDA->getAsString().str();
      }
      if (S.Bundle.empty())
        S.Bundle = "gmem"; // C++ default bundle name
      // Slot 3 = offset_mode. Empty (zero array) → "slave" default; "direct"
      // string → no s_axi_control register, address passed at m_axi port.
      if (OBU.Inputs.size() >= 4) {
        if (auto *CDA = dyn_cast<ConstantDataArray>(OBU.Inputs[3])) {
          if (CDA->isString() && CDA->getAsString() == "direct")
            S.OffsetDirect = true;
        }
      }
      Sites.push_back(S);
    }
  }
  if (Sites.empty())
    return false;

  // Declare or fetch `_ssdm_op_SpecInterface` and `_ssdm_op_SpecBitsMap`.
  // Both are variadic, return void.
  // SpecBitsMap referencing %gmem is what keeps the adaptor pointer
  // alive through the backend's DCE — without it the adaptor is dropped
  // and the cosim TB generator never sees the depth.
  FunctionType *SITy =
      FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/true);
  FunctionCallee SIFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SITy);
  FunctionCallee SBMFn =
      M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SITy);

  Type *I32 = IntegerType::get(Ctx, 32);
  Type *I8 = IntegerType::get(Ctx, 8);
  PointerType *GmemPtrTy = PointerType::get(I32, /*addrspace=*/1);
  Constant *NullGmem = ConstantPointerNull::get(GmemPtrTy);

  // Each string literal must be a private global like
  // `@N = private unnamed_addr constant [Len x i8] c"...\00"`, passed
  // as `[Len x i8]*`. csynth checks that string
  // arg slots are pointers, not values — emitting `[N x i8]` directly
  // triggers `Error: ssdm parameter not matched, index: K`. So we wrap
  // each literal in a private global and use the pointer.
  StringMap<Constant *> StrGlobals;
  auto strPtr = [&](StringRef s) -> Constant * {
    auto It = StrGlobals.find(s);
    if (It != StrGlobals.end())
      return It->second;
    Constant *Init = ConstantDataArray::getString(Ctx, s, /*AddNull=*/true);
    auto *GV = new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                                  GlobalValue::PrivateLinkage, Init,
                                  ".str.maxi");
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    Constant *Ptr = GV;
    StrGlobals[s] = Ptr;
    return Ptr;
  };
  Constant *EmptyStrPtr = strPtr("");
  Constant *MAxiStrPtr = strPtr("m_axi");
  Constant *SlaveStrPtr = strPtr("slave");
  Constant *DirectStrPtr = strPtr("direct");

  // Group sites by kernel so we emit at the entry of each.
  StringMap<SmallVector<Site *, 4>> ByKernel;
  for (Site &S : Sites)
    ByKernel[S.Kernel->getName()].push_back(&S);
  for (auto &KV : ByKernel) {
    Function *F = M.getFunction(KV.first());
    if (!F)
      continue;
    // array_stencil kernels: the src/dst m_axi args flow through the backend's
    // line-buffer/burst adaptor (created from the xlx_m_axi sideeffect bundle),
    // which yields exactly ONE real `SpecInterface(ptr addrspace(1) %gmem1)`.
    // Emitting our phantom
    // `SpecInterface(i32 addrspace(1)* null, depth N)` here adds a SECOND,
    // null-backed m_axi port that the backend carries through. During top
    // scheduling the LatencyEstimator dereferences that null port and crashes
    // (Abnormal 11 / getEstimatedSchedLatency). Skip
    // the phantom for stencil kernels.
    // injectMAxiSpecInterface runs BEFORE injectArrayStencil, so the
    // `fpga_array_stencil` op-bundle is not emitted yet. Detect the stencil
    // kernel by the still-present `__vxx_array_stencil` marker call instead.
    bool IsStencilKernel = false;
    if (Function *StencilMarker = M.getFunction("__vxx_array_stencil")) {
      for (User *U : StencilMarker->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          if (CI->getFunction() == F) {
            IsStencilKernel = true;
            break;
          }
        }
      }
    }
    if (IsStencilKernel) {
      // Still strip dereferenceable/align off the m_axi args (cosim TB needs the
      // depth-derived helper type), but emit no phantom SpecInterface/BitsMap.
      for (Site *S : KV.second) {
        if (auto *ArgV = dyn_cast<Argument>(S->Ptr)) {
          ArgV->removeAttr(Attribute::Dereferenceable);
          ArgV->removeAttr(Attribute::Alignment);
        }
      }
      vxxDbg() << "vxx: skipped phantom m_axi SpecInterface for stencil "
                "kernel " << F->getName() << "\n";
      continue;
    }
    IRBuilder<> B(&F->getEntryBlock(),
                  F->getEntryBlock().getFirstInsertionPt());
    for (Site *S : KV.second) {
      Constant *BundleStrPtr = strPtr(S->Bundle);
      Constant *OffsetModeStrPtr = S->OffsetDirect ? DirectStrPtr : SlaveStrPtr;
      Value *Args[] = {
          NullGmem,                                    // 0
          MAxiStrPtr,                                  // 1
          ConstantInt::get(I32, 0),                    // 2
          ConstantInt::get(I32, 0),                    // 3
          EmptyStrPtr,                                 // 4
          ConstantInt::get(I32, 0),                    // 5
          ConstantInt::get(I32, S->Depth),             // 6 ★ depth
          BundleStrPtr,                                // 7
          OffsetModeStrPtr,                            // 8 ★ slave|direct
          EmptyStrPtr,                                 // 9
          ConstantInt::get(I32, 16), ConstantInt::get(I32, 16), // 10,11
          ConstantInt::get(I32, 16), ConstantInt::get(I32, 16), // 12,13
          EmptyStrPtr,                                 // 14
          EmptyStrPtr,                                 // 15
          ConstantInt::get(I32, (uint64_t)-1, true),   // 16: latency
          ConstantInt::get(I32, 0),                    // 17
          ConstantInt::get(I32, 0),                    // 18
          ConstantInt::get(I32, 4),                    // 19: alignment
      };
      B.CreateCall(SIFn, Args);
      // Also emit SpecBitsMap(%gmem) to mark the adaptor as live so
      // the backend's DCE doesn't drop the SpecInterface call. Both calls
      // must appear together for each m_axi bundle.
      B.CreateCall(SBMFn, ArrayRef<Value *>{NullGmem});
      // Strip `dereferenceable(N)` and `align N` from the m_axi arg.
      // Rustc emits these from `&[T; N]` reference attribs; the cosim TB generator
      // appears to read them and decides the helper type is plain
      // `i32*` (single element) instead of the depth-derived
      // `[N x i32]*`.
      if (auto *ArgV = dyn_cast<Argument>(S->Ptr)) {
        ArgV->removeAttr(Attribute::Dereferenceable);
        ArgV->removeAttr(Attribute::Alignment);
      }
      Changed = true;
    }
  }
  if (Changed) {
    vxxDbg() << "vxx: injected " << Sites.size()
           << " m_axi SpecInterface call(s)\n";
  }
  return Changed;
}



// Inject `_ssdm_op_ReadReq.m_axi.pXiY(gep_base, count)` and `_ssdm_op_WriteReq`
// calls at top fn entry for each m_axi-marked arg. These are normally emitted
// automatically (auto-burst inference); Rust kernel body's byte-arith pattern
// (`mul i64 idx, 4` + `add base + ofs`) confuses the auto-burst inference, so we
// hoist explicit ReadReq/WriteReq from the m_axi marker depth + offset_noalias
// SpecInterface info.
bool injectMAxiReadWriteReq(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);

  // Walk xlx_m_axi bundles. For each (kernel, arg_ptr, depth, isWrite),
  // emit a `_ssdm_op_{Read,Write}Req.m_axi.<mangled>` call at fn entry on
  // the arg_ptr (cast through bitcast to i32 addrspace(1)*).
  // Auto-burst inference emits this from `arr[i][j]` access
  // patterns; rustc's GEPs are equivalent but the backend can't infer bursts
  // from them once it byte-lowers them — so we hoist the markers in source.
  struct Site {
    Function *F;
    Value *ArgPtr;       // [N x T]* kernel arg or similar
    uint64_t Depth;
    bool IsWrite;
  };
  SmallVector<Site, 8> Sites;

  Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  if (!SE) return false;
  for (User *U : SE->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Function *F = CI->getFunction();
    if (!F || F->isDeclaration()) continue;
    for (unsigned BI = 0; BI < CI->getNumOperandBundles(); ++BI) {
      OperandBundleUse OBU = CI->getOperandBundleAt(BI);
      if (OBU.getTagName() != "xlx_m_axi") continue;
      if (OBU.Inputs.size() < 3) continue;
      Value *ArgPtr = OBU.Inputs[0];
      auto *DC = dyn_cast<ConstantInt>(OBU.Inputs[2]);
      if (!DC) continue;
      uint64_t D = DC->getZExtValue();
      if (D == 0 || D == (uint64_t)-1) continue;
      // Skip if ArgPtr isn't a kernel arg ptr (we only emit ReadReq for
      // top-level args, not for nested ptrs).
      if (!isa<Argument>(ArgPtr)) continue;
      Sites.push_back({F, ArgPtr, D, /*IsWrite=*/false});
    }
  }
  if (Sites.empty()) return false;

  // Heuristic for read/write direction: scan kernel body for stores
  // through this arg's GEP chain. Pure-load → ReadReq; pure-store →
  // WriteReq; mixed → emit BOTH (read + write).
  for (auto &S : Sites) {
    auto *ArgA = dyn_cast<Argument>(S.ArgPtr);
    if (!ArgA) continue;
    bool HasStore = false, HasLoad = false;
    SmallPtrSet<Value *, 16> Derived; Derived.insert(ArgA);
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (BasicBlock &BB : *S.F) {
        for (Instruction &I : BB) {
          if (Derived.count(&I)) continue;
          // Propagate: GEP/bitcast of derived → derived
          if (auto *G = dyn_cast<GetElementPtrInst>(&I)) {
            if (Derived.count(G->getPointerOperand())) {
              Derived.insert(G); Changed = true;
            }
          } else if (auto *BC = dyn_cast<BitCastInst>(&I)) {
            if (Derived.count(BC->getOperand(0))) {
              Derived.insert(BC); Changed = true;
            }
          }
        }
      }
    }
    for (BasicBlock &BB : *S.F) {
      for (Instruction &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (Derived.count(SI->getPointerOperand())) HasStore = true;
        } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (Derived.count(LI->getPointerOperand())) HasLoad = true;
        }
      }
    }
    S.IsWrite = HasStore && !HasLoad;
  }

  // Emit ReadReq/WriteReq at function entry (right before original first
  // non-_ssdm instr). The ptr cast is from `[N x T]*` to `i32 addrspace(1)*`
  // — the backend propagates the cast and rewrites the call appropriately.
  PointerType *GmemI32Ty = PointerType::get(Type::getInt32Ty(Ctx), 1);
  unsigned Emitted = 0;
  for (auto &S : Sites) {
    BasicBlock &Entry = S.F->getEntryBlock();
    Instruction *InsertBefore = nullptr;
    for (Instruction &I : Entry) {
      // Skip alloca/store/load that rustc emits before the SpecInterface
      // sideeffects, plus the SpecInterface calls themselves.
      if (isa<AllocaInst>(I)) continue;
      auto *CI = dyn_cast<CallInst>(&I);
      if (CI && CI->getCalledFunction() &&
          (CI->getCalledFunction()->getName().startswith("_ssdm_op_Spec") ||
           CI->getCalledFunction()->getName() == "llvm.sideeffect" ||
           CI->getCalledFunction()->getName().startswith("llvm.dbg") ||
           CI->getCalledFunction()->getName().startswith("llvm.lifetime"))) {
        continue;
      }
      // bitcast/store/load: also skip ahead — these may be rustc preamble
      // for stack-spilled args.
      if (isa<BitCastInst>(I) || isa<StoreInst>(I) || isa<LoadInst>(I)) continue;
      InsertBefore = &I;
      break;
    }
    if (!InsertBefore) InsertBefore = Entry.getTerminator();
    IRBuilder<> B(InsertBefore);
    // Explicitly clear DebugLoc — VXXIRDowngrader's spFlags/retainedNodes
    // strip breaks DI scope chains; inheriting a broken DebugLoc causes
    // the backend's ADCE collectLiveScopes to segfault on null scope parent.
    // No-DebugLoc instructions are skipped by ADCE's scope walk.
    B.SetCurrentDebugLocation(DebugLoc());

    Value *CastPtr = S.ArgPtr;
    if (CastPtr->getType() != GmemI32Ty) {
      auto *PT = cast<PointerType>(CastPtr->getType());
      // Order matters: bitcast to i32* in the source addrspace first,
      // THEN addrspacecast to addrspace(1). LLVM rules:
      //  - bitcast can change pointee type but NOT addrspace
      //  - addrspacecast can change addrspace but NOT pointee type
      // The previous order (addrspacecast first) created a
      // `[100 x float] addrspace(1)*` intermediate, which LLVM 7's
      // ConstantExpr::getAddrSpaceCast doesn't fold cleanly → crash.
      if (PT->getElementType() != Type::getInt32Ty(Ctx)) {
        PointerType *I32SrcPT =
            PointerType::get(Type::getInt32Ty(Ctx), PT->getAddressSpace());
        CastPtr = B.CreateBitCast(CastPtr, I32SrcPT);
      }
      auto *NowPT = cast<PointerType>(CastPtr->getType());
      if (NowPT->getAddressSpace() != 1) {
        CastPtr = B.CreateAddrSpaceCast(CastPtr, GmemI32Ty);
      }
    }

    StringRef IntrName = S.IsWrite ? "_ssdm_op_WriteReq.m_axi.p1i32"
                                    : "_ssdm_op_ReadReq.m_axi.p1i32";
    FunctionType *RWTy =
        FunctionType::get(I1, {GmemI32Ty, I64}, /*Var=*/false);
    FunctionCallee RWFn = M.getOrInsertFunction(IntrName, RWTy);
    B.CreateCall(RWFn, {CastPtr, ConstantInt::get(I64, S.Depth)});
    Emitted++;
  }
  if (Emitted)
    vxxDbg() << "VXXPrep: injected " << Emitted
           << " m_axi ReadReq/WriteReq call(s)\n";
  return Emitted > 0;
}



// The legacy `__vxx_maxi_config` marker (latency / burst-length /
// outstanding knobs) was removed when m_axi unified onto a single marker —
// those slots never reached the `xlx_m_axi` op-bundle anyway (hardcoded to
// the -1 sentinels). This stays as a harmless no-op (the marker no longer
// exists, so the drop finds nothing) to keep the pass list stable.
bool injectMaxiConfig(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_config");
}



bool injectMaxiIntrinsics(Module &M) {
  bool A = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_read_i");
  bool B = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_read_req_i");
  bool C = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_write_i");
  bool D = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_write_req_i");
  bool E = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_maxi_write_response_i");
  return A || B || C || D || E;
}



bool injectSimplePtrBundle(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_top_param");
}



// Rename `%barista_hls::BurstMaxi<i32>` → `%class.hls::burst_maxi<int>` and
// strip its `[0 x i64]` padding fields (rustc emits a single-field
// `#[repr(C)] BurstMaxi { ptr: *mut T }` as `{ [0 x i64], i32*, [0 x i64] }`
// — three fields, only the middle is real).
//
// Why: the HLS backend rejects struct-by-value top-fn args containing pointers
// (HLS 214-298 "Struct type with pointer type inside on top function
// argument is not supported, please disaggregate"). The exception is
// `class.hls::burst_maxi<*>` (and a few other recognised template names),
// which the HLS pipeline recognises. We rename the rustc-emitted struct to match,
// after which the backend lets it through (and the cosim TB's _ir wrapper loads
// the inner ptr field).
//
// The strip+rewrite logic mirrors stripZeroSizeArrayStructFields but is
// scoped to BurstMaxi names so we don't relax the multi-field guard there.
// Also rewrites all `extractvalue %BurstMaxi, 1` → `extractvalue %BurstMaxi,
// 0` and equivalent insertvalue / GEP index shifts.
bool renameAndStripBurstMaxi(Module &M) {
  Function *KernelMarker = M.getFunction("__vxx_top_kernel");
  if (!KernelMarker || KernelMarker->user_empty())
    return false;

  SmallVector<StructType *, 4> Targets;
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (ST->isOpaque())
      continue;
    StringRef N = ST->getName();
    if (!N.startswith("barista_hls::BurstMaxi<"))
      continue;
    Targets.push_back(ST);
  }
  if (Targets.empty())
    return false;

  // LLVM-rule-compliant replacement (StructType bodies are immutable once set,
  // so the in-place setBody loop below was illegal — now dead): build a NEW
  // burst_maxi struct with the canonical C++ name + the single ptr field, then
  // remap the whole module (by-value signature, extractvalue/insertvalue/GEP
  // field selectors, globals) old → new.
  {
    DenseMap<Type *, Type *> TypeMap;
    DenseMap<StructType *, std::vector<int>> FieldMaps;
    for (StructType *ST : Targets) {
      int RealIdx = -1;
      Type *RealField = nullptr;
      const DataLayout &DL = M.getDataLayout();
      for (unsigned i = 0; i < ST->getNumElements(); ++i) {
        Type *FT = ST->getElementType(i);
        // Padding = any ZERO-SIZE field: `[0 x T]` alignment markers AND
        // `core::marker::PhantomData<T>` (an empty struct — the elem-type
        // tag on the promoted `*mut i32` handle).
        bool IsZSA = FT->isSized() && DL.getTypeAllocSize(FT) == 0;
        if (!IsZSA) {
          if (RealIdx >= 0) { RealIdx = -1; break; }
          RealIdx = (int)i;
          RealField = FT;
        }
      }
      if (RealIdx < 0 || !RealField || !RealField->isPointerTy())
        continue;
      std::string OldName = ST->getName().str();
      std::string Inner = OldName.substr(strlen("barista_hls::BurstMaxi<"));
      if (!Inner.empty() && Inner.back() == '>')
        Inner.pop_back();
      std::string CName = Inner;
      if (Inner == "i32") CName = "int";
      else if (Inner == "i64") CName = "long";
      else if (Inner == "i16") CName = "short";
      else if (Inner == "i8") CName = "char";
      else if (Inner == "u32") CName = "unsigned int";
      else if (Inner == "u64") CName = "unsigned long";
      // ApInt/ApUint elements: the handle field is ALREADY the promoted
      // `i32*` (BurstMaxi<T> stores `*mut i32` + PhantomData<T>; C++
      // `burst_maxi<ap_int<W>>` keeps its m_axi data path at p1i32) — only
      // the canonical NAME needs the int spelling. Unifying EVERY width to
      // `burst_maxi<int>` is intentional and sound: all widths share the
      // identical `{ i32* }` body (element conversion happens at the access
      // via BurstElem), so distinct-width handles in one module (e.g.
      // manual_burst_with_conditionals' ApInt<30>+ApInt<20>) merge into one
      // layout-identical struct.
      if (Inner.find("ApInt<") != std::string::npos ||
          Inner.find("ApUint<") != std::string::npos)
        CName = "int";
      std::string NewName = "class.hls::burst_maxi<" + CName + ">";
      std::vector<int> Map(ST->getNumElements(), -1);
      Map[RealIdx] = 0;
      ST->setName(OldName + ".old"); // free the name for the replacement
      StructType *NewST = M.getTypeByName(NewName);
      if (!NewST || NewST->getNumElements() != 1 ||
          NewST->getElementType(0) != RealField)
        NewST = StructType::create(M.getContext(), {RealField},
                                   NewName, ST->isPacked());
      TypeMap[ST] = NewST;
      FieldMaps[ST] = Map;
      vxxDbg() << "vxx: renamed " << OldName << " → " << NewName
             << " (new-struct remap)\n";
    }
    hlsrs::vxx::remapStructsInModule(M, TypeMap, &FieldMaps);
    return !TypeMap.empty();
  }

  for (StructType *ST : Targets) {
    // Find the single non-ZSA field (the ptr) and its index.
    int RealIdx = -1;
    Type *RealField = nullptr;
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *FT = ST->getElementType(i);
      bool IsZSA = false;
      if (auto *AT = dyn_cast<ArrayType>(FT))
        if (AT->getNumElements() == 0)
          IsZSA = true;
      if (!IsZSA) {
        if (RealIdx >= 0) {
          vxxDbg() << "vxx: BurstMaxi has >1 real field, skipping "
                 << ST->getName() << "\n";
          RealIdx = -1;
          break;
        }
        RealIdx = (int)i;
        RealField = FT;
      }
    }
    if (RealIdx < 0 || !RealField || !RealField->isPointerTy())
      continue;

    // Strip padding: setBody to just the real field.
    ST->setBody({RealField}, /*Packed=*/false);

    // Rename: `barista_hls::BurstMaxi<i32>` → `class.hls::burst_maxi<int>`.
    // Map common Rust int names to C++ canonical names. Default: lowercase.
    std::string OldName = ST->getName().str();
    std::string Inner = OldName.substr(strlen("barista_hls::BurstMaxi<"));
    if (!Inner.empty() && Inner.back() == '>')
      Inner.pop_back();
    std::string CName = Inner;
    if (Inner == "i32") CName = "int";
    else if (Inner == "i64") CName = "long";
    else if (Inner == "i16") CName = "short";
    else if (Inner == "i8")  CName = "char";
    else if (Inner == "u32") CName = "unsigned int";
    else if (Inner == "u64") CName = "unsigned long";
    std::string NewName = "class.hls::burst_maxi<" + CName + ">";
    ST->setName(NewName);

    // Rewrite all extractvalue/insertvalue/GEP that index this struct
    // by old field index → new index 0. Collect into worklists first
    // (mutating during iteration would skip neighbours after erase).
    SmallVector<ExtractValueInst *, 8> EVs;
    SmallVector<InsertValueInst *, 8> IVs;
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
            if (EV->getAggregateOperand()->getType() != ST) continue;
            ArrayRef<unsigned> Idx = EV->getIndices();
            if (!Idx.empty() && (int)Idx[0] == RealIdx)
              EVs.push_back(EV);
          } else if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
            if (IV->getAggregateOperand()->getType() != ST) continue;
            ArrayRef<unsigned> Idx = IV->getIndices();
            if (!Idx.empty() && (int)Idx[0] == RealIdx)
              IVs.push_back(IV);
          }
        }
      }
    }
    for (ExtractValueInst *EV : EVs) {
      ArrayRef<unsigned> Idx = EV->getIndices();
      SmallVector<unsigned, 4> NewIdx(Idx.begin(), Idx.end());
      NewIdx[0] = 0;
      IRBuilder<> B(EV);
      Value *NewEV = B.CreateExtractValue(EV->getAggregateOperand(),
                                          NewIdx, EV->getName());
      EV->replaceAllUsesWith(NewEV);
      EV->eraseFromParent();
    }
    for (InsertValueInst *IV : IVs) {
      ArrayRef<unsigned> Idx = IV->getIndices();
      SmallVector<unsigned, 4> NewIdx(Idx.begin(), Idx.end());
      NewIdx[0] = 0;
      IRBuilder<> B(IV);
      Value *NewIV = B.CreateInsertValue(IV->getAggregateOperand(),
                                          IV->getInsertedValueOperand(),
                                          NewIdx, IV->getName());
      IV->replaceAllUsesWith(NewIV);
      IV->eraseFromParent();
    }
    for (Function &F : M) {
      SmallVector<GetElementPtrInst *, 8> ToRewrite;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
            if (GEP->getSourceElementType() != ST) continue;
            ToRewrite.push_back(GEP);
          }
        }
      }
      for (GetElementPtrInst *GEP : ToRewrite) {
        // After strip, the struct is `{ <real-field> }` (single field at
        // byte offset 0). All in-struct GEPs (regardless of which old
        // field they indexed: leading padding `(0,0,0)`, real field
        // `(0,RealIdx)` or `(0,RealIdx,0..)`, trailing padding) point to
        // byte offset 0. Replace each GEP with a bitcast of the struct
        // pointer to the GEP's result type — semantically equivalent at
        // offset 0, and avoids producing GEP indices that no longer match
        // the new field layout.
        IRBuilder<> B(GEP);
        Value *Src = GEP->getPointerOperand();
        Value *NewVal;
        if (GEP->getType() == Src->getType()) {
          NewVal = Src;
        } else {
          NewVal = B.CreateBitCast(Src, GEP->getType());
        }
        GEP->replaceAllUsesWith(NewVal);
        if (NewVal != Src) {
          if (auto *NewBC = dyn_cast<Instruction>(NewVal))
            NewBC->takeName(GEP);
        }
        GEP->eraseFromParent();
      }
    }

    vxxDbg() << "vxx: renamed " << OldName << " → " << NewName
           << " (stripped padding, real field idx " << RealIdx << ")\n";
  }
  return !Targets.empty();
}



// Stamp `fpga.decayed.dim.hint` on undecayed `[N x T]*` m_axi array args that
// lack it, to match the opaque-ptr ports (e.g. coeff `[256 x i8]*` →
// dim.hint="256"). Gated to args with a `__vxx_m_axi` marker.
bool stampUndecayedArrayMAxiDimHint(Module &M) {
  static const char *const MAxiMarkers[] = {
      "__vxx_m_axi",
  };
  bool Changed = false;
  for (const char *MN : MAxiMarkers) {
    Function *F = M.getFunction(MN);
    if (!F) continue;
    for (User *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getNumArgOperands() < 1) continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg) continue;
      auto *PT = dyn_cast<PointerType>(Arg->getType());
      if (!PT) continue;
      auto *AT = dyn_cast<ArrayType>(PT->getElementType());
      if (!AT) continue; // only pointer-to-array (undecayed) args
      if (Arg->getParent()->getAttributes().getParamAttributes(Arg->getArgNo())
              .hasAttribute("fpga.decayed.dim.hint"))
        continue; // already stamped (decayed path)
      Arg->addAttr(Attribute::get(M.getContext(), "fpga.decayed.dim.hint",
                                  std::to_string(AT->getNumElements())));
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: stamped dim.hint on undecayed m_axi array args\n";
  return Changed;
}



} } // namespace hlsrs::vxx
