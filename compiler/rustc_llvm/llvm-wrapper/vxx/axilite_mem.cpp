//===----------------------------------------------------------------------===//
//
// axilite_mem.cpp — s_axilite / ap_memory / ap_fifo / ap_scalar interface passes.
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

// For each top kernel (marked by `__vxx_top_kernel`), emit the
// canonical kernel header:
//   - `_ssdm_op_SpecTopModule(<fn_name_str>)` at entry
//   - For each arg without a mode-specific marker:
//       `_ssdm_op_SpecBitsMap(arg), !map !{!{}}`
//       `_ssdm_op_SpecInterface(arg, "ap_auto", ...20-arg-shape)`
// Without these, the HLS backend falls back to `ap_ctrl_none` (combinational)
// for plain scalar kernels — kills ap_start/done/idle/ready handshake.
// `ap_auto` defers protocol selection to the HLS backend's auto-detection,
// which lands on `ap_ctrl_hs` when no s_axilite/m_axi indicators are present.
//
// Args with mode-specific markers (m_axi/axis/stream/directio/bram/
// ap_memory/s_axilite_port) are skipped — their own inject passes emit
// the canonical SpecInterface; double-emit causes SYN 201-504.
bool injectDefaultApAutoSpec(Module &M) {
  // Collect top fns via the `fpga.top.func` attribute that
  // injectKernelTopAttribute set from the `__vxx_top_kernel()` marker.
  // We can't use the marker directly: injectKernelTopAttribute already
  // erased its call sites by the time we run.
  SmallPtrSet<Function*, 4> Tops;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("fpga.top.func")) continue;
    // Skip top fns binding a uram_ecc memory (kind 29): emitting
    // SpecInterface/SpecTopModule on the array args (HLS auto-infers the
    // BRAM/control interface) makes our ap_auto emit conflict with the URAM
    // scheduling and SIGSEGV the LatencyEstimator.
    // uram_ecc is ecc_flags-only, so no other example is
    // affected. This pass runs before injectBindStorageSpecResource, so the
    // bind is still the `__vxx_bind_storage(ptr, kind, lat)` marker (kind=arg 1).
    bool HasUramEcc = false;
    if (Function *BSF = M.getFunction("__vxx_bind_storage")) {
      for (User *U : BSF->users()) {
        auto *CI = dyn_cast<CallInst>(U);
        if (!CI || CI->getFunction() != &F || CI->arg_size() < 2) continue;
        if (auto *K = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
          if (K->getZExtValue() == 29) { HasUramEcc = true; break; }
      }
    }
    if (HasUramEcc) continue;
    Tops.insert(&F);
  }
  if (Tops.empty()) return false;

  // Args that already have a non-default SpecInterface emit — skip
  // ap_auto for these. By the time this pass runs (Phase 6, after
  // injectMAxiSpecInterface / injectApFifo / injectAxis / etc.), the
  // markers themselves are dropped but the resulting `SpecInterface`
  // calls carry a non-"" mode string ("ap_fifo", "m_axi", "axis", etc.).
  // Reading those calls is more reliable than walking the (dropped)
  // marker chains.
  DenseSet<std::pair<Function*, unsigned>> MarkedArgs;
  for (Function *Top : Tops) {
    Function *SIFn = M.getFunction("_ssdm_op_SpecInterface");
    if (!SIFn) break;
    for (User *U : SIFn->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getFunction() != Top) continue;
      if (CI->arg_size() < 2) continue;
      // arg 0: the kernel arg (or value derived from it). arg 1: mode
      // string global. Trace arg 0 to its Argument.
      Argument *A = resolveMarkerArg(CI->getArgOperand(0));
      if (A && A->getParent() == Top)
        MarkedArgs.insert({Top, A->getArgNo()});
    }
    // Also pick up args carried by `llvm.sideeffect [ "xlx_*"(arg, ...) ]`
    // op-bundles (m_axi/axis/ap_fifo/etc), which don't appear in
    // `_ssdm_op_SpecInterface` calls. Without this, a kernel like
    // `aliasing_axi_master_ports` (xlx_m_axi op-bundle, no SpecInterface
    // for its aximA/aximB args) gets spurious SpecInterface(ap_auto)
    // emitted by us → segfault during csynth / export.
    Function *Side = M.getFunction("llvm.sideeffect");
    if (Side) {
      for (User *U : Side->users()) {
        auto *CI = dyn_cast<CallInst>(U);
        if (!CI || CI->getFunction() != Top) continue;
        for (unsigned BI = 0; BI < CI->getNumOperandBundles(); ++BI) {
          auto OB = CI->getOperandBundleAt(BI);
          StringRef Tag = OB.getTagName();
          if (!Tag.startswith("xlx_")) continue;
          if (OB.Inputs.empty()) continue;
          Argument *A = resolveMarkerArg(OB.Inputs[0].get());
          if (A && A->getParent() == Top)
            MarkedArgs.insert({Top, A->getArgNo()});
        }
      }
    }
  }

  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  FunctionCallee SBMFn = M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SpecTy);
  // Both are declared with `nounwind`.
  if (auto *F = dyn_cast<Function>(SpecFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  if (auto *F = dyn_cast<Function>(SBMFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);

  GlobalVariable *ApAutoStr = getOrCreateCStrGlobal(M, "ap_auto");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");

  // !map metadata signals to the HLS backend that the SpecBitsMap is canonical
  // (skip auto-add of ap_auto a second time).
  MDNode *MapInner = MDNode::get(Ctx, ArrayRef<Metadata*>{});
  MDNode *MapMD = MDNode::get(Ctx, ArrayRef<Metadata*>{MapInner});

  // Stencil tops carry an m_axi interface (handled by injectMAxiSpecInterface)
  // and a pipelined 2D loop that the HLS backend outlines into separate
  // `_Pipeline_` leaf modules. They reach this pass incidentally: after the
  // m_axi reads lower to `_ssdm_op_Read.m_axi`/intrinsics the body has no
  // *user* calls, so the combinational gate (b) below fires. Emitting our own
  // bare SpecTopModule here then collides with the one the HLS backend always
  // adds from `set_top` (there must be exactly ONE, with !fpga.pragma.source) →
  // the resulting double-declared top makes Architecture Synthesis inline the
  // outlined leaves back into the top, and the inlined burst-in-pipelined-loop
  // yields a null PerformanceInfo → getEstimatedSchedLatency SIGSEGV (stencil_2d
  // crash). Stencil needs none of the ap_auto combinational treatment, so skip
  // it wholesale here and let the HLS backend supply the single canonical
  // SpecTopModule.
  SmallPtrSet<Function*, 4> StencilTops;
  if (Function *ASten = M.getFunction("__vxx_array_stencil"))
    for (User *U : ASten->users())
      if (auto *SCI = dyn_cast<CallInst>(U))
        if (Function *F = SCI->getFunction()) StencilTops.insert(F);

  // FFT tops (callers of the barista_fft_proxy shim) carry hls::stream + i1
  // status ports whose interfaces are set explicitly (renameFftComplexArgs +
  // ap_fifo/stream markers). Adding a default ap_auto SpecInterface here would
  // double-spec those ports -> SYN 201-504. Skip them entirely (same rationale
  // as StencilTops).
  SmallPtrSet<Function*, 4> FftTops;
  for (const char *PName : {"barista_fft_proxy", "barista_fft_proxy_ssr",
                            "barista_fft_proxy_ssr_arr"})
    if (Function *PF = M.getFunction(PName))
      for (User *U : PF->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (Function *F = CI->getFunction()) FftTops.insert(F);

  bool Changed = false;
  for (Function *Top : Tops) {
    if (StencilTops.count(Top)) continue;
    if (FftTops.count(Top)) continue;
    // Gate 1: fire when EITHER
    //   (a) the top has at least one stream-marked arg (FFT pattern), OR
    //   (b) the top body has NO calls to non-marker user functions
    //       (pure-combinational kernel, e.g. using_C++_templates).
    //
    // Rationale: case (a) silences the HLS backend's "auto-emit ap_auto on
    // every arg when one is stream" behavior. Case (b) handles
    // kernels that need SpecInterface(ap_auto) on every arg — without our
    // injection, HLS adds a spurious ap_clk
    // port.  Avoid PerformanceInfo segfault on rtl_as_blackbox
    // style scalar-ref kernels by REQUIRING the body has no calls.
    bool HasMarkedArg = false;
    for (Argument &A : Top->args())
      if (MarkedArgs.count({Top, A.getArgNo()})) { HasMarkedArg = true; break; }

    bool BodyHasUserCall = false;
    for (BasicBlock &BB : *Top) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee) { BodyHasUserCall = true; break; }
        StringRef N = Callee->getName();
        if (N.startswith("_ssdm_") || N.startswith("llvm.") ||
            N.startswith("__vxx_") || N.startswith("__vxxprep_"))
          continue;
        BodyHasUserCall = true;
        break;
      }
      if (BodyHasUserCall) break;
    }

    if (!HasMarkedArg && BodyHasUserCall) continue;

    // Gate 2: skip if the top has a non-void return type. HLS treats
    // return values as implicit ap_return ports — adding ap_auto to
    // scalar args of a function with return value causes
    // PerformanceInfo segfault.
    if (!Top->getReturnType()->isVoidTy()) continue;

    BasicBlock &Entry = Top->getEntryBlock();
    IRBuilder<> B(&*Entry.getFirstInsertionPt());

    // NO SpecTopModule here. The HLS backend ALWAYS adds its own
    // SpecTopModule (with !fpga.pragma.source) from `set_top`; a second, bare
    // one from us makes the scheduler's module-level PerformanceInfo
    // inconsistent, and with a burst-inferred m_axi access inside an
    // explicitly pipelined loop that surfaces as a null PerformanceInfo →
    // LatencyEstimator::initialize → getEstimatedSchedLatency SIGSEGV right
    // after "Finished scheduling" (maxi_cache_conditional /
    // auto_burst_inference_failure / aggregation_of_nested_structs; same
    // mechanism as the stencil_2d double-top crash documented above).
    // Removing only the duplicate fixes all three; C++'s a.pp.bc carries
    // exactly one SpecTopModule for the same reason.

    // Per-arg SpecBitsMap + SpecInterface("ap_auto").
    // - INTEGER-typed scalar args (i1, i8, i16, i32, i64): always.
    // - PTR-to-scalar args (i8*, i16*, i32*, i64*): only when
    //   BodyHasUserCall is false (pure-combinational, like
    //   using_C++_templates) — these need SpecInterface(ap_auto)
    //   on the ptr so HLS doesn't add a spurious ap_clk. We avoid
    //   the rtl_as_blackbox PerformanceInfo segfault case by
    //   gating on BodyHasUserCall.
    GlobalVariable *ApMemoryStr = getOrCreateCStrGlobal(M, "ap_memory");
    GlobalVariable *ApFifoStr = getOrCreateCStrGlobal(M, "ap_fifo");
    // SpecStream decl for FIFO emit (matches injectApFifo pattern).
    FunctionCallee SSFn =
        M.getOrInsertFunction("_ssdm_SpecStream", SpecTy);
    if (auto *F = dyn_cast<Function>(SSFn.getCallee()))
      F->addFnAttr(Attribute::NoUnwind);
    // For each arg, count uses inside llvm.fpga.fifo.{pop,push} intrinsics
    // — these tell us the arg is a FIFO port (Stream<T>* in Rust).
    // Depth = total pops + pushes (matches Vitis behavior of inferring
    // depth from usage; user can still override with explicit
    // stream_depth marker).
    DenseMap<Argument *, unsigned> FifoDepth;
    for (BasicBlock &BB : *Top) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee) continue;
        StringRef N = Callee->getName();
        if (!N.startswith("llvm.fpga.fifo.pop.") &&
            !N.startswith("llvm.fpga.fifo.push.")) continue;
        // Pop: arg 0 is the FIFO ptr. Push: arg 1 is the FIFO ptr.
        unsigned ArgIdx = N.startswith("llvm.fpga.fifo.pop.") ? 0 : 1;
        if (CI->arg_size() <= ArgIdx) continue;
        Argument *A = resolveMarkerArg(CI->getArgOperand(ArgIdx));
        if (A && A->getParent() == Top) FifoDepth[A]++;
      }
    }
    bool AllowPtr = !BodyHasUserCall;
    // AXIS sub-channel args (data/keep/strb/user/last/id/dest) are grouped
    // into ONE axis interface by disaggAxisStructKernelSig's grouped
    // SpecInterface("axis") + SpecAXISSideChannel. They must NOT also get a
    // per-channel SpecInterface("ap_auto") here — that scalar interface makes
    // cosim TB model each sub-channel as a scalar `Register` (reading *param =
    // the queue-handle bytes) instead of a `Stream<Byte<N>>` that pops the
    // queue, so cosim captures all-zero input (212-361). The C++ reference emits
    // ONLY the grouped axis interface for these, no per-channel ap_auto.
    // Collect every arg referenced by a _ssdm_op_SpecAXISSideChannel call.
    DenseSet<Argument *> AxisChanArgs;
    for (BasicBlock &BB : *Top) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        if (CI->getCalledFunction()->getName() != "_ssdm_op_SpecAXISSideChannel")
          continue;
        for (Value *Op : CI->args())
          if (auto *A = dyn_cast<Argument>(Op->stripPointerCasts()))
            if (A->getParent() == Top) AxisChanArgs.insert(A);
      }
    }
    for (Argument &A : Top->args()) {
      if (MarkedArgs.count({Top, A.getArgNo()})) continue;
      if (AxisChanArgs.count(&A)) continue;  // AXIS sub-channel: grouped axis only
      Type *AT = A.getType();
      bool IsInt = AT->isIntegerTy();
      bool IsPtrToScalar = false;
      bool IsPtrToArray = false;
      if (auto *PT = dyn_cast<PointerType>(AT)) {
        Type *ET = PT->getElementType();
        IsPtrToScalar = ET->isIntegerTy();
        if (auto *AT2 = dyn_cast<ArrayType>(ET))
          IsPtrToArray = AT2->getElementType()->isIntegerTy();
      }
      // FIFO-used arg → emit ap_fifo (independent of AllowPtr gate).
      bool IsFifo = FifoDepth.count(&A) > 0;
      if (!IsInt && !IsFifo &&
          !(AllowPtr && (IsPtrToScalar || IsPtrToArray))) continue;
      // Mode string + arg layout (per-mode shape):
      //   ap_fifo:   mode_str, -1, -1, "", 0, depth, ...     (FIFO depth at pos 6)
      //   ap_memory: mode_str,  0,  0, "", -1, 0, ...        (depth=-1 at pos 5)
      //   ap_auto:   mode_str,  0,  0, "", 0, 0, ...         (no depth slot)
      Value *ModeStr;
      Value *Arg2, *Arg3, *Arg5, *Arg6;
      Value *Zero = ConstantInt::get(I32, 0);
      Value *MOne = ConstantInt::getSigned(I32, -1);
      if (IsFifo) {
        ModeStr = (Value *)ApFifoStr;
        Arg2 = MOne; Arg3 = MOne;
        Arg5 = Zero;
        Arg6 = (Value *)ConstantInt::get(I32, FifoDepth[&A]);
      } else if (IsPtrToArray && !IsInt && !IsPtrToScalar) {
        ModeStr = (Value *)ApMemoryStr;
        Arg2 = Zero; Arg3 = Zero;
        Arg5 = MOne; Arg6 = Zero;
      } else {
        ModeStr = (Value *)ApAutoStr;
        Arg2 = Zero; Arg3 = Zero;
        Arg5 = Zero; Arg6 = Zero;
      }
      // For FIFO args: also emit SpecStream (matches injectApFifo
      // pattern) and strip the nocapture attribute. Stream<T>* args need
      // these; without them the HLS backend auto-adds a duplicate
      // SpecInterface(ap_auto) → 201-504.
      if (IsFifo) {
        Top->removeParamAttr(A.getArgNo(), Attribute::NoCapture);
        Top->removeParamAttr(A.getArgNo(), Attribute::Alignment);
        Value *Depth = Arg6;  // already set to fifo depth above
        B.CreateCall(SSFn, {&A, Zero, Depth, EmptyStr});
      }
      // Order matters: emit SpecInterface FIRST, then SpecBitsMap.
      // Reverse order (SpecBitsMap first) triggers the HLS backend
      // to auto-add a duplicate SpecInterface(ap_auto), causing
      // SYN 201-504 conflict.
      B.CreateCall(SpecFn,
        {&A,
         ModeStr,
         Arg2, Arg3,
         EmptyStr,
         Arg5, Arg6,
         EmptyStr, EmptyStr, EmptyStr,
         Zero, Zero, Zero, Zero,
         EmptyStr, EmptyStr,
         MOne,
         Zero, Zero, Zero});
      CallInst *SBM = B.CreateCall(SBMFn, {&A});
      SBM->setMetadata("map", MapMD);
    }
    Changed = true;
  }
  return Changed;
}

// Interface markers: just drop. The SpecInterface arg layout has ~20 fields
// that vary by mode; emitting a wrong-arity call causes Vitis HLS to SIGSEGV
// in "Analysis Interface Mode". Without an explicit SpecInterface call HLS
// infers the interface from the IR pattern (ptr + load/store = ap_memory;
// stream intrinsics = ap_fifo; etc) — which is usually correct.
// For each `__vxx_ap_fifo(ptr, depth, bundle)` marker, emit
// `_ssdm_op_SpecInterface(ptr, "ap_fifo", ...)` at containing function
// entry. Without it HLS infers ap_memory (BRAM) for array-pointer args
// instead of ap_fifo (FIFO pop/push) — produces BRAM ports
// (`arr_address0/ce0/q0`) instead of FIFO ports (`arr_dout/empty_n/read`).
// Rewrite GEP-indexed load/store on ap_fifo'd kernel args into
// `llvm.fpga.fifo.{pop,push}` intrinsic calls. CURRENTLY DISABLED via
// injectApFifo (drop-only) — body rewrite alone doesn't help HLS recognize
// the FIFO pattern; would need _ssdm_op_IfRead.Stream-style intrinsic.
bool rewriteApFifoBody(Module &M,
                              DenseMap<Argument*, uint64_t> &OutDepth) {
  Function *Marker = M.getFunction("__vxx_ap_fifo");
  if (!Marker) return false;
  // Collect ap_fifo'd Argument*s AND depths. Filter to PRIMITIVE-typed
  // args only — struct/array-of-array types need IfRead.Stream-style
  // rewrite (deferred). Skip those, leave markers for injectApFifo to drop.
  SmallPtrSet<Argument *, 8> Targets;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 1) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Type *T = Arg->getType();
    if (!T->isPointerTy()) continue;
    Type *ET = cast<PointerType>(T)->getElementType();
    Type *Elem = ET->isArrayTy() ? cast<ArrayType>(ET)->getElementType() : ET;
    if (!(Elem->isIntegerTy() || Elem->isFloatingPointTy())) continue;
    // Only wide-element types (i128+, f64+). For i32/i16/i8 arrays, HLS
    // produces ap_vld scalar interface from IfRead.Stream pattern (not
    // FIFO din/dout) — wrong shape. Wide elements get FIFO interface.
    unsigned Bits = Elem->isIntegerTy()
                        ? cast<IntegerType>(Elem)->getBitWidth()
                        : Elem->getPrimitiveSizeInBits();
    if (Bits < 64) continue;
    Targets.insert(Arg);
    MarkerCalls.push_back(CI);
    uint64_t Depth = 0;
    if (CI->arg_size() >= 5)
      if (auto *DC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        Depth = DC->getZExtValue();
    OutDepth[Arg] = Depth;
  }
  for (CallInst *CI : MarkerCalls) CI->eraseFromParent();
  if (Targets.empty()) return false;

  bool Changed = false;
  // Strip dereferenceable + align attrs from ap_fifo'd args — HLS uses
  // these to infer memory access and may emit ap_none scalar port instead
  // of FIFO din/dout/empty_n triple.
  for (Argument *Arg : Targets) {
    Arg->removeAttr(Attribute::Dereferenceable);
    Arg->removeAttr(Attribute::DereferenceableOrNull);
    Arg->removeAttr(Attribute::Alignment);
    Arg->removeAttr(Attribute::NoAlias);
    Arg->removeAttr(Attribute::NoCapture);
    Arg->removeAttr(Attribute::ReadOnly);
    Arg->removeAttr(Attribute::WriteOnly);
    Changed = true;
  }
  // For each target arg, find GEP users and rewrite their load/store users.
  for (Argument *Arg : Targets) {
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    // ElemTy = scalar element type. After decay it's iN directly. For
    // undecayed `[N x iN]*`, peel to iN so we rewrite the body's GEP+load
    // pattern to fpga.fifo.pop. Filtered to 64-bit+ above.
    Type *ElemTy = PT->getElementType();
    if (ElemTy->isArrayTy()) ElemTy = cast<ArrayType>(ElemTy)->getElementType();
    if (!ElemTy->isIntegerTy()) continue;
    IntegerType *IntTy = cast<IntegerType>(ElemTy);

    // Use llvm.fpga.fifo.pop/push — HLS recognizes these intrinsics as
    // ap_fifo stream operations and emits dout/empty_n/read (read) or
    // din/full_n/write (write) ports. Works for single-load-per-iter
    // (aggregation_of_struct u128 args).
    Function *PopFn = hlsrs::vxx::getOrInsertFifoPopAny(M, IntTy);
    Function *PushFn = hlsrs::vxx::getOrInsertFifoPushAny(M, IntTy);
    SmallVector<Instruction *, 16> Dead;
    // Collect all GEPs reachable from Arg through bitcast chains.
    // decayKernelArrayParams introduces `%cast = bitcast T* %arg to [N x T]*`
    // and the body's GEPs operate on %cast — so direct Arg->users() won't
    // find them.
    SmallVector<GetElementPtrInst *, 8> Geps;
    SmallVector<Value *, 8> ToScan{Arg};
    SmallPtrSet<Value *, 16> Visited;
    while (!ToScan.empty()) {
      Value *V = ToScan.pop_back_val();
      if (!Visited.insert(V).second) continue;
      for (User *U : V->users()) {
        if (auto *G = dyn_cast<GetElementPtrInst>(U)) {
          Geps.push_back(G);
        } else if (isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U)) {
          ToScan.push_back(U);
        }
      }
    }
    for (GetElementPtrInst *GEP : Geps) {
      // Last index must produce ElemTy*. For decayed `T* %arg + i` it's a
      // 2-operand GEP. For undecayed `[N x T]* %cast + 0, i` it's a 3-operand
      // GEP. Both produce T*.
      if (GEP->getResultElementType() != ElemTy) continue;

      // fpga.fifo.pop/push intrinsic expects `iN*` ptr. Bitcast undecayed
      // `[N x iN]*` → `iN*` for the pop/push call.
      PointerType *ElemPtrTy = PointerType::get(IntTy, 0);
      SmallVector<User *, 4> GepUsers(GEP->users());
      for (User *GU : GepUsers) {
        if (auto *LI = dyn_cast<LoadInst>(GU)) {
          if (LI->getType() != IntTy) continue;
          IRBuilder<> B(LI);
          Value *ArgCast = Arg->getType() == ElemPtrTy
                              ? (Value*)Arg
                              : B.CreateBitCast(Arg, ElemPtrTy);
          Value *Popped = B.CreateCall(PopFn, {ArgCast});
          LI->replaceAllUsesWith(Popped);
          Dead.push_back(LI);
          Changed = true;
        } else if (auto *SI = dyn_cast<StoreInst>(GU)) {
          if (SI->getValueOperand()->getType() != IntTy) continue;
          IRBuilder<> B(SI);
          Value *ArgCast = Arg->getType() == ElemPtrTy
                              ? (Value*)Arg
                              : B.CreateBitCast(Arg, ElemPtrTy);
          B.CreateCall(PushFn, {SI->getValueOperand(), ArgCast});
          Dead.push_back(SI);
          Changed = true;
        }
      }
    }
    for (Instruction *I : Dead) I->eraseFromParent();

    // Strip dead chain reachable from Arg (bitcast → bitcast → GEP →
    // load that's been replaced, etc.). DFS from Arg, mark instructions
    // as dead if all their users are also dead or are dead now.
    // Simple iterative DCE on Arg's reachable instructions:
    bool ChangedDCE = true;
    while (ChangedDCE) {
      ChangedDCE = false;
      // Collect all instrs reachable from Arg through pointer-bitcasts/GEPs.
      SmallVector<Value *, 16> Worklist{Arg};
      SmallPtrSet<Value *, 32> Reachable;
      while (!Worklist.empty()) {
        Value *V = Worklist.pop_back_val();
        if (!Reachable.insert(V).second) continue;
        for (User *U : V->users()) {
          if (auto *I = dyn_cast<Instruction>(U)) {
            if (isa<BitCastInst>(I) || isa<GetElementPtrInst>(I) ||
                isa<AddrSpaceCastInst>(I))
              Worklist.push_back(I);
          }
        }
      }
      // Erase any reachable instruction (excluding Arg itself) that has
      // no users. Iterate in reverse order so users get cleaned first.
      SmallVector<Instruction *, 16> Cands;
      for (Value *V : Reachable) {
        if (V == Arg) continue;
        if (auto *I = dyn_cast<Instruction>(V))
          if (I->use_empty()) Cands.push_back(I);
      }
      for (Instruction *I : Cands) {
        I->eraseFromParent();
        ChangedDCE = true;
      }
    }
  }
  return Changed;
}

// Emit SpecStream + SpecInterface(ap_fifo) + SpecBitsMap
// for each `__vxx_ap_fifo(arr, ..., depth)` marker. Args must be pre-decayed
// to scalar pointer by decayKernelArrayParams (above) so HLS sees `T*` not
// `[N x T]*`. SpecStream BEFORE SpecInterface is critical: it tells the HLS
// backend the arg is a stream pointer, preventing the auto-add of `ap_auto`
// which would otherwise trigger SYN 201-504 "Interface type specification
// conflicts".
//
// Emitted shape:
//   call void @_ssdm_SpecStream(i32* %in1, i32 0, i32 0, [1 x i8]* @0)
//   call void @_ssdm_op_SpecInterface(i32* %in1, [8 x i8]* @1 "ap_fifo",
//       i32 0, i32 0, [1 x i8]* @0, i32 0, i32 0, [1 x i8]* @0, [1 x i8]* @0,
//       [1 x i8]* @0, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @0, [1 x i8]* @0,
//       i32 -1, i32 0, i32 0, i32 0)
//   call void @_ssdm_op_SpecBitsMap(i32* %in1), !map !N
bool injectApFifo(Module &M) {
  Function *Marker = M.getFunction("__vxx_ap_fifo");
  if (!Marker) return false;
  // First: rewrite body GEP+load/store on ap_fifo'd args → fpga.fifo.pop/push
  // intrinsics. Without this, HLS sees array-style memory access on a port
  // marked `ap_fifo` and raises 201-504 "Interface type specification
  // conflicts" (the body looks like ap_memory but the marker says ap_fifo).
  // Filters to wide-element types (i64+, f64+) — narrow types use a different
  // (ap_vld scalar) path. The body rewrite ALSO consumes the markers it
  // processed, so the loop below only sees markers on narrow args (which it
  // still wants for SpecInterface emit).
  DenseMap<Argument *, uint64_t> ApFifoDepth;
  (void)rewriteApFifoBody(M, ApFifoDepth);
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  FunctionCallee SBMFn = M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SpecTy);
  // These are declared with `#0 = { nounwind }`. Add it here.
  if (auto *F = dyn_cast<Function>(SpecFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  if (auto *F = dyn_cast<Function>(SBMFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *FifoStr = getOrCreateCStrGlobal(M, "ap_fifo");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Zero = ConstantInt::get(I32, 0);
  Constant *NegOne = ConstantInt::get(I32, -1);
  // `!map !{!{}}` metadata required on SpecBitsMap — the HLS backend uses this
  // as the "this arg's interface is canonical, do not auto-add ap_auto"
  // signal. Stamp it on every SpecBitsMap.
  MDNode *MapMD = MDNode::get(Ctx, {MDNode::get(Ctx, {})});
  // ssdm calls all carry `nounwind` attr.
  AttrBuilder NwAttrs;
  NwAttrs.addAttribute(Attribute::NoUnwind);
  AttributeList NwAttrList = AttributeList::get(Ctx, AttributeList::FunctionIndex, NwAttrs);
  // Pass globals as `[N x i8]*` directly (NOT i8* via GEP).
  SmallSet<std::pair<Function*, unsigned>, 8> Emitted;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Function *F = Arg->getParent();
    if (!Emitted.insert({F, Arg->getArgNo()}).second) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    if (isa<ArrayType>(PT->getElementType())) continue;  // decay required
    // Strip `fpga.signal_name` string attr — emitted by decay pass to suppress
    // HLS auto ap_auto add. But on args that ALSO get SpecInterface(ap_fifo),
    // HLS sees the signal_name as a separate scalar interface mode hint and
    // raises SYN 201-504 "Interface type specification conflicts". ap_fifo args
    // should carry only `noalias` (`i32* noalias %d_s_in`), no signal_name.
    {
      Function *FP = Arg->getParent();
      AttributeSet PA = FP->getAttributes().getParamAttributes(Arg->getArgNo());
      if (PA.hasAttribute("fpga.signal_name"))
        FP->removeParamAttr(Arg->getArgNo(), "fpga.signal_name");
    }
    // Strip `nocapture` attr — ap_fifo args should have `noalias`
    // only. nocapture may confuse HLS's stream port inference (saw 201-504).
    F->removeParamAttr(Arg->getArgNo(), Attribute::NoCapture);
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    // Extract user-specified depth from the marker call's 5th arg
    // (signature: `__vxx_ap_fifo(arg, register_flag, name_ptr,
    // name_len, depth_i64)` per barista-hls/src/lib.rs:804).
    Value *DepthVal = Zero;
    if (CI->arg_size() >= 5)
      if (auto *DepthC = dyn_cast<ConstantInt>(CI->getArgOperand(4))) {
        uint64_t D = DepthC->getZExtValue();
        DepthVal = ConstantInt::get(I32, D);
      }
    // 1. SpecStream(arg, 0, depth, "") — required for stream
    //    args. Depth must match expected FIFO depth or the HLS backend inserts
    //    a phantom volatile load on the output-only port (SYNCHK 200-91 in
    //    FFT placeholder bodies). Auto_disagg works with depth=0 because
    //    the loop trip count is small; FFT needs depth=1024.
    FunctionType *SSTy = FunctionType::get(Type::getVoidTy(M.getContext()), /*Var=*/true);
    FunctionCallee SSFn = M.getOrInsertFunction("_ssdm_SpecStream", SSTy);
    if (auto *FF = dyn_cast<Function>(SSFn.getCallee())) FF->addFnAttr(Attribute::NoUnwind);
    CallInst *SSCall = B.CreateCall(SSFn,
        {(Value*)Arg, Zero, DepthVal, (Value*)EmptyStr});
    SSCall->setAttributes(NwAttrList);
    // 2. SpecInterface(arg, "ap_fifo", -1, -1, "", 0, depth, ...).
    //    Depth at index 6 (zero-based) must match.
    Value *SIArgs[] = {
        (Value*)Arg, (Value*)FifoStr, NegOne, NegOne, (Value*)EmptyStr, Zero, DepthVal,
        (Value*)EmptyStr, (Value*)EmptyStr, (Value*)EmptyStr,
        Zero, Zero, Zero, Zero,
        (Value*)EmptyStr, (Value*)EmptyStr, NegOne, Zero, Zero, Zero};
    CallInst *SICall = B.CreateCall(SpecFn, SIArgs);
    SICall->setAttributes(NwAttrList);
    // 3. SpecBitsMap(arg) with !map metadata — the HLS backend's canonical signal
    CallInst *SBMCall = B.CreateCall(SBMFn, ArrayRef<Value*>{(Value*)Arg});
    SBMCall->setMetadata("map", MapMD);
    SBMCall->setAttributes(NwAttrList);
    // (Dead bitcast cleanup deferred to final stripTopKernelNoalias pass
    // — at injectApFifo time, body might still reference the bitcast via
    // decayKernelArrayParams's bridge.)
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_ap_fifo");
  return true;
}




// Emit `llvm.sideeffect [ "xlx_ap_<mode>"(ptr, ...) ]` for each
// `__vxx_ap_scalar(ptr, mode, register)` marker. Modes:
//   0 → xlx_ap_none, 1 → xlx_ap_ack, 2 → xlx_ap_vld, 3 → xlx_ap_ovld,
//   4 → xlx_ap_hs,    5 → xlx_ap_stable
// Without this bundle, HLS sees `volatile T*` kernel args + no interface
// pragma → falls into SVUVM dataflow_monitor scaffold which xelab can't
// elab (svr_pkg/allsvr_*_done undeclared errors).
bool injectApScalar(Module &M) {
  Function *Marker = M.getFunction("__vxx_ap_scalar");
  if (!Marker) return false;

  LLVMContext &Ctx = M.getContext();
  Type *I32 = IntegerType::get(Ctx, 32);
  Type *I64 = IntegerType::get(Ctx, 64);
  Constant *EmptyArr = ConstantAggregateZero::get(ArrayType::get(IntegerType::get(Ctx, 8), 0));
  Function *Sideeffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);

  // Collect marker call info first (Arg + Mode), then emit. Group by
  // function so each xlx_ap_* call lands at function entry.
  struct Spec { Argument *Arg; unsigned Mode; };
  StringMap<SmallVector<Spec, 4>> ByFn;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    MarkerCalls.push_back(CI);
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!ModeC) continue;
    unsigned Mode = ModeC->getZExtValue();
    if (Mode > 5) continue;
    ByFn[Arg->getParent()->getName()].push_back({Arg, Mode});
  }

  // Cache `_ssdm_op_SpecInterface` decl + an empty-string global.
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  if (auto *F = dyn_cast<Function>(SpecFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  // SpecInterface mode string per ap_scalar mode.
  static const char *SpecModeStrings[] = {
      "ap_none", "ap_ack", "ap_vld", "ap_ovld", "ap_hs", "ap_stable"};

  bool Changed = false;
  for (auto &KV : ByFn) {
    Function *F = M.getFunction(KV.first());
    if (!F) continue;
    IRBuilder<> B(&F->getEntryBlock(),
                  F->getEntryBlock().getFirstInsertionPt());
    for (auto &S : KV.second) {
      static const char *Names[] = {
          "xlx_ap_none", "xlx_ap_ack", "xlx_ap_vld",
          "xlx_ap_ovld", "xlx_ap_hs",  "xlx_ap_stable"};
      const char *BundleName = Names[S.Mode];
      // Also emit `_ssdm_op_SpecInterface(arg, "<mode>", -1, -1, ...)`
      // for every interface-tagged scalar.
      // Without it HLS's IP-XACT export omits the matching handshake
      // signals (e.g. `d_i_ap_ack`, `d_i_ap_vld` on basic_pointers).
      // The mode-string match also lets `injectDefaultApAutoSpec`'s
      // MarkedArgs detection skip ap_auto emit on this arg.
      //
      // Only emit SpecInterface on kernels where NO arg has any other
      // interface SpecInterface emitted yet (simple ap_*-only kernels
      // like basic_pointers / stream_better).  Mixed-marker kernels
      // (m_axi + AXIS + ap_hs scalar, e.g.
      // Task_level_Parallelism/Control_driven/directio/ap_hs) tickle a
      // HLS PerformanceInfo segfault when ap_hs SpecInterface is added
      // alongside m_axi/axis SpecInterface — skip on those.
      bool AnyOtherSpecOnF = false;
      if (Function *SIFn = M.getFunction("_ssdm_op_SpecInterface")) {
        for (User *U : SIFn->users()) {
          auto *CI2 = dyn_cast<CallInst>(U);
          if (CI2 && CI2->getFunction() == F) { AnyOtherSpecOnF = true; break; }
        }
      }
      if (!AnyOtherSpecOnF &&
          S.Mode < (sizeof(SpecModeStrings)/sizeof(*SpecModeStrings))) {
        GlobalVariable *ModeStr =
            getOrCreateCStrGlobal(M, SpecModeStrings[S.Mode]);
        Value *MOne = ConstantInt::get(I32, (uint64_t)-1, true);
        Value *Zero = ConstantInt::get(I32, 0);
        B.CreateCall(SpecFn,
          {S.Arg,
           ModeStr,
           MOne, MOne,
           EmptyStr,
           Zero, Zero,
           EmptyStr, EmptyStr, EmptyStr,
           Zero, Zero, Zero, Zero,
           EmptyStr, EmptyStr,
           MOne,
           Zero, Zero, Zero});
      }
      // Bundle shape: (ptr, i32 -1, [0 x i8] zeroinitializer, i64 -1, i32 0)
      Value *MinusOne32 = ConstantInt::get(I32, (uint64_t)-1, true);
      Value *MinusOne64 = ConstantInt::get(I64, (uint64_t)-1, true);
      Value *Zero32 = ConstantInt::get(I32, 0);
      Value *Args[] = {S.Arg, MinusOne32, EmptyArr, MinusOne64, Zero32};
      OperandBundleDef OB(BundleName,
                          SmallVector<Value *, 5>(std::begin(Args), std::end(Args)));
      CallInst *NewCI = B.CreateCall(Sideeffect, {}, {OB});
      NewCI->addAttribute(AttributeList::FunctionIndex,
                          Attribute::InaccessibleMemOnly);
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      // Compute port bitwidth (e.g. i32 → 32). Stamp the attr.
      if (auto *PT = dyn_cast<PointerType>(S.Arg->getType())) {
        if (auto *IT = dyn_cast<IntegerType>(PT->getElementType())) {
          std::string BW = std::to_string(IT->getBitWidth());
          NewCI->addAttribute(AttributeList::FunctionIndex,
                              Attribute::get(Ctx, "xlx.port.bitwidth", BW));
        }
      }
      NewCI->addAttribute(AttributeList::FunctionIndex,
                          Attribute::get(Ctx, "xlx.source", "user"));
      Changed = true;
    }
  }

  // Erase markers + declaration regardless of whether we emitted bundles
  // (some markers may have arg resolution fail — still drop them).
  for (CallInst *CI : MarkerCalls) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  if (Changed) vxxDbg() << "vxx: injected " << ByFn.size()
                       << " xlx_ap_* sideeffect bundle(s)\n";
  return Changed;
}

// Emit SpecInterface(arg, "bram" | "ap_memory", ...) for the corresponding
// markers. BOTH ap_memory and bram args get an explicit SpecInterface — HLS
// uses the explicit emit instead of auto-defaulting, so they don't conflict.
//
// Per-arg shape:
//   SpecInterface(arg, "ap_memory" | "bram", 0, 0, "", -1, 0, "", "", "",
//                 0, 0, 0, 0, "", "", -1, 0, 0, 0)
// Args MUST be kept as `[N x T]*` (typed arrays), NOT decayed.
bool injectMemoryInterfaceMode(Module &M, StringRef MarkerName,
                                       StringRef ModeName) {
  Function *Marker = M.getFunction(MarkerName);
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  FunctionCallee SBMFn = M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SpecTy);
  if (auto *F = dyn_cast<Function>(SpecFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  if (auto *F = dyn_cast<Function>(SBMFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  // llvm.sideeffect intrinsic for op-bundle form (the canonical shape;
  // a SpecInterface call alone leaves the HLS backend free to auto-add
  // ap_memory on top — 203-801 conflict on c).
  Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  GlobalVariable *ModeStr = getOrCreateCStrGlobal(M, ModeName);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Zero = ConstantInt::get(I32, 0);
  Constant *NegOne = ConstantInt::get(I32, -1);
  Constant *NegOne64 = ConstantInt::get(I64, -1);
  Constant *Zero64 = ConstantInt::get(I64, 0);
  Constant *One64 = ConstantInt::get(I64, 1);
  // [0 x i8] zeroinitializer constant for sig_name placeholder
  ArrayType *ZeroByteArrTy = ArrayType::get(Type::getInt8Ty(Ctx), 0);
  Constant *ZeroByteArr = ConstantAggregateZero::get(ZeroByteArrTy);
  MDNode *MapMD = MDNode::get(Ctx, {MDNode::get(Ctx, {})});
  AttrBuilder NwAttrs;
  NwAttrs.addAttribute(Attribute::NoUnwind);
  AttributeList NwAttrList = AttributeList::get(Ctx, AttributeList::FunctionIndex, NwAttrs);

  // ap_memory → "xlx_ap_memory", bram → "xlx_bram". 9-arg form.
  StringRef BundleName = (ModeName == "bram") ? "xlx_bram" : "xlx_ap_memory";
  // bram uses i64 0 at arg 7, ap_memory uses i64 1.
  Constant *Arg7 = (ModeName == "bram") ? Zero64 : One64;

  // Pre-scan: which args also have `__vxx_aggregate` marker? Op-bundle emit
  // is needed ONLY for aggregate-paired args. Emitting the op-bundle on plain
  // ap_memory ports (no aggregate) makes cosim hang.
  DenseSet<Argument*> AggregateArgs;
  if (Function *AggMarker = M.getFunction("__vxx_aggregate")) {
    for (User *U : AggMarker->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() < 1) continue;
      if (Argument *A = resolveMarkerArg(CI->getArgOperand(0)))
        AggregateArgs.insert(A);
    }
  }

  bool Changed = false;
  SmallSet<std::pair<Function*, unsigned>, 8> Emitted;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 1) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    Function *F = Arg->getParent();
    if (!Emitted.insert({F, Arg->getArgNo()}).second) continue;
    if (!Arg->getType()->isPointerTy()) continue;
    // C++ keeps args as `[N x T]*` — do not decay.
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());

    // Emit op-bundle (canonical form) ONLY for aggregate-paired args. Plain
    // ap_memory ports without aggregate work fine via SpecInterface alone;
    // adding an op-bundle on them caused a cosim hang on an ap_fifo + ap_memory
    // mix.
    if (AggregateArgs.count(Arg)) {
      Value *BundleArgs[] = {
          (Value*)Arg,
          NegOne64, NegOne64, NegOne64,
          (Value*)ZeroByteArr,
          NegOne64, (Value*)ZeroByteArr,
          Arg7, Zero};
      OperandBundleDef OBD(BundleName.str(), ArrayRef<Value*>(BundleArgs, 9));
      CallInst *SECall = B.CreateCall(SEFn, None, ArrayRef<OperandBundleDef>{OBD});
      SECall->setOnlyAccessesInaccessibleMemory();
      SECall->setDoesNotThrow();
    }

    // Also emit SpecInterface+SpecBitsMap — required for csim_design / cosim
    // top fn discovery (without these, cosim_design errors with
    // "Top function not found: there is no function named 'top'"). Order:
    // op-bundle first (the HLS backend consumes it), SpecInterface second (csim
    // discovers it). The HLS backend dedupes them downstream.
    Value *Args[] = {
        (Value*)Arg, (Value*)ModeStr,
        Zero, Zero, (Value*)EmptyStr,
        NegOne, Zero,
        (Value*)EmptyStr, (Value*)EmptyStr, (Value*)EmptyStr,
        Zero, Zero, Zero, Zero,
        (Value*)EmptyStr, (Value*)EmptyStr,
        NegOne, Zero, Zero, Zero};
    CallInst *SICall = B.CreateCall(SpecFn, Args);
    SICall->setAttributes(NwAttrList);
    CallInst *SBMCall = B.CreateCall(SBMFn, ArrayRef<Value*>{(Value*)Arg});
    SBMCall->setMetadata("map", MapMD);
    SBMCall->setAttributes(NwAttrList);
    Changed = true;
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, MarkerName);
  return Changed;
}

bool injectMemoryInterface(Module &M) {
  bool A = injectMemoryInterfaceMode(M, "__vxx_bram", "bram");
  bool B = injectMemoryInterfaceMode(M, "__vxx_ap_memory", "ap_memory");
  // autoEmitApMemorySpecNarrow disabled: with op-bundle emit above, the HLS
  // backend handles the auto-add itself — no need for our prep to layer extra
  // SpecInterface calls (caused 203-801 conflict).
  // (definitions removed — reference-only)
  return A || B;
}

// For each `__vxx_s_axilite_return(bundle)` marker, emit a
// `_ssdm_op_SpecInterface(i32 0, "s_axilite", ..., bundle, ...)` at the
// containing function's entry (`#pragma HLS INTERFACE s_axilite port=return`
// equivalent) — required for HLS to infer ap_ctrl_chain (interrupt port).
// Without it, kernels with only m_axi/axis args default to ap_ctrl_hs
// (4 ap_* ports) instead of ap_ctrl_chain (interrupt + s_axi_control).
bool injectSAxiliteReturnSpec(Module &M) {
  Function *Marker = M.getFunction("__vxx_s_axilite_return");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  GlobalVariable *SaxiStr = getOrCreateCStrGlobal(M, "s_axilite");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
  Constant *SaxiPtr = ConstantExpr::getInBoundsGetElementPtr(
      SaxiStr->getValueType(), SaxiStr, Idxs);
  Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
      EmptyStr->getValueType(), EmptyStr, Idxs);

  // Track (Function*, bundle_str) — emit at most once per (fn, bundle).
  StringMap<bool> Seen;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Dead.push_back(CI);
    Function *F = CI->getFunction();
    if (!F || F->isDeclaration()) continue;
    // Bundle name from marker arg 0 (i8* to global string).
    // Rust `b"BUS_A"` byte-array literal is emitted by rustc as
    //   private unnamed_addr constant <{ [N x i8] }> <{ [N x i8] c"..." }>
    // i.e. a packed single-field struct wrapping a non-null-terminated
    // [N x i8]. Unwrap the struct, then read N bytes (length from marker
    // arg 1 if a usize literal is present, otherwise from array size).
    std::string Bundle = "control";
    if (CI->arg_size() >= 1) {
      Value *V = CI->getArgOperand(0);
      while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
      if (auto *CE = dyn_cast<ConstantExpr>(V))
        if (CE->getOpcode() == Instruction::GetElementPtr)
          V = CE->getOperand(0);
      if (auto *GV2 = dyn_cast<GlobalVariable>(V)) {
        Constant *RawInit = GV2->getInitializer();
        if (auto *CS = dyn_cast<ConstantStruct>(RawInit))
          if (CS->getNumOperands() == 1)
            RawInit = CS->getOperand(0);
        if (auto *Init = dyn_cast<ConstantDataArray>(RawInit)) {
          if (Init->isString()) {
            Bundle = Init->getAsString().str();
          } else if (Init->getElementType()->isIntegerTy(8)) {
            unsigned ByteCount = Init->getNumElements();
            if (CI->arg_size() >= 2) {
              if (auto *LenCI = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
                uint64_t L = LenCI->getZExtValue();
                if (L > 0 && L <= ByteCount) ByteCount = L;
              }
            }
            Bundle.clear();
            for (unsigned i = 0; i < ByteCount; ++i)
              Bundle.push_back((char)Init->getElementAsInteger(i));
          }
        }
      }
    }
    std::string Key = F->getName().str() + ":" + Bundle;
    auto Ins = Seen.try_emplace(Key, true);
    if (!Ins.second) continue;
    GlobalVariable *BundleStr = getOrCreateCStrGlobal(M, Bundle);
    Constant *BundlePtr = ConstantExpr::getInBoundsGetElementPtr(
        BundleStr->getValueType(), BundleStr, Idxs);
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    B.CreateCall(SpecFn,
                 {ConstantInt::get(I32, 0),       // 0: ptr=0 means "return"
                  SaxiPtr,                         // 1: "s_axilite"
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  BundlePtr,                       // 7: bundle name
                  EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

// For each `__vxx_s_axilite_port(ptr, bundle, offset)` marker on a non-return
// arg, emit a SpecInterface (ptr, "s_axilite", ..., bundle, ...).
// Without this, HLS defaults to ap_none for scalar args → exposed as
// top-level RTL port instead of bound to s_axi_control register.
bool injectSAxilitePortSpec(Module &M) {
  Function *Marker = M.getFunction("__vxx_s_axilite_port");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 8> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  FunctionCallee SBMFn = M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SpecTy);
  // `_ssdm_op_SpecInterface` / `_ssdm_op_SpecBitsMap` are declared
  // with the `nounwind` function attribute. Without it the HLS backend's
  // TOP-recognition logic rejects per-arg SpecInterface(i8* %arg,
  // "s_axilite", ...) calls.
  if (auto *F1 = dyn_cast<Function>(SpecFn.getCallee())) {
    F1->addFnAttr(Attribute::NoUnwind);
  }
  if (auto *F2 = dyn_cast<Function>(SBMFn.getCallee())) {
    F2->addFnAttr(Attribute::NoUnwind);
  }
  GlobalVariable *SaxiStr = getOrCreateCStrGlobal(M, "s_axilite");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  // Pass string globals to _ssdm_op_SpecInterface as `[N x i8]*` (the
  // global's natural type), NOT GEP-decayed `i8*`.
  // The HLS backend type-checks the SpecInterface call args by their static
  // LLVM type — GEP-decay to `i8*` makes HLS reject `_ssdm_op_SpecInterface
  // (i8* %arg, i8* <gep>, ...)` because it expects the 2nd arg to be a
  // typed array pointer, not a generic i8*.
  Constant *SaxiPtr = SaxiStr;
  Constant *EmptyPtr = EmptyStr;
  (void)I32;  // keep used elsewhere
  Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
  (void)Idxs;

  // Pre-collect args that are already m_axi-tagged (via xlx_m_axi op-bundle
  // on llvm.sideeffect). Those get s_axilite SpecInterface automatically
  // from injectMAxiSpecInterface; emitting another s_axilite here causes
  // HLS 200-1986 "Could not apply TOP directive" + clang segfault.
  SmallPtrSet<Argument *, 8> MAxiArgs;
  if (Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect)) {
    for (User *U : SE->users()) {
      auto *CI2 = dyn_cast<CallInst>(U);
      if (!CI2) continue;
      for (unsigned BI = 0; BI < CI2->getNumOperandBundles(); ++BI) {
        OperandBundleUse OBU = CI2->getOperandBundleAt(BI);
        if (OBU.getTagName() != "xlx_m_axi" || OBU.Inputs.empty()) continue;
        if (Argument *A = resolveMarkerArg(OBU.Inputs[0]))
          MAxiArgs.insert(A);
      }
    }
  }

  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Dead.push_back(CI);
    if (CI->arg_size() < 2) continue;
    // arg 0 = ptr (Argument bitcast i8*)
    Value *Ptr = CI->getArgOperand(0);
    while (auto *BC = dyn_cast<BitCastInst>(Ptr)) Ptr = BC->getOperand(0);
    while (auto *BCC = dyn_cast<BitCastOperator>(Ptr)) Ptr = BCC->getOperand(0);
    // Bundle name from arg 1 + length from arg 2 (Rust `&[u8]` pattern:
    // marker(ptr, bundle.as_ptr(), bundle.len(), ...)).
    // `b"BUS_A"` is NOT null-terminated, so `Init->isString()` is false
    // — use the explicit length from arg 2 to slice the right bytes.
    std::string Bundle = "control";
    Value *BV = CI->getArgOperand(1);
    while (auto *BC = dyn_cast<BitCastOperator>(BV)) BV = BC->getOperand(0);
    if (auto *CE = dyn_cast<ConstantExpr>(BV))
      if (CE->getOpcode() == Instruction::GetElementPtr) BV = CE->getOperand(0);
    if (auto *GV2 = dyn_cast<GlobalVariable>(BV)) {
      Constant *RawInit = GV2->getInitializer();
      // Rust wraps `b"X"` byte-array literals as `<{ [N x i8] }>` packed
      // struct (rustc emits `private unnamed_addr constant <{ [N x i8] }>
      // <{ [N x i8] c"..." }>`). Unwrap single-field struct.
      if (auto *CS = dyn_cast<ConstantStruct>(RawInit))
        if (CS->getNumOperands() == 1)
          RawInit = CS->getOperand(0);
      if (auto *Init = dyn_cast<ConstantDataArray>(RawInit)) {
        if (Init->isString()) {
          // Null-terminated CStr: getAsString strips trailing null.
          Bundle = Init->getAsString().str();
        } else if (Init->getElementType()->isIntegerTy(8)) {
          // Non-null-terminated [N x i8] (e.g. `b"BUS_A"`): use length arg.
          unsigned ByteCount = Init->getNumElements();
          if (CI->arg_size() >= 3) {
            if (auto *LenCI = dyn_cast<ConstantInt>(CI->getArgOperand(2))) {
              uint64_t L = LenCI->getZExtValue();
              if (L > 0 && L <= ByteCount) ByteCount = L;
            }
          }
          Bundle.clear();
          for (unsigned i = 0; i < ByteCount; ++i)
            Bundle.push_back((char)Init->getElementAsInteger(i));
        }
      }
    }
    GlobalVariable *BundleStr = getOrCreateCStrGlobal(M, Bundle);
    Constant *BundlePtr = BundleStr;  // C++ uses array-typed ptr, not GEP

    auto *ArgV = dyn_cast<Argument>(Ptr);
    // If Ptr resolved to an alloca, look for `store T %arg, T* %alloca`
    // pattern (Rust's `&i` on a scalar arg materializes via alloca + store).
    // Use the source arg as the SpecInterface target — emit dual
    // SpecInterface (s_axilite + ap_none) at entry.
    AllocaInst *Slot = nullptr;
    if (!ArgV) {
      Slot = dyn_cast<AllocaInst>(Ptr);
      if (Slot) {
        for (User *SU : Slot->users()) {
          auto *SI = dyn_cast<StoreInst>(SU);
          if (!SI) continue;
          if (auto *SrcA = dyn_cast<Argument>(SI->getValueOperand())) {
            ArgV = SrcA;
            break;
          }
        }
      }
    }
    if (!ArgV) continue;
    // m_axi args: emit an explicit `xlx_s_axilite` op-bundle for each
    // m_axi data pointer's offset register (src/dst/coeff in stencil). Without
    // it the HLS backend infers s_axilite from the m_axi-offset model but with
    // different offset fields (0,0 vs the expected -1,-1). The op-bundle form
    // (not SpecInterface) avoids HLS 200-1986. Gated to stencil kernels to keep
    // other examples' s_axilite handling unchanged.
    if (MAxiArgs.count(ArgV)) {
      Function *FK = CI->getFunction();
      bool IsStencil = false;
      if (Function *SM = M.getFunction("__vxx_array_stencil"))
        for (User *MU : SM->users())
          if (auto *MC = dyn_cast<CallInst>(MU))
            if (MC->getFunction() == FK) { IsStencil = true; break; }
      if (!IsStencil && FK)
        for (BasicBlock &BB : *FK)
          for (Instruction &I : BB)
            if (auto *C2 = dyn_cast<CallInst>(&I))
              for (unsigned bi = 0; bi < C2->getNumOperandBundles(); ++bi)
                if (C2->getOperandBundleAt(bi).getTagName() == "fpga_array_stencil")
                  IsStencil = true;
      if (IsStencil && FK) {
        Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
        Type *I8b = Type::getInt8Ty(Ctx);
        Type *I64b = Type::getInt64Ty(Ctx);
        Constant *EmptyArrB = ConstantAggregateZero::get(ArrayType::get(I8b, 0));
        Value *CtrlName =
            ConstantDataArray::getString(Ctx, Bundle, /*AddNull=*/false);
        Value *SaxArgs[] = {
            ArgV,                                       // 0: ptr
            CtrlName,                                   // 1: "control"
            ConstantInt::get(I64b, (uint64_t)-1, true), // 2: -1
            ConstantInt::get(I32, (uint64_t)-1, true),  // 3: -1
            EmptyArrB, EmptyArrB, EmptyArrB,            // 4,5,6
            ConstantInt::get(I32, 0),                   // 7
        };
        IRBuilder<> Bsax(&*FK->getEntryBlock().getFirstInsertionPt());
        OperandBundleDef SaxOB(
            "xlx_s_axilite",
            SmallVector<Value *, 8>(std::begin(SaxArgs), std::end(SaxArgs)));
        CallInst *SaxCI = Bsax.CreateCall(FunctionCallee(SE), {}, {SaxOB});
        SaxCI->addAttribute(AttributeList::FunctionIndex,
                            Attribute::InaccessibleMemOnly);
        SaxCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
        SaxCI->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(Ctx, "xlx.port.bitwidth", "0"));
        SaxCI->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(Ctx, "xlx.source", "user"));
        Changed = true;
      }
      continue;
    }
    // Insert at entry block.
    Function *F = CI->getFunction();
    if (!F) continue;
    // Build `!map !{!{}}` metadata for SpecBitsMap.
    LLVMContext &MdCtx = F->getContext();
    MDNode *EmptyMD = MDNode::get(MdCtx, {});
    MDNode *MapMD = MDNode::get(MdCtx, {EmptyMD});
    // Offset from arg 3 of __vxx_s_axilite_port(ptr, bundle, len, offset).
    // The offset is encoded as a DECIMAL STRING at SpecInterface
    // pos 8 (e.g. 0x20 → "32"). When no explicit offset, pass empty.
    std::string OffsetStr;
    if (CI->arg_size() >= 4) {
      Value *OV = CI->getArgOperand(3);
      if (auto *CIo = dyn_cast<ConstantInt>(OV)) {
        int64_t Off = CIo->getSExtValue();
        if (Off != 0) OffsetStr = std::to_string(Off);
      }
    }
    Constant *OffsetPtr = EmptyPtr;
    if (!OffsetStr.empty()) {
      OffsetPtr = getOrCreateCStrGlobal(M, OffsetStr);
    }
    GlobalVariable *ApAutoStr = getOrCreateCStrGlobal(M, "ap_auto");
    Constant *ApAutoPtr = ApAutoStr;
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    if (ArgV->getType()->isPointerTy()) {
      // Pointer arg: SpecBitsMap + dual SpecInterface (s_axilite + ap_auto).
      // SpecBitsMap FIRST then two SpecInterface calls.
      CallInst *SBM = B.CreateCall(SBMFn, ArrayRef<Value*>{ArgV});
      SBM->setMetadata("map", MapMD);
      // 1st: s_axilite mode with -1,-1 / offset / -1 at pos 16
      B.CreateCall(SpecFn,
                   {ArgV,                                      // 0
                    SaxiPtr,                                    // 1
                    ConstantInt::get(I32, (uint64_t)-1, true), // 2 = -1
                    ConstantInt::get(I32, (uint64_t)-1, true), // 3 = -1
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    BundlePtr,                                  // 7
                    OffsetPtr,                                  // 8: offset str
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, (uint64_t)-1, true), // 16 = -1
                    ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0)});
      // 2nd: ap_auto mode, all zeros except -1 at pos 16
      B.CreateCall(SpecFn,
                   {ArgV,
                    ApAutoPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, (uint64_t)-1, true),
                    ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0)});
    } else {
      // Scalar by-value arg: emit dual SpecInterface (s_axilite + ap_none)
      // with arg 16 = -1. SpecBitsMap first.
      GlobalVariable *ApNoneStr = getOrCreateCStrGlobal(M, "ap_none");
      Constant *ApNonePtr = ApNoneStr;  // array-typed, not GEP
      B.CreateCall(SBMFn, ArrayRef<Value*>{ArgV});
      B.CreateCall(SpecFn,
                   {ArgV,
                    SaxiPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    BundlePtr,
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
      B.CreateCall(SpecFn,
                   {ArgV,
                    ApNonePtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
    }
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  // Add `!axi4.slave.bundlemap = !{}` named module metadata whenever any
  // s_axilite SpecInterface is emitted. The HLS backend uses this to register
  // the bundle map (without it, per-arg s_axilite SpecInterface on pointer args
  // triggers HLS-200-1986).
  if (Changed) {
    if (!M.getNamedMetadata("axi4.slave.bundlemap")) {
      M.getOrInsertNamedMetadata("axi4.slave.bundlemap");
    }
  }
  return Changed;
}

// For each `__vxx_top_s_axilite_control()` marker, emit SpecInterface for
// the bound "control" bundle. Same shape as s_axilite_return but always
// uses "control" bundle. Triggers HLS ap_ctrl_chain inference.
bool injectSAxiliteControlSpec(Module &M) {
  Function *Marker = M.getFunction("__vxx_top_s_axilite_control");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  GlobalVariable *SaxiStr = getOrCreateCStrGlobal(M, "s_axilite");
  GlobalVariable *CtrlStr = getOrCreateCStrGlobal(M, "control");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
  Constant *SaxiPtr = ConstantExpr::getInBoundsGetElementPtr(
      SaxiStr->getValueType(), SaxiStr, Idxs);
  Constant *CtrlPtr = ConstantExpr::getInBoundsGetElementPtr(
      CtrlStr->getValueType(), CtrlStr, Idxs);
  Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
      EmptyStr->getValueType(), EmptyStr, Idxs);
  SmallPtrSet<Function *, 4> Tops;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Tops.insert(CI->getParent()->getParent());
    Dead.push_back(CI);
  }
  GlobalVariable *ApNoneStr = getOrCreateCStrGlobal(M, "ap_none");
  Constant *ApNonePtr = ConstantExpr::getInBoundsGetElementPtr(
      ApNoneStr->getValueType(), ApNoneStr, Idxs);
  FunctionCallee SBMFn = M.getOrInsertFunction("_ssdm_op_SpecBitsMap", SpecTy);
  for (Function *Top : Tops) {
    IRBuilder<> B(&*Top->getEntryBlock().getFirstInsertionPt());
    // Function-level "ap_ctrl_chain" via SpecInterface(0, "s_axilite", ...,
    // "control", ...) — same shape as s_axilite_return.
    B.CreateCall(SpecFn,
                 {ConstantInt::get(I32, 0),
                  SaxiPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  CtrlPtr, EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
    Changed = true;
    // Auto-fold ALL scalar kernel args into the control bundle. C++ pragma
    // `s_axilite port=return bundle=control` implicitly maps all scalar
    // args to the same control bundle (Vivado HLS default). Without this,
    // each scalar arg is exposed as a standalone RTL port (extra `i`,
    // `Iter`, etc) instead of being bound to the control bundle.
    for (Argument &A : Top->args()) {
      if (A.getType()->isPointerTy()) continue;
      // Skip args already covered by another SpecInterface call.
      bool Covered = false;
      if (Function *SIF = M.getFunction("_ssdm_op_SpecInterface")) {
        for (User *U : SIF->users()) {
          auto *CI2 = dyn_cast<CallInst>(U);
          if (!CI2 || CI2->arg_size() < 1) continue;
          Value *V = CI2->getArgOperand(0);
          while (auto *BC = dyn_cast<BitCastInst>(V)) V = BC->getOperand(0);
          if (V == &A) { Covered = true; break; }
        }
      }
      if (Covered) continue;
      B.CreateCall(SBMFn, ArrayRef<Value*>{&A});
      B.CreateCall(SpecFn,
                   {&A, SaxiPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    CtrlPtr, EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
      B.CreateCall(SpecFn,
                   {&A, ApNonePtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    EmptyPtr, EmptyPtr,
                    ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
    }
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

bool injectSAxiliteAll(Module &M) {
  Function *Marker = M.getFunction("__vxx_s_axilite");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  FunctionCallee SBMFn  = M.getOrInsertFunction("_ssdm_op_SpecBitsMap",   SpecTy);
  FunctionCallee TopFn  = M.getOrInsertFunction("_ssdm_op_SpecTopModule", SpecTy);
  if (auto *F1 = dyn_cast<Function>(SpecFn.getCallee())) F1->addFnAttr(Attribute::NoUnwind);
  if (auto *F2 = dyn_cast<Function>(SBMFn.getCallee()))  F2->addFnAttr(Attribute::NoUnwind);
  if (auto *F3 = dyn_cast<Function>(TopFn.getCallee()))  F3->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *SaxiStr   = getOrCreateCStrGlobal(M, "s_axilite");
  GlobalVariable *ApAutoStr = getOrCreateCStrGlobal(M, "ap_auto");
  GlobalVariable *EmptyStr  = getOrCreateCStrGlobal(M, "");
  Constant *SaxiPtr = SaxiStr;
  Constant *ApAutoPtr = ApAutoStr;
  Constant *EmptyPtr = EmptyStr;

  // Group markers by containing function, preserve source order.
  DenseMap<Function *, SmallVector<CallInst *, 8>> ByFn;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (Function *F = CI->getFunction())
        ByFn[F].push_back(CI);
  if (ByFn.empty()) return false;

  bool Changed = false;
  SmallVector<CallInst *, 16> Dead;
  MDNode *EmptyMD = MDNode::get(Ctx, ArrayRef<Metadata*>{});
  MDNode *MapMD   = MDNode::get(Ctx, ArrayRef<Metadata*>{EmptyMD});

  for (auto &Pair : ByFn) {
    Function *F = Pair.first;
    if (F->isDeclaration()) continue;
    // Sort markers by their position in the function (source order).
    auto &Calls = Pair.second;
    std::sort(Calls.begin(), Calls.end(), [](CallInst *A, CallInst *B) {
      return A->comesBefore(B);
    });

    // Separate port markers from return marker.
    SmallVector<std::tuple<Argument *, std::string, int64_t>, 8> Ports;
    std::string ReturnBundle;
    bool HasReturn = false;
    for (CallInst *CI : Calls) {
      Dead.push_back(CI);
      if (CI->arg_size() < 5) continue;
      std::string Port   = hlsrs::vxx::vxxParseRustStrSlice(CI->getArgOperand(0), CI->getArgOperand(1));
      std::string Bundle = hlsrs::vxx::vxxParseRustStrSlice(CI->getArgOperand(2), CI->getArgOperand(3));
      int64_t Offset = 0;
      if (auto *OffCI = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        Offset = OffCI->getSExtValue();
      if (Port == "return") {
        ReturnBundle = Bundle;
        HasReturn = true;
        continue;
      }
      // Look up Argument by name.
      Argument *Arg = nullptr;
      for (Argument &A : F->args()) {
        if (A.getName() == Port) { Arg = &A; break; }
      }
      if (!Arg) continue;
      Ports.emplace_back(Arg, Bundle, Offset);
    }

    // Single IRBuilder at entry — all calls go in this order.
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());

    // SpecTopModule(<fnname>) — emitted as the FIRST instruction in the body
    // whenever SpecInterface markers follow. Without it the HLS scheduler can't
    // find the top function (200-1986) and segfaults at
    // platform_agent::CoreAgent::configDelayBudget. The function-attr form
    // (`fpga.top.func` / `!fpga.function.pragma`) is NOT enough — the HLS
    // backend doesn't translate those into the call form the scheduler needs.
    // The attr/metadata forms must NOT be kept — both are stripped
    // when the SpecTopModule call is emitted (otherwise HLS sees a
    // double-declared top and gets confused).
    {
      std::string FName = F->getName().str();
      GlobalVariable *FNameStr = getOrCreateCStrGlobal(M, FName);
      unsigned FNameLen = FName.size() + 1;
      Value *FNameTyped = ConstantExpr::getBitCast(
          FNameStr, PointerType::get(ArrayType::get(IntegerType::get(Ctx, 8), FNameLen), 0));
      CallInst *TopCall = B.CreateCall(TopFn, {FNameTyped});
      // The SpecTopModule call carries !fpga.pragma.source !{!"user"}
      MDNode *PragmaSrc = MDNode::get(Ctx, {MDString::get(Ctx, "user")});
      TopCall->setMetadata("fpga.pragma.source", PragmaSrc);
      // Module-level `!axi4.slave.bundlemap = !{}` — required when s_axilite
      // kernels exist, so the HLS scheduler registers the s_axilite bundles.
      if (!M.getNamedMetadata("axi4.slave.bundlemap"))
        M.getOrInsertNamedMetadata("axi4.slave.bundlemap");
    }

    // The form the HLS scheduler reads uses op-bundles attached to
    // llvm.sideeffect — NOT SpecInterface calls. The HLS backend only
    // converts SpecInterface → xlx_s_axilite op-bundle when the cosim TB-linked
    // LLVM 16 path runs (which Rust doesn't have), so for the synthesis path
    // we must emit op-bundles directly. Without them the HLS scheduler fails to
    // register the s_axilite bundle → CoreAgent null deref at
    // configDelayBudget.
    //
    // Op-bundle shape:
    //   call void @llvm.sideeffect() #attrs [ "xlx_s_axilite"(
    //     i8* %arg-or-null,                  ; arg ptr; i8* null for "return"
    //     [N x i8] c"BUNDLE_NAME",           ; inline array, NO null term, N = strlen
    //     i64 offset,                        ; i64 -1 for "return"
    //     i32 -1, [0 x i8] zeroinit, [0 x i8] zeroinit, [0 x i8] zeroinit, i32 0
    //   ) ]
    // Call attrs: { inaccessiblememonly nounwind "xlx.port.bitwidth"="0"
    //               "xlx.source"="user" }
    Type *I64 = Type::getInt64Ty(Ctx);
    Type *I8  = Type::getInt8Ty(Ctx);
    Type *Z0  = ArrayType::get(I8, 0);  // [0 x i8]
    Constant *Z0Const = ConstantAggregateZero::get(Z0);
    FunctionType *SETy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/false);
    FunctionCallee SE = M.getOrInsertFunction("llvm.sideeffect", SETy);
    if (auto *F4 = dyn_cast<Function>(SE.getCallee())) {
      F4->addFnAttr(Attribute::NoUnwind);
      F4->addFnAttr(Attribute::InaccessibleMemOnly);
    }

    auto makeBundleArr = [&](const std::string &Bundle) -> Constant * {
      // Build an inline [N x i8] constant (NO null terminator).
      std::vector<Constant *> Bytes;
      Bytes.reserve(Bundle.size());
      for (char c : Bundle) Bytes.push_back(ConstantInt::get(I8, (uint8_t)c));
      return ConstantArray::get(ArrayType::get(I8, Bundle.size()), Bytes);
    };

    AttributeList CallAttrs = AttributeList::get(Ctx, AttributeList::FunctionIndex,
        AttrBuilder()
          .addAttribute(Attribute::NoUnwind)
          .addAttribute(Attribute::InaccessibleMemOnly)
          .addAttribute("xlx.port.bitwidth", "0")
          .addAttribute("xlx.source", "user"));

    auto emitOpBundle = [&](Value *PtrVal, const std::string &Bundle, int64_t Offset) {
      Constant *BundleArr = makeBundleArr(Bundle);
      OperandBundleDef OB("xlx_s_axilite", std::vector<Value*>{
        PtrVal,
        BundleArr,
        ConstantInt::get(I64, (uint64_t)Offset, /*signed=*/true),
        ConstantInt::get(I32, (uint64_t)-1, true),
        Z0Const, Z0Const, Z0Const,
        ConstantInt::get(I32, 0)
      });
      CallInst *Call = B.CreateCall(SE, {}, {OB});
      Call->setAttributes(CallAttrs);
    };

    for (auto &P : Ports)
      emitOpBundle(std::get<0>(P), std::get<1>(P), std::get<2>(P));
    if (HasReturn) {
      // Return port: ptr=null, offset=-1
      emitOpBundle(ConstantPointerNull::get(PointerType::get(I8, 0)),
                   ReturnBundle, (int64_t)-1);
    }
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

// Normalize ref-form s_axilite markers (`__vxx_s_axilite_port` /
// `__vxx_s_axilite_return`, emitted by `interface(&p).s_axilite()` /
// `interface(Return).s_axilite()`) into the string-form `__vxx_s_axilite`
// marker FOR FUNCTIONS THAT HAVE NO m_axi. This routes s_axilite-only kernels
// (e.g. scalar args on a user-defined-offset AXI4-Lite, no m_axi) through the
// unified injectSAxiliteAll path, which emits the full synthesis-correct form
// (SpecTopModule + !axi4.slave.bundlemap + per-port `xlx_s_axilite` op-bundles).
// The lighter injectSAxilitePortSpec path only emits SpecInterface, which the
// HLS scheduler does NOT read for synthesis → 200-1986 + segfault when no m_axi
// exists to supply SpecTopModule and the offset-register op-bundles.
// m_axi-coexisting kernels are skipped here (m_axi already supplies those), so
// they keep the per-port path unchanged.
bool normalizeSAxiliteRefToString(Module &M) {
  Function *PortM = M.getFunction("__vxx_s_axilite_port");
  Function *RetM = M.getFunction("__vxx_s_axilite_return");
  if (!PortM && !RetM) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);

  // Functions carrying an xlx_m_axi op-bundle keep the per-port path.
  SmallPtrSet<Function *, 8> MAxiFns;
  if (Function *SE = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect)) {
    for (User *U : SE->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        for (unsigned bi = 0; bi < CI->getNumOperandBundles(); ++bi)
          if (CI->getOperandBundleAt(bi).getTagName() == "xlx_m_axi")
            if (Function *F = CI->getFunction()) MAxiFns.insert(F);
  }

  FunctionType *VT = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee StrFn = M.getOrInsertFunction("__vxx_s_axilite", VT);
  GlobalVariable *RetName = getOrCreateCStrGlobal(M, "return");
  bool Changed = false;
  SmallVector<CallInst *, 16> Dead;

  if (PortM)
    for (User *U : PortM->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      Function *F = CI->getFunction();
      if (!F || MAxiFns.count(F) || CI->arg_size() < 4) continue;
      Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
      if (!Arg || Arg->getName().empty()) continue;
      GlobalVariable *NameG = getOrCreateCStrGlobal(M, Arg->getName().str());
      Value *Args[] = {NameG, ConstantInt::get(I32, Arg->getName().size()),
                       CI->getArgOperand(1), CI->getArgOperand(2),
                       CI->getArgOperand(3)};
      IRBuilder<>(CI).CreateCall(StrFn, Args);
      Dead.push_back(CI);
      Changed = true;
    }
  if (RetM)
    for (User *U : RetM->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      Function *F = CI->getFunction();
      if (!F || MAxiFns.count(F) || CI->arg_size() < 2) continue;
      Value *Args[] = {RetName, ConstantInt::get(I32, 6),
                       CI->getArgOperand(0), CI->getArgOperand(1),
                       ConstantInt::get(I64, (uint64_t)-1, true)};
      IRBuilder<>(CI).CreateCall(StrFn, Args);
      Dead.push_back(CI);
      Changed = true;
    }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  return Changed;
}

// ── phase-2 B3: interface().mode(s_axilite/...) token-chain reassembly ──
//
// The s_axilite-family interface sub-builders emit one marker per clause threaded
// by an SSA token (barista-hls src/marker.rs, `__vxx_saxi_*` / `__vxx_saxiret_*`).
// These passes follow each chain from its `_begin` marker and reassemble the exact
// LEGACY marker the old Drop-builder emitted, so the consumers below are unchanged.
// Convertible (unlike aggregate/disagg/alias) because the operands are retype-safe:
// per-port s_axilite binds a scalar/pointer control arg whose Argument identity
// survives shape retyping (resolveMarkerArg sees the SAME port Value the old Drop
// passed), and the return-control chain carries NO port operand at all.

// `interface().mode(s_axilite).port(<scalar>)[.bundle()][.offset()]` chain →
// `__vxx_s_axilite_port(port, bundle_ptr, bundle_len, offset)`. Reassembled at the
// last clause node (port operand dominance). Defaults: bundle null/0 (consumer
// falls back to "control", matching the old `&[]` dangling-ptr path), offset -1.
bool injectSAxilitePortChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_saxi_begin");
  if (!Begin) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Function *PortMk = M.getFunction("__vxx_saxi_port");
  Function *BundleMk = M.getFunction("__vxx_saxi_bundle");
  Function *OffsetMk = M.getFunction("__vxx_saxi_offset");
  FunctionType *VT =
      FunctionType::get(Type::getVoidTy(Ctx), {I8P, I8P, I32, I64}, /*Var=*/false);
  FunctionCallee PortFn = M.getOrInsertFunction("__vxx_s_axilite_port", VT);
  if (auto *F = dyn_cast<Function>(PortFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI) continue;
    ToErase.push_back(BeginCI);
    Value *Tok = BeginCI, *Port = nullptr, *BundlePtr = nullptr;
    uint64_t BundleLen = 0;
    int64_t Offset = -1;
    CallInst *InsertPt = BeginCI;
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == PortMk || Callee == BundleMk || Callee == OffsetMk) {
          Next = CI;
          break;
        }
      }
      if (!Next) break;
      Function *Callee = Next->getCalledFunction();
      if (Callee == PortMk && Next->arg_size() >= 2) {
        Port = Next->getArgOperand(1);
      } else if (Callee == BundleMk && Next->arg_size() >= 3) {
        BundlePtr = Next->getArgOperand(1);
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(2)))
          BundleLen = C->getZExtValue();
      } else if (Callee == OffsetMk && Next->arg_size() >= 2) {
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1)))
          Offset = C->getSExtValue();
      }
      InsertPt = Next;
      ToErase.push_back(Next);
      Tok = Next;
    }
    if (!Port) continue;
    IRBuilder<> B(InsertPt);
    Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
    Value *BP =
        BundlePtr ? (BundlePtr->getType() == I8P ? BundlePtr
                                                 : B.CreateBitCast(BundlePtr, I8P))
                  : (Value *)ConstantPointerNull::get(cast<PointerType>(I8P));
    CallInst *NewCI = B.CreateCall(
        PortFn, {P, BP, ConstantInt::get(I32, BundleLen),
                 ConstantInt::get(I64, (uint64_t)Offset, true)});
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    Changed = true;
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty()) Begin->eraseFromParent();
  for (Function *F : {PortMk, BundleMk, OffsetMk})
    if (F && F->use_empty()) F->eraseFromParent();
  return Changed;
}

// `interface().mode(s_axilite).port(Return)[.bundle()|.autorestart()]` chain →
// bare: `__vxx_top_s_axilite_control()` / bundle: `__vxx_s_axilite_return(name,len)`
// / autorestart: `__vxx_top_autorestart()` (autorestart wins, matching the old
// Drop). No port operand → reassemble at the begin site. MUST run before
// injectAutorestart (which consumes __vxx_top_autorestart).
bool injectSAxiliteReturnChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_saxiret_begin");
  if (!Begin) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  Function *BundleMk = M.getFunction("__vxx_saxiret_bundle");
  Function *AutoMk = M.getFunction("__vxx_saxiret_autorestart");
  FunctionType *V0 = FunctionType::get(Type::getVoidTy(Ctx), {}, /*Var=*/false);
  FunctionType *V2 =
      FunctionType::get(Type::getVoidTy(Ctx), {I8P, I32}, /*Var=*/false);
  FunctionCallee CtrlFn = M.getOrInsertFunction("__vxx_top_s_axilite_control", V0);
  FunctionCallee AutoFn = M.getOrInsertFunction("__vxx_top_autorestart", V0);
  FunctionCallee RetFn = M.getOrInsertFunction("__vxx_s_axilite_return", V2);
  for (FunctionCallee FC : {CtrlFn, AutoFn, RetFn})
    if (auto *F = dyn_cast<Function>(FC.getCallee()))
      F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI) continue;
    ToErase.push_back(BeginCI);
    Value *Tok = BeginCI, *BundlePtr = nullptr;
    uint64_t BundleLen = 0;
    bool HasBundle = false, HasAuto = false;
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == BundleMk || Callee == AutoMk) {
          Next = CI;
          break;
        }
      }
      if (!Next) break;
      Function *Callee = Next->getCalledFunction();
      if (Callee == BundleMk && Next->arg_size() >= 3) {
        HasBundle = true;
        BundlePtr = Next->getArgOperand(1);
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(2)))
          BundleLen = C->getZExtValue();
      } else if (Callee == AutoMk) {
        HasAuto = true;
      }
      ToErase.push_back(Next);
      Tok = Next;
    }
    IRBuilder<> B(BeginCI);
    CallInst *NewCI;
    if (HasAuto) {
      NewCI = B.CreateCall(AutoFn, {});
    } else if (HasBundle) {
      Value *BP =
          (BundlePtr && BundlePtr->getType() == I8P)
              ? BundlePtr
              : (BundlePtr ? B.CreateBitCast(BundlePtr, I8P)
                           : (Value *)ConstantPointerNull::get(cast<PointerType>(I8P)));
      NewCI = B.CreateCall(RetFn, {BP, ConstantInt::get(I32, BundleLen)});
    } else {
      NewCI = B.CreateCall(CtrlFn, {});
    }
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    Changed = true;
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty()) Begin->eraseFromParent();
  for (Function *F : {BundleMk, AutoMk})
    if (F && F->use_empty()) F->eraseFromParent();
  return Changed;
}

bool injectSAxiliteSideeffect(Module &M) {
  injectSAxilitePortChain(M);  // token-chain s_axilite port() builder → legacy marker
  normalizeSAxiliteRefToString(M);  // route no-m_axi ref s_axilite → unified path
  bool U = injectSAxiliteAll(M);  // unified pass — preferred
  bool A = injectSAxilitePortSpec(M);
  bool B = injectSAxiliteReturnSpec(M);
  bool C = injectSAxiliteControlSpec(M);
  (void)U;
  // injectApMemorySpec disabled — explicit ap_memory SpecInterface doesn't
  // change HLS port shape vs default inference for KPN/hls::task examples.
  // (definition removed — reference-only)
  return A || B || C;
}

// For each `__vxx_bind_storage(ptr, kind, latency)` marker, emit
// `_ssdm_op_SpecResource(ptr, 666, kind, latency, false)`, e.g.
// `SpecResource([5 x i64]* %res1_i4, 666, 29, -1, false)`
// — kind=29 = RamS2pUramEcc, 666 = "BIND_STORAGE" magic. Without this,
// HLS doesn't emit ECC flag port `ap_ecc_res1_U`.
bool injectBindStorageSpecResource(Module &M) {
  Function *Marker = M.getFunction("__vxx_bind_storage");
  if (!Marker) return false;
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecResFn = M.getOrInsertFunction("_ssdm_op_SpecResource", SpecTy);

  SmallVector<CallInst *, 8> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 3) continue;
    Dead.push_back(CI);
    Value *PtrArg = CI->getArgOperand(0);
    // Strip bitcasts to recover the underlying alloca/global/argument.
    Value *Ptr = PtrArg;
    while (auto *BC = dyn_cast<BitCastInst>(Ptr)) Ptr = BC->getOperand(0);
    while (auto *BCC = dyn_cast<BitCastOperator>(Ptr)) Ptr = BCC->getOperand(0);
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *LatC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!KindC || !LatC) continue;
    uint64_t Kind = KindC->getZExtValue();
    int64_t Lat = LatC->getSExtValue();
    // Rename bind_storage'd alloca to drop the `.i` inline suffix and
    // append `1` so HLS produces matching port names: variable `res` →
    // alloca `res1` (HLS disambiguator).
    // Rust IR: `res.i` (rustc inline suffix) → produces port `ap_ecc_res_i_U`
    // instead of expected `ap_ecc_res1_U`. Normalize to base + "1".
    if (auto *AI = dyn_cast<AllocaInst>(Ptr)) {
      StringRef Nm = AI->getName();
      std::string Base = Nm.str();
      // Strip trailing `.i`, `.i.<N>`, `_i`, `_i.<N>` style inline suffixes.
      while (true) {
        // Strip `.<digits>` first.
        size_t Dot = Base.rfind('.');
        if (Dot != std::string::npos && Dot + 1 < Base.size()) {
          bool AllDigits = true;
          for (size_t i = Dot + 1; i < Base.size(); ++i)
            if (!isdigit((unsigned char)Base[i])) { AllDigits = false; break; }
          if (AllDigits) { Base.resize(Dot); continue; }
        }
        // Strip trailing `.i` or `_i`.
        if (Base.size() >= 2 &&
            (Base[Base.size()-2] == '.' || Base[Base.size()-2] == '_') &&
            Base[Base.size()-1] == 'i') {
          Base.resize(Base.size() - 2);
          continue;
        }
        break;
      }
      // Reset to the un-suffixed source name: the Rust local now carries the
      // C++ spelling verbatim (ecc_flags `res1`), so the historical `+ "1"`
      // reconstruction (source `res` -> port name `res1`) would double the
      // digit (`res11` -> port ap_ecc_res11_U != C++ ap_ecc_res1_U).
      if (!Base.empty() && Base != Nm) {
        AI->setName(Base);
      }
    }
    IRBuilder<> B(CI);
    B.CreateCall(SpecResFn,
                 {Ptr,
                  ConstantInt::get(I64, 666),
                  ConstantInt::get(I64, Kind),
                  ConstantInt::get(I64, Lat, /*isSigned=*/true),
                  ConstantInt::get(I1, 0)});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

} } // namespace hlsrs::vxx
