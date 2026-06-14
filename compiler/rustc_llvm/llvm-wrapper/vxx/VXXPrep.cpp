//===----------------------------------------------------------------------===//
//
// VXXPrep — preprocess rustc-emitted bitcode for Vitis HLS consumption.
//
// This is a port of `vitis-narrow/src/VitisPrep.cpp` (originally an opt
// plugin running on LLVM 7) into the rustc LLVM wrapper. It is invoked
// from `rustc_codegen_llvm::back::write::codegen` whenever the compiler
// is targeting fpga32 / fpga64, so the same transformations that the
// standalone opt pass used to perform are now applied automatically as
// part of the rustc pipeline (LLVM 11, in-tree).
//
// Responsibilities (kept identical to the original VitisPrep):
//
//  * Rewrite `barista_hls::Stream<T>::read` bodies so the internal
//    `load iN, iN* %p` becomes
//    `call iN @llvm.fpga.fifo.pop.iN.p0iN(iN* %p)`.
//  * Rewrite `barista_hls::Stream<T>::write` bodies so the internal
//    `store iN %v, iN* %p` becomes
//    `call void @llvm.fpga.fifo.push.iN.p0iN(iN %v, iN* %p)`.
//  * For every top wrapper with a `__vxx_top_stream_param(idx)`
//    marker call, inject
//    `call void @llvm.sideeffect() [ "stream_interface"(T* %arg) ]`
//    at function entry so Vitis HLS recognises the parameter as a
//    FIFO port rather than an `ap_none` scalar.
//  * Apply `fpga.static.pipeline = "-1.-1"` + `!fpga.function.pragma`
//    to functions that call `__vxx_top_pipeline()`.
//  * Attach `!llvm.loop` metadata containing
//    `!"llvm.loop.unroll.count"` to the latch terminator of loops
//    that contain a `__vxx_loop_unroll(factor)` marker.
//  * Erase the marker function definitions so they do not leak into
//    the final bitcode handed to Vitis HLS.
//
// LLVM 7 → 11 porting notes:
//  * `Module::getOrInsertFunction` now returns `FunctionCallee`
//    (a {FunctionType*, Constant*} pair). We dyn_cast the underlying
//    callee to `Function*`.
//  * The pass used to be a `ModulePass` and pulled `LoopInfo` via
//    `getAnalysis<LoopInfoWrapperPass>`. Since this code now runs
//    outside of a PassManager we build `DominatorTree` + `LoopInfo`
//    on demand for the few functions we need them on.
//  * `IRBuilder::CreateCall` and `OperandBundleDef` APIs are
//    source-compatible.
//
//===----------------------------------------------------------------------===//

#include "VXXPrep.h"
#include "VXXShared.h"
#include "vxx_internal.h"  // cross-cutting helpers (hlsrs::vxx:: namespace)
#include "vxx_passes.h"    // domain-pass declarations (hlsrs::vxx:: namespace)

#include <map>
#include <set>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsFPGA.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;
// bare calls to the moved domain passes (declared in vxx_passes.h, defined
// in {maxi,loop,dsp_fft,dataflow_kpn}.cpp) resolve through this. Coexists
// harmlessly with the explicit `hlsrs::vxx::` qualifications.
using namespace hlsrs::vxx;

namespace {

// Mangle a type for a Vitis FPGA intrinsic name. Mirrors LLVM's overload
// naming: integers → `iN`, pointers → `p<AS><Pointee>`, structs → `s_<Name>`.
// Falls back to a raw-string dump for anything we don't model explicitly.

// Get or insert `llvm.fpga.fifo.{pop,push}` for arbitrary element types
// (integer or struct). For struct types we let the FIFO intrinsic carry
// the original struct type so Vitis HLS does not see an `iN`-bitcast on
// the parameter — that bitcast is what triggers the
// "type conversion operators" rejection on AGGREGATE-eligible streams.

// isAxisDisabledType moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// getOrInsertAxisPopOrPush moved to vxx_common.cpp (hlsrs::vxx:: namespace).











// fifoIntrinsicName / getOrInsertFifoPop / getOrInsertFifoPush moved to
// vxx_common.cpp (hlsrs::vxx:: namespace).













// getSsdmOp / makeSsdmStr now live in vxx_common.cpp (hlsrs::vxx:: namespace);
// declared in vxx_internal.h, no in-file forward decl needed.

// =============================================================================
// REMOVED: injectKernelInfo / injectMAxi / coalesceMAxiBundles /
// injectAutoLoopPipeline / injectLoopPipeline / injectFunctionPipeline /
// injectFunctionDataflow / injectSAxiliteControl
//
// These functions emitted Vitis HLS _ssdm_op_* intrinsic call sequences
// (SpecTopModule / SpecBitsMap / SpecPipeline / SpecLoopName / SpecLoopTripCount).
// They were replaced by emitters that produce standard llvm.sideeffect operand
// bundles — the same shape Xilinx' open-source HLS clang itself emits
// (public-API LLVM intrinsics + the documented operand bundle mechanism).
// =============================================================================
// Forward decl — defined later in the same namespace.
static void propagateAddrSpace(Value *Old, Value *New);

// Parse a Rust Itanium-mangled name and synthesise a C-style
// `<crate>.<var>` string, matching the `<fn>.<var>` shape the HLS
// back-end uses for function-local statics. Returns "" if the name
// doesn't fit the Rust mangling pattern.
//
// Rust mangling for a function-local `static mut FOO` looks like:
//   _ZN<len><crate><len><impl_path>...<len><FOO>17h<hash>E
// Each segment is `<u32 length>` followed by `<that-many bytes>`.
// The trailing `17h<16 hex chars>` is the Rust type/symbol hash.

// Demangle a Rust function symbol `_ZN<crate><fn-segments>17h<hash>E`
// down to `fn-segments` (joined by `::`). Returns "" if not a Rust
// mangle. Used to rename function-local statics by their owning fn.

// Module pass: for each internal global named `<crate>.<var>[.N]`, find
// which function uses it. If exactly one function (other than the top
// kernel) reads/writes it, rename the global to `<fn-demangle>.<var>`
// so that two functions sharing the same `var` name get distinguished
// by their owning function — like the C++ qualified global names
// `func_with_static<5>(int, int*)::acc`.

// Rewrite Rust-style internal `static [mut]` globals from
//   `<{ [N x i8] }> {init}`  →  `iN 0` (for static mut scalars) or
//                               `[K x T] {parsed init}` (for const arrays)
// so loads/stores via the bitcast chain Rust emits collapse to a plain
// `load/store iN` or `gep [K x T]`. Vitis HLS Clang emits the C-style
// scalar/array-typed global directly; this rewrite gets us closer to
// byte-equivalent IR without the user having to give up `static mut`
// ergonomics or hand-rewriting `static A: [T; N]`.

// For globals that are already StructTy with homogeneous fields and have
// per-field `xlx_bind_storage` calls, SPLIT the global into N separate
// per-field globals, so that a struct with per-field
// `#pragma HLS BIND_STORAGE` becomes distinct globals
// `@"ts.A"`, `@"ts.B"`, `@"ts.C"` rather than one struct.
//
// Without splitting, even after redirecting the whole-struct anchor for
// field 0 to a typed `i8* bitcast ([K x ElemTy]* GEP to i8*)` shape
// matching fields 1..N, the HLS back-end still associates the field-0
// pragma with the parent struct and silently falls back to "auto RAM"
// instead of the user's BRAM choice. Splitting yields the per-field
// global shape directly.
//
// Pre-existing behavior (rewriteStaticIntGlobals' reshape path handling
// at lines 1794-1842) only fires for globals that came in via the Rust
// `<{[N x i8]}>` packed-byte shape; the redirect-only fix doesn't work
// when the global was emitted directly as a `%struct.X = { [K x T], ... }`
// from rustc with #[repr(C)] semantics.
//
// Without this, Vitis HLS emits:
//   ERROR: [SYN 201-306] memory assignment of 'RAM_2P_LUTRAM':
//     conflicting assignments between 'RAM_2P_LUTRAM' and 'RAM_2P_BRAM'.
//   ERROR: [SYN 201-306] memory assignment of 'RAM_2P_URAM':
//     conflicting assignments between 'RAM_2P_URAM' and 'RAM_2P_BRAM'.
// because all 3 per-field pragmas try to apply to the same struct.

// Peephole: collapse the `alloca iN; bitcast iN* to i128*; store i128 v;
// load iN` round-trip Rust emits when narrowing a `BitInt<N>` value back
// from its internal i128 representation to the user-facing iN. The
// equivalent narrowing is a single `trunc i128 to iN`; replacing the
// load with a `trunc` exposes that shape to downstream synthesis.

// Body peephole for BitInt arithmetic: `shl X, K` followed by
// `ashr exact _, K` is the canonical "extract low (128-K) bits and
// sign-extend back" idiom that BitInt::__normalise emits after every
// arith op. When X is already known to fit in (128-K) sign-extended
// bits (typically via `sext iN to i128` where N <= 128-K), the pair
// is a no-op and we can replace it with X directly. This collapses
// the chain enough that downstream InstCombine can sometimes fold
// the wider arithmetic away.

// Body peephole: collapse the BitInt-mul normalise pattern
//   %s = shl Y, K
//   %m = mul X, %s          (or mul %s, X)
//   %r = ashr exact %m, K
// into `%r = mul X, Y`. The shl/ashr pair is BitInt::__normalise's
// post-mul re-extension that's mathematically a no-op when paired
// with a single shifted operand inside the mul.

// Body peephole: narrow `trunc i128 (mul i128 (sext iA), (sext iB)) to iC`
// to `mul iC (sext iA to iC), (sext iB to iC)` when A+B <= C (so the
// product fits without overflow). Bridges the BitInt internal-i128
// representation to native iN multiplication.

// Module pass: narrow `iN` static globals (typically i128) to i(N-K)
// when every load/store is paired with an `ashr exact (shl V, K), K`
// clamp on the same K. This rewrites `static mut ACC_RAW: i128` whose
// value is bounded to i(N-K) via the BitInt clamp into the matching
// native iM.

// Body peephole: narrow `phi i128` accumulators that are clamped to
// i(128-K) on every feedback edge via `ashr exact (shl X, K), K`. Such
// PHIs only ever carry an i(128-K)-bit signed value, so we can rewrite
// them at the narrow width. Used when the BitInt loop accumulator
// pattern (perfect_loop, pipelined_loop) keeps the wide i128 PHI even
// though the value is bounded.

// Body peephole: narrow `trunc i128 (and i128 (add/sub/mul/or/and/xor
// (sext/zext iA, sext/zext iB), MASK), iC)` where MASK = (1<<C) - 1 to
// the equivalent native iC operation. This is the BitUint<C>'s
// post-arithmetic mask-then-truncate pattern that Rust's BitUint<N>::new
// emits, rewritten at the native width.

// Body peephole: collapse the `BitFixed<W,I>` multiply pattern that Rust
// emits into the equivalent native iC computation. The pattern is:
//   %a128 = sext/zext iA → i128
//   %b128 = sext/zext iB → i128
//   %sa  = shl i128 %a128, K1
//   %sb  = shl i128 %b128, K2
//   %m   = mul nsw i128 %sa, %sb
//   %r   = ashr exact i128 %m, K3
//   %t   = trunc i128 %r to iC
// If `(A + K1) + (B + K2) - K3 <= C`, we can do the math at iC width:
//   %a' = sext/zext iA → iC
//   %b' = sext/zext iB → iC
//   %m' = mul iC %a', %b'
//   %t  = (mul scaled by 2^(K1+K2-K3) via shl, then trunc)
// This is the native-width form for `ap_fixed * ap_fixed`.

// Helper: get-or-insert a `void (...)` _ssdm_op_* intrinsic declaration.
// Every `_ssdm_op_*` decl is marked `nounwind`.
// getSsdmOp / makeSsdmStr moved to vxx_common.cpp (hlsrs::vxx:: namespace).


// For a `T (*)[M]` argument that has been narrowed to `T' (*)[M]`,
// mirror every GEP/load/store reachable from OldArg onto NewArg. Loads
// produce iN values that get sext/zext-ed back to iBig so the rest of
// the body keeps working unchanged. Stores are trunc-ed from iBig down
// to iN before writing.

// For a `iBig*` argument narrowed to `iN*`, rewrite loads/stores in
// the same shape as the array variant.

// Rewrite a single function's signature so each `__vxx_top_param`
// (and optional `__vxx_top_return`) marker maps the corresponding
// scalar / pointer / array-pointer down to its native iN width.

// Drive the narrowing for every kernel function that carries
// `__vxx_top_param` / `__vxx_top_return` markers from the
// `#[barista_hls::top]` proc-macro.

// Add the module-level metadata + flags Vitis HLS Clang sets on every
// translation unit. Idempotent.
// =============================================================================
// OSS-frontend-style attribute emitters
//
// The functions below reproduce the IR shape Xilinx's open-source HLS clang
// (Apache-2.0, github.com/Xilinx/HLS) emits when it lowers `#pragma HLS *`
// directives. Each consumes a `__vxx_*` Rust marker and stamps the
// equivalent `fpga.*` parameter / function attribute (or LLVM metadata)
// instead of the closed-source `_ssdm_op_*` intrinsic call sequence Vitis
// HLS' proprietary LTO emits AFTER frontend output. Letting Vitis HLS run
// its own attr→call lowering on our IR keeps us out of the business of
// reverse-engineering closed-source passes.
// =============================================================================

namespace {
// singleUserKernel moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// Walk a `__vxx_*` marker's calls and yield (call, parent-fn). Calls
// are returned in source order (Marker->users() is LIFO so we reverse).
SmallVector<std::pair<CallInst *, Function *>, 4>
markerCallsInSourceOrder(Module &M, StringRef MarkerName) {
  SmallVector<std::pair<CallInst *, Function *>, 4> Out;
  Function *F = M.getFunction(MarkerName);
  if (!F)
    return Out;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Out.push_back({CI, CI->getParent()->getParent()});
  // LIFO → source order.
  std::reverse(Out.begin(), Out.end());
  return Out;
}

// `resolveMarkerArg` (marker first-arg -> kernel Argument resolver) is
// called by both the moved domain passes and many passes still in this file.
// Its definition was moved to maxi.cpp ; it is declared in
// vxx_passes.h. Unqualified callers here resolve via `using namespace hlsrs::vxx;`.
// (Could not be re-namespaced in place: this sits in the outer anonymous
// namespace, which would have made it `(anonymous)::hlsrs::vxx` and ambiguous.)

// extractByteSlice moved to vxx_common.cpp (hlsrs::vxx:: namespace).
} // anonymous namespace

// `getOrCreateCStrGlobal` (a generic private-string-constant helper) is
// called by both the moved domain passes and many passes still in this file.
// Its definition was moved to maxi.cpp ; it is declared in
// vxx_passes.h. The unqualified callers that remain in this file resolve to it
// via the file-scope `using namespace hlsrs::vxx;`. (It could not be wrapped in
// `namespace hlsrs::vxx` here because this point is inside the file's outer anonymous
// namespace, which would have made it `(anonymous)::hlsrs::vxx` and ambiguous.)





// For each `__vxx_top_kernel()` marker, set `fpga.top.func=<fn name>`
// on the enclosing function. This is the same attribute the OSS HLS clang
// emits when it sees the `sdx_kernel(...)` attribute on a function.
static bool injectKernelTopAttribute(Module &M) {
  Function *Marker = M.getFunction("__vxx_top_kernel");
  if (!Marker)
    return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    Function *Top = CI->getParent()->getParent();
    // `fpga.demangled.name` is now auto-emitted by rustc_codegen_llvm
    // for every function on FPGA targets — we only need to set
    // `fpga.top.func` here to mark this as the synthesis top. The
    // attribute value is the user-visible source name; we re-use the
    // demangled.name attribute (set by rustc) instead of the IR symbol
    // (which may be Rust-mangled).
    if (!Top->hasFnAttribute("fpga.top.func")) {
      StringRef SourceName =
          Top->hasFnAttribute("fpga.demangled.name")
              ? Top->getFnAttribute("fpga.demangled.name").getValueAsString()
              : Top->getName();
      Top->addFnAttr("fpga.top.func", SourceName);
      Changed = true;
    }
    // Add `noinline` to the top kernel. Without it, the HLS back-end
    // auto-emits `SpecInterface(arg, "ap_auto")` on stream-marked args
    // because it considers the body inlineable and re-evaluates interface
    // mode (causing SYN 201-504 conflict with our SpecInterface(ap_fifo)
    // markers — observed on FFT-style placeholders).
    //
    // Skip for pure-combinational kernels (no calls to non-marker fns)
    // — noinline triggers HLS IP-XACT export to add a spurious ap_clk
    // busInterface on these (using_C++_templates off-by-one
    // ap_clk port).  rtl_as_blackbox style with helper calls keeps
    // noinline via the body-call check.
    bool BodyHasUserCallForNoInline = false;
    for (BasicBlock &BB : *Top) {
      for (Instruction &I : BB) {
        auto *CI2 = dyn_cast<CallInst>(&I);
        if (!CI2) continue;
        Function *Callee = CI2->getCalledFunction();
        if (!Callee) { BodyHasUserCallForNoInline = true; break; }
        StringRef N = Callee->getName();
        if (N.startswith("_ssdm_") || N.startswith("llvm.") ||
            N.startswith("__vxx_") || N.startswith("__vxxprep_"))
          continue;
        BodyHasUserCallForNoInline = true;
        break;
      }
      if (BodyHasUserCallForNoInline) break;
    }
    if (BodyHasUserCallForNoInline && !Top->hasFnAttribute(Attribute::NoInline)) {
      Top->addFnAttr(Attribute::NoInline);
      Changed = true;
    }
    // Also attach `!fpga.function.pragma = !{!{"fpga.top", "user", null}}`
    // metadata, expected on every #pragma HLS-tagged function; the third
    // tuple element is a DILocation pointing at the pragma source. We don't
    // carry debug info, so use null.
    // Vitis HLS' AXIS RTL handshake checks for this metadata in addition
    // to the `fpga.top.func` attribute when generating cosim test
    // infrastructure for `hls::stream<T>` parameters.
    {
      LLVMContext &Ctx = Top->getContext();
      MDNode *Existing = Top->getMetadata("fpga.function.pragma");
      Metadata *Inner[] = {
          MDString::get(Ctx, "fpga.top"),
          MDString::get(Ctx, "user"),
          nullptr,
      };
      MDNode *InnerNode = MDNode::get(Ctx, Inner);
      // Append fpga.top to any existing metadata list (e.g. set by
      // injectDataflow which adds fpga.dataflow.func). Skip if fpga.top
      // is already present.
      SmallVector<Metadata *, 4> Ops;
      bool AlreadyHas = false;
      if (Existing) {
        for (auto &Op : Existing->operands()) {
          Ops.push_back(Op.get());
          if (auto *MD = dyn_cast_or_null<MDNode>(Op.get()))
            if (MD->getNumOperands() >= 1)
              if (auto *S = dyn_cast<MDString>(MD->getOperand(0).get()))
                if (S->getString() == "fpga.top")
                  AlreadyHas = true;
        }
      }
      if (!AlreadyHas) {
        Ops.push_back(InnerNode);
        Top->setMetadata("fpga.function.pragma", MDNode::get(Ctx, Ops));
        Changed = true;
      }
    }
    Dead.push_back(CI);
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  return Changed;
}


// Inline calls to libcore Iterator/Range trait methods that rustc leaves
// out-of-line at -O1. For `<I as IntoIterator>::into_iter` on `Range<T>`,
// the body is just `self` — i.e. construct the same struct from its args.
// libcore's bc cannot be linked in (uses LLVM 11-only `freeze`), so we
// recognise the call signature and synthesise the equivalent struct directly.
// Simplify inlined libcore iterator loops by running LLVM's standard
// JumpThreading + SimplifyCFG passes on kernel functions. In particular,
// the inlined `StepBy::next` body has a 2-PHI loop header (iter +
// first_take flag) where first_take is constant 1 on first iter, then
// constant 0 on subsequent iters. JumpThreading peels the first iter
// (where flag=1), and SimplifyCFG collapses the dead first_take branch
// in the remaining loop, leaving a clean single-path countable loop —
// required for HLS dual-port BRAM write inference.
//
// Conservative: only runs on functions with `fpga.top.func` attribute
// (kernel tops). Excludes InstCombine which renames SSA values and
// triggers LLVM 7 parser errors downstream.
// Fold `extractvalue` through `insertvalue`/`phi`/constant aggregates. At -O0
// LLVM7's instcombine does NOT push extractvalue through an aggregate phi, so
// the synthesized Range::next's `{disc,value}` Option struct survives as
// extractvalue(phi{struct}) — leaving the for-loop's exit as a two-step
// (icmp slt -> some/none -> disc-switch) that Vitis reads as a *variable* trip
// count (HLS 214-187), blocking window unroll + stencil recognition (214-332).
// Folding the disc/value to scalars lets the subsequent SimplifyCFG/JumpThreading
// in simplifyKernelLoops collapse the loop to a canonical counted form with a
// constant trip count. No-op at -O1 (instcombine already did this).
// Promote allocas to SSA (SROA + mem2reg) on defined functions. At -O0 rustc
// keeps locals (incl. the synthesized Range struct) in allocas; promoting them
// BEFORE foldExtractValue exposes the Option {disc,value} as select/phi of
// structs that foldExtractValue can then collapse. No-op at -O1 (already SSA).
// Delete `llvm.memcpy` whose SOURCE is an alloca that is never stored to (only
// lifetime/gep/memcpy users) — i.e. it copies uninitialized memory. rustc's
// `uninit_array()` (used to avoid a zero-fill memset) lowers to a by-value move
// = memcpy of undef into the local, which the HLS back-end expands into a
// residual load-store-loop in the TOP function (using array-typed m_axi). That
// residual loop perturbs the synthesized module hierarchy so the flattened
// stencil loop is not synthesized as a sub-module -> getEstimatedSchedLatency
// null SIGSEGV. Copying undef is dead (the dest is fully written afterwards),
// so deleting it is safe. Iterates so chained uninit copies (A->B->C) all drop.



// Mark counted-loop induction-variable increments as `nsw`/`nuw`. rustc at -O0
// emits the IV step (`%iv.next = add i64 %iv, 1`) WITHOUT no-wrap flags. The
// flags matter downstream: the HLS back-end linearizes a 2-D `[W x i8]` array
// access into `gmem + (y*W + x)` and then runs an m_axi burst-inference (SCEV)
// pass. Without no-wrap on the IV the SCEV can't prove the linearized address
// is monotonic/in-range, so it leaves the access as a per-element
// `ReadReq(.,1)` instead of coalescing it into the whole-array `ReadReq(.,W*H)`
// burst. A per-element m_axi inside a pipelined merged loop can't be scheduled
// as a sub-module → null PerformanceInfo → getEstimatedSchedLatency SIGSEGV
// (using_array_stencil_2d -O0 crash). Marking the canonical counted-loop step
// (`add iv, +C` where `iv` is a header PHI whose only backedge value is this
// add) is provably no-wrap for a `0..N` loop with N < 2^63, so this adds the
// hint without changing semantics.

// rustc lowers `for i in 0..N` into a *saturating* induction step:
//   %next = add nuw i64 %iv, (zext (icmp ult %iv, N))
// the IV increments by 1 while in range and 0 at the boundary (Range's peekable
// look-ahead carries a second PHI offset by one). The data-dependent
// `zext(icmp)` step defeats SCEV — IndVarSimplify can't form a canonical IV, so
// the 2-D `gmem + y*W + x` stencil access stays non-affine and the m_axi
// burst-inference declines (per-element ReadReq → using_array_stencil_2d
// schedule SIGSEGV). A canonical IV yields a whole-array burst instead.
//
// Safety: the step's `icmp` is the SAME SSA value as the latch branch condition,
// so on every iteration that actually takes the backedge it is `true` ⇒ the step
// is +1 for every *used* next-value; only the dead exit-iteration result differs
// (and it is never read). Rewriting the `zext(icmp)` step operand to constant 1
// is therefore semantics-preserving, and it lets the downstream IndVarSimplify
// collapse the look-ahead dual-PHI into a canonical counted loop so the burst
// is inferred.

// rustc's `0..N` Range lowers to a *look-ahead dual-counter*: a header holds two
// coupled PHIs —
//   %la   = phi [init+1, ph], [%la.next, latch]   ; look-ahead, e.g. 1,2,..,N
//   %iter = phi [init,   ph], [%la,     latch]     ; current,    e.g. 0,1,..,N-1
//   %la.next = add %la, 1     (after canonicalizeRangeLoopStep)
// `%iter`'s backedge value is the *other* PHI (`%la`), not `add %iter, 1`, so
// IndVarSimplify doesn't recognise it as a canonical IV and keeps BOTH PHIs (in
// i64). The desired form is ONE i32 IV per loop. The extra coupled loop-carried
// PHI per loop is a leading suspect for why the HLS back-end declines to
// flatten/merge the y+x nest (it outlines the x-loop alone, leaking ~600 regs →
// SIGSEGV).
//
// Decouple: rewire `%iter`'s backedge to a self-contained `add %iter, 1`
// (numerically identical: %iter still steps 0,1,2,…). `%la` is left for the exit
// test; the now-clean `%iter` ({init,+,1}) lets the downstream IndVarSimplify
// express the exit in terms of `%iter` and delete the redundant look-ahead PHI.



// Transform each canonical counted loop  PH→H{phi;body}→L{iv.next=add iv,1;
// ec=icmp ne iv.next,N; br ec,H,E}  into:
//   PH→Cond{phi; ec=icmp ult iv,N; br ec,H,E}; H{body}; L{iv.next; br Cond}
// Gated to stencil-marked functions for now.


// Lower `llvm.{u,s}{add,sub,mul}.with.overflow.iN` intrinsics to explicit
// IR (add + icmp). Vitis HLS LLVM 7 rejects these intrinsics as
// unsupported. They're emitted by libcore for checked arithmetic — in
// particular `usize::checked_add` inside `StepBy::next` / Range iteration.
// Faithful lowering preserves the overflow flag; downstream DCE prunes if
// unused.



// Rewrite kernel-arg `[N x intT]*` (Rust `&mut [Stream<intT>; N]`,
// flattened by repr(transparent)) to the C-decayed
// `class.hls::stream<intT>* "fpga.decayed.dim.hint"="N"` form used for
// `hls::stream<intT> arr[N]` parameters. Without
// this, the cosim TB generator sees `[N x intT]*` (raw memory ptr) and the wrapper
// doesn't match the C++ TB's `class.hls::stream<intT>*` call shape —
// stream interface reads uninit memory ("hls::stream is read while
// empty").
//
// Detection: kernel arg is `[N x intT]*` AND has at least one GEP user
// of shape `getelementptr [N x intT], [N x intT]* %arg, i64 0, i64 idx`
// whose result is operand to `llvm.fpga.fifo.{pop,push}` (= a stream
// access). Other `[N x intT]*` uses (m_axi etc.) don't trigger.
//
// Rewrite:
//   - Type: `[N x intT]*` → `class.hls::stream<intT>*`
//     (named struct `{ intT }` reused if it already exists)
//   - Arg attr: `fpga.decayed.dim.hint="N"`
//   - Body GEPs: `[N x intT], i64 0, i64 j` → `class.hls::stream<intT>,
//     i64 j, i32 0` (both yield `intT*`)
//   - Add `llvm.sideeffect "stream_interface"(stream*)` at fn entry
//   - Other uses (rare for stream args) get a bitcast back to
//     `[N x intT]*` so downstream injectors see the original shape.
static bool renameArrayOfStreamArgs(Module &M) {
  LLVMContext &Ctx = M.getContext();

  struct Cand {
    Function *F;
    unsigned ArgIdx;
    ArrayType *OldArrTy;     // [N x intT]
    IntegerType *ElemTy;     // intT
    uint64_t N;
  };
  SmallVector<Cand, 4> Cands;

  // Snapshot `__vxx_ap_fifo` marker -> Argument mapping; ap_fifo-markered
  // args (Rust `&[T;N] + barista_hls::ap_fifo(arr,...)` pattern) are
  // semantically streams too, even when their body uses plain array
  // load/store. These need `class.hls::stream<T>*` +
  // `stream_interface` bundle; the cosim TB generator
  // expects that shape to create the `sim/tv/stream_size/` +
  // `sim/tv/directio_size/` TV subdirs at cosim_design setup time.
  SmallPtrSet<Argument *, 8> ApFifoArgs;
  // Scan `llvm.sideeffect [ "xlx_ap_fifo"(arg, ...) ]` op-bundles. ONLY
  // promote ap_fifo+array args to stream type IF the same function also
  // contains a `directio_interface` bundle — i.e. mixed directio+ap_fifo
  // kernels where cosim_design needs to see a streaming context to
  // create the `sim/tv/directio_size/` TV subdir. Pure ap_fifo+array
  // kernels (simple_data_driven etc.) work fine via the A2Stream cosim TB
  // path and would regress under unconditional rename (simple_data_driven
  // cosim breaks when this trigger is unconditional).
  DenseSet<Function *> DirectioFns;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee || Callee->getIntrinsicID() != Intrinsic::sideeffect)
          continue;
        for (unsigned i = 0, e = CI->getNumOperandBundles(); i < e; ++i) {
          if (CI->getOperandBundleAt(i).getTagName() == "directio_interface") {
            DirectioFns.insert(&F);
            break;
          }
        }
      }
    }
  }
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!DirectioFns.count(&F)) continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee || Callee->getIntrinsicID() != Intrinsic::sideeffect)
          continue;
        for (unsigned i = 0, e = CI->getNumOperandBundles(); i < e; ++i) {
          OperandBundleUse OB = CI->getOperandBundleAt(i);
          if (OB.getTagName() != "xlx_ap_fifo") continue;
          if (OB.Inputs.empty()) continue;
          Value *V = OB.Inputs[0].get();
          while (true) {
            if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
            if (auto *AS = dyn_cast<AddrSpaceCastInst>(V)) { V = AS->getOperand(0); continue; }
            if (auto *BCO = dyn_cast<BitCastOperator>(V)) { V = BCO->getOperand(0); continue; }
            break;
          }
          if (auto *A = dyn_cast<Argument>(V)) ApFifoArgs.insert(A);
        }
      }
    }
  }
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("fpga.top.func")) continue;
    for (Argument &A : F.args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT) continue;
      auto *AT = dyn_cast<ArrayType>(PT->getElementType());
      if (!AT) continue;
      auto *ElemT = dyn_cast<IntegerType>(AT->getElementType());
      if (!ElemT) continue;
      // Verify at least one GEP user that feeds a fifo intrinsic, OR
      // the arg carries an `__vxx_ap_fifo` marker.
      bool MatchedFifo = ApFifoArgs.count(&A) > 0;
      for (User *U : A.users()) {
        if (MatchedFifo) break;
        auto *GEP = dyn_cast<GetElementPtrInst>(U);
        if (!GEP) continue;
        if (GEP->getNumOperands() != 3) continue;
        auto *Idx0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
        if (!Idx0 || !Idx0->isZero()) continue;
        for (User *GU : GEP->users()) {
          auto *CI = dyn_cast<CallInst>(GU);
          if (!CI) continue;
          Function *Callee = CI->getCalledFunction();
          if (!Callee) continue;
          if (Callee->getName().startswith("llvm.fpga.fifo.pop") ||
              Callee->getName().startswith("llvm.fpga.fifo.push")) {
            MatchedFifo = true;
            break;
          }
        }
        if (MatchedFifo) break;
      }
      if (!MatchedFifo) continue;
      Cands.push_back({&F, A.getArgNo(), AT, ElemT, AT->getNumElements()});
    }
  }
  if (Cands.empty()) return false;

  // Group by function so we can recreate each fn once with all its
  // array-of-stream args rewritten together.
  DenseMap<Function *, SmallVector<Cand, 4>> ByFn;
  for (auto &C : Cands) ByFn[C.F].push_back(C);

  bool Changed = false;
  for (auto &KV : ByFn) {
    Function *F = KV.first;
    auto &Group = KV.second;
    DenseMap<unsigned, Cand *> ByIdx;
    for (auto &C : Group) ByIdx[C.ArgIdx] = &C;

    FunctionType *OldFT = F->getFunctionType();
    unsigned NParams = OldFT->getNumParams();
    SmallVector<Type *, 8> NewParamTys;
    SmallVector<StructType *, 4> StreamTys(NParams, nullptr);
    for (unsigned i = 0; i < NParams; ++i) {
      auto It = ByIdx.find(i);
      if (It == ByIdx.end()) {
        NewParamTys.push_back(OldFT->getParamType(i));
        continue;
      }
      Cand *C = It->second;
      // Build the integer-typed name. C++ uses "int" for i32; for other
      // widths approximate the C type so the cosim TB generator's name matching
      // recognises the wrapper.
      std::string IntName;
      switch (C->ElemTy->getBitWidth()) {
        case 8:  IntName = "char"; break;
        case 16: IntName = "short"; break;
        case 32: IntName = "int"; break;
        case 64: IntName = "long"; break;
        default: IntName = ("i" + std::to_string(C->ElemTy->getBitWidth())); break;
      }
      std::string StreamName = "class.hls::stream<" + IntName + ">";
      StructType *ST = M.getTypeByName(StreamName);
      if (!ST)
        ST = StructType::create(Ctx, {C->ElemTy}, StreamName, false);
      StreamTys[i] = ST;
      NewParamTys.push_back(PointerType::get(ST, 0));
    }

    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParamTys, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       F->getName() + ".aos_tmp",
                                       F->getParent());
    NewF->copyAttributesFrom(F);
    // Clear param attrs on rewritten args — copyAttributesFrom carried
    // over `align/dereferenceable` sized for the OLD `[N x intT]*`
    // pointee, which would mismatch the new `class.hls::stream<intT>*`
    // and trigger HLS back-end internal errors.
    {
      AttributeList AL = NewF->getAttributes();
      for (auto &KV2 : ByIdx) {
        unsigned i = KV2.first;
        AL = AL.removeParamAttributes(Ctx, i);
      }
      NewF->setAttributes(AL);
    }
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());

    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);

    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      // Carry over arg attrs for unchanged args; for rewritten ones
      // drop alignment/dereferenceable (would mismatch new pointee
      // size) and stamp `fpga.decayed.dim.hint`.
      if (StreamTys[i] == nullptr) {
        AttributeSet AS = F->getAttributes().getParamAttributes(i);
        for (Attribute A : AS) NewA->addAttr(A);
      } else {
        // Strip `align`/`dereferenceable`/`nocapture` etc. on rewritten
        // args — those were sized for the old `[N x intT]*` pointee
        // and are wrong for `class.hls::stream<intT>*`. Add only the
        // dim.hint that lets cosim TB compute the buffer size.
        NewA->addAttr(Attribute::get(Ctx, "fpga.decayed.dim.hint",
                                      std::to_string(ByIdx[i]->N)));
      }
    }

    // Bridge old uses to new args.
    BasicBlock &Entry = NewF->getEntryBlock();
    Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
    IRBuilder<> B(InsertBefore);
    Type *I32Ty = Type::getInt32Ty(Ctx);

    OldA = F->arg_begin();
    NewA = NewF->arg_begin();
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      if (StreamTys[i] == nullptr) {
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Inject `stream_interface(stream*)` sideeffect.
      Function *Sideeffect =
          Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
      Value *SArgs[] = {&*NewA};
      OperandBundleDef SBundle("stream_interface",
          SmallVector<Value *, 1>(std::begin(SArgs), std::end(SArgs)));
      auto *SCall = B.CreateCall(Sideeffect, ArrayRef<Value *>(), {SBundle});
      SCall->setOnlyAccessesInaccessibleMemory();
      SCall->setDoesNotThrow();

      // Rewrite GEP uses: gep [N x T], 0, j → gep stream*, j, 0.
      // Other uses get a bitcast fallback to the old type.
      Cand *C = ByIdx[i];
      Value *BitcastFallback = nullptr;
      auto getBitcast = [&]() -> Value * {
        if (!BitcastFallback)
          BitcastFallback =
              B.CreateBitCast(&*NewA, OldA->getType(),
                               NewA->getName() + ".aos_bc");
        return BitcastFallback;
      };
      SmallVector<Use *, 8> Uses;
      for (Use &U : OldA->uses()) Uses.push_back(&U);
      for (Use *U : Uses) {
        User *Usr = U->getUser();
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
          if (GEP->getPointerOperand() == &*OldA && GEP->getNumOperands() == 3) {
            auto *Idx0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
            if (Idx0 && Idx0->isZero()) {
              IRBuilder<> GB(GEP);
              Value *Idxs[] = {
                  GEP->getOperand(2),
                  ConstantInt::get(I32Ty, 0),
              };
              auto *NewGEP = GB.CreateInBoundsGEP(
                  StreamTys[i], &*NewA, Idxs, GEP->getName());
              GEP->replaceAllUsesWith(NewGEP);
              GEP->eraseFromParent();
              continue;
            }
          }
        }
        // Fallback for non-(0,j) GEPs and other users.
        U->set(getBitcast());
      }
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: rewrote array-of-stream kernel arg(s)\n";
  return Changed;
}





// Rewrite kernel-arg `T*` (Rust `&mut ApNone<T>` etc., flattened by
// repr(transparent) over T) to `class.hls::directio<T, mode>*` named
// struct pointer. mode is 0=hs,
// 1=vld, 2=ack, 3=none. Without this rename, the cosim TB generator sees raw `T*`
// and the wrapper buffer doesn't match the C++ TB's
// `class.hls::directio<...>*` allocator → uninit memory access.
//
// Detection: walk `__vxx_top_directio_param(idx, mode)` markers in
// the kernel body. For each, retype the arg-idx kernel param.
//
// Body rewrite is minimal: the underlying volatile load/store on the
// raw `T*` keeps working since C++'s directio class is a single-field
// `{ T }`. We only need to GEP through field 0 at access sites — but
// because the raw `T*` is wrapped only at the arg-decl level, the
// existing accesses use `T*` directly. We bridge via a single
// bitcast at function entry: `class.hls::directio<T,mode>*` → `T*`.
// (The HLS back-end accepts directio bitcasts in this direction.)
static bool renameDirectioArgs(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Function *Marker = M.getFunction("__vxx_top_directio_param");
  if (!Marker) return false;

  struct Cand {
    Function *F;
    unsigned ArgIdx;
    unsigned Mode;
    Type *ElemTy;
  };
  SmallVector<Cand, 4> Cands;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    MarkerCalls.push_back(CI);
    Function *F = CI->getParent()->getParent();
    auto *IdxC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!IdxC || !ModeC) continue;
    unsigned ArgIdx = (unsigned)IdxC->getZExtValue();
    unsigned Mode = (unsigned)ModeC->getZExtValue();
    if (ArgIdx >= F->arg_size()) continue;
    Argument *A = F->getArg(ArgIdx);
    auto *PT = dyn_cast<PointerType>(A->getType());
    if (!PT) continue;
    Cands.push_back({F, ArgIdx, Mode, PT->getElementType()});
  }
  // Erase markers regardless of whether retype succeeds.
  for (CallInst *CI : MarkerCalls) CI->eraseFromParent();

  if (Cands.empty()) return false;

  bool Changed = false;

  // DESIGN top: the C++ pre-reflow keeps a typed directio arg as a PLAIN
  // scalar — `i32* %din_A` + `SpecInterface(ptr, "ap_none", .., 2)` +
  // `_ssdm_op_Read.ap_none.volatile` (measured on
  // using_directio_none_in_tasks' a.pp.bc; the stream/directio class form
  // never appears). Route those args through the proven pragma path by
  // synthesizing the `__vxx_ap_scalar(arg, mode, 0)` marker the
  // interface() builder would have emitted (renameApScalarArgsToSsdm +
  // injectApScalar run right after this pass). Mode map: directio_param
  // {hs=0, vld=1, ack=2, none=3} → ap_scalar {hs=4, vld=2, ack=1, none=0}.
  // The retype below stays for the cpp_proxy build's rust_<top> adapter,
  // whose cosim TB wrapper needs the class-typed args. Design tops are
  // recognized by their `__vxx_top_kernel` marker (`fpga.top.func` is only
  // stamped later, by injectKernelTopAttribute); the cpp_proxy adapter
  // carries the directio markers but never `__vxx_top_kernel`.
  {
    SmallPtrSet<Function *, 4> DesignTops;
    if (Function *TopMk = M.getFunction("__vxx_top_kernel"))
      for (User *U : TopMk->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          DesignTops.insert(CI->getFunction());
    SmallVector<Cand, 4> DesignCands;
    Cands.erase(std::remove_if(Cands.begin(), Cands.end(),
                               [&](const Cand &C) {
                                 if (!DesignTops.count(C.F))
                                   return false;
                                 DesignCands.push_back(C);
                                 return true;
                               }),
                Cands.end());
    if (!DesignCands.empty()) {
      Type *VoidTy = Type::getVoidTy(Ctx);
      Type *I32Ty = Type::getInt32Ty(Ctx);
      // Reuse the module's existing marker declaration when present — the
      // Rust-side extern types the pointer param `{}*`, and a mismatched
      // getOrInsertFunction would wrap our calls in a constant bitcast that
      // renameApScalarArgsToSsdm / injectApScalar (which walk the function's
      // direct CallInst users) would silently miss.
      Function *ApScalarF = M.getFunction("__vxx_ap_scalar");
      FunctionType *ApScalarTy =
          ApScalarF ? ApScalarF->getFunctionType()
                    : FunctionType::get(
                          VoidTy,
                          {Type::getInt8PtrTy(Ctx), I32Ty, I32Ty},
                          /*Var=*/false);
      FunctionCallee ApScalar =
          ApScalarF ? FunctionCallee(ApScalarTy, ApScalarF)
                    : M.getOrInsertFunction("__vxx_ap_scalar", ApScalarTy);
      static const unsigned ModeMap[4] = {/*hs*/ 4, /*vld*/ 2, /*ack*/ 1,
                                          /*none*/ 0};
      for (auto &C : DesignCands) {
        if (C.Mode > 3) continue;
        Argument *A = C.F->getArg(C.ArgIdx);
        IRBuilder<> B(&C.F->getEntryBlock(),
                      C.F->getEntryBlock().getFirstInsertionPt());
        Value *Ptr = B.CreateBitCast(A, ApScalarTy->getParamType(0));
        B.CreateCall(ApScalar, {Ptr, ConstantInt::get(I32Ty, ModeMap[C.Mode]),
                                ConstantInt::get(I32Ty, 0)});
        Changed = true;
      }
    }
    if (Cands.empty()) return Changed;
  }

  DenseMap<Function *, SmallVector<Cand, 4>> ByFn;
  for (auto &C : Cands) ByFn[C.F].push_back(C);

  for (auto &KV : ByFn) {
    Function *F = KV.first;
    auto &Group = KV.second;
    DenseMap<unsigned, Cand *> ByIdx;
    for (auto &C : Group) ByIdx[C.ArgIdx] = &C;

    FunctionType *OldFT = F->getFunctionType();
    unsigned NParams = OldFT->getNumParams();
    SmallVector<Type *, 8> NewParamTys;
    SmallVector<StructType *, 4> DioTys(NParams, nullptr);
    for (unsigned i = 0; i < NParams; ++i) {
      auto It = ByIdx.find(i);
      if (It == ByIdx.end()) {
        NewParamTys.push_back(OldFT->getParamType(i));
        continue;
      }
      Cand *C = It->second;
      // Build C++ type name. For scalars use the C type name.
      std::string TName;
      if (auto *IT = dyn_cast<IntegerType>(C->ElemTy)) {
        switch (IT->getBitWidth()) {
          case 8:  TName = "char"; break;
          case 16: TName = "short"; break;
          case 32: TName = "int"; break;
          case 64: TName = "long"; break;
          default: TName = ("i" + std::to_string(IT->getBitWidth())); break;
        }
      } else if (C->ElemTy->isFloatTy()) {
        TName = "float";
      } else if (C->ElemTy->isDoubleTy()) {
        TName = "double";
      } else {
        // Unsupported elem type — skip this arg.
        NewParamTys.push_back(OldFT->getParamType(i));
        continue;
      }
      // Rename target depends on directio mode:
      //   mode 3 (ap_none): use `class.hls::stream<T>*` for this case. The
      //     cosim TB generator then emits KpnDirectIO wrapper for these args
      //     (not Register).
      //   modes 0/1/2 (ap_hs/ap_vld/ap_ack): retain
      //     `struct.ssdm_int<W, signed>*` shape for ap_hs/vld/ack handshakes.
      std::string DioName;
      if (C->Mode == 3) {
        DioName = "class.hls::stream<" + TName + ">";
      } else if (auto *IT2 = dyn_cast<IntegerType>(C->ElemTy)) {
        DioName = "struct.ssdm_int<" + std::to_string(IT2->getBitWidth()) +
                  ", false>";
      } else {
        // Float/double scalars: fall back to old directio form.
        DioName = "class.hls::directio<" + TName + ", " +
                  std::to_string(C->Mode) + ">";
      }
      StructType *ST = M.getTypeByName(DioName);
      if (!ST)
        ST = StructType::create(Ctx, {C->ElemTy}, DioName, false);
      DioTys[i] = ST;
      NewParamTys.push_back(PointerType::get(ST, 0));
    }

    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParamTys, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       F->getName() + ".dio_tmp",
                                       F->getParent());
    NewF->copyAttributesFrom(F);
    {
      AttributeList AL = NewF->getAttributes();
      for (auto &KV2 : ByIdx)
        AL = AL.removeParamAttributes(Ctx, KV2.first);
      NewF->setAttributes(AL);
    }
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);

    BasicBlock &Entry = NewF->getEntryBlock();
    Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
    IRBuilder<> B(InsertBefore);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);

    // For mode 3 (ap_none) directio args we emit
    //   call void @llvm.sideeffect() [ "directio_interface"(<ptr>) ]
    // at function entry, for `hls::ap_none<T>&` kernel args. The HLS back-end
    // uses this bundle to generate `directiocpy_hls` wrapper attrs → the cosim
    // TB generator emits KpnDirectIO instead of Register.
    Function *Sideeffect =
        Intrinsic::getDeclaration(NewF->getParent(), Intrinsic::sideeffect);

    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    SmallVector<Argument *, 4> DirectioArgs;
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (DioTys[i] == nullptr) {
        AttributeSet AS = F->getAttributes().getParamAttributes(i);
        for (Attribute A : AS) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Bridge: GEP through directio struct's field 0 to get T*.
      Value *Idxs[] = {
          ConstantInt::get(I64Ty, 0),
          ConstantInt::get(I32Ty, 0),
      };
      Value *Bridge = B.CreateInBoundsGEP(DioTys[i], &*NewA, Idxs,
                                           NewA->getName() + ".inner");
      OldA->replaceAllUsesWith(Bridge);
      // Track mode-3 args for directio_interface bundle emission.
      Cand *C = ByIdx[i];
      if (C && C->Mode == 3)
        DirectioArgs.push_back(&*NewA);
    }
    for (Argument *DA : DirectioArgs) {
      OperandBundleDef DB("directio_interface", ArrayRef<Value *>(DA));
      B.CreateCall(Sideeffect, ArrayRef<Value *>(), {DB});
    }
    // If we have any mode-3 (ap_none) directio args, add module-level
    // `!has_MT_tasks = !{}` metadata, expected for
    // kernels using hls::task and hls::ap_none<T> together; vitis_hls
    // cosim_design uses this flag to decide whether to create the
    // `sim/tv/directio_size/` dir at runtime. Without it, cosim.tv.exe
    // tries to open files in a non-existent dir and aborts with
    // "Error on TV file".
    if (!DirectioArgs.empty()) {
      Module *Mod = NewF->getParent();
      if (!Mod->getNamedMetadata("has_MT_tasks")) {
        NamedMDNode *NMD = Mod->getOrInsertNamedMetadata("has_MT_tasks");
        (void)NMD;
      }
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: renamed directio kernel arg(s)\n";
  return Changed;
}

// Rename kernel scalar args carrying `__vxx_ap_scalar(arr, mode_idx, 0)`
// markers (from `barista_hls::ap_hs/ap_none/ap_vld/ap_ack/ap_stable(&i32)`)
// from plain `i32*` to the `struct.ssdm_int<W, false>*` form. Without this,
// the cosim TB's wrapper generator
// sees raw `i32*` and applies the wrong handshake count, causing RTL
// sim to hang at 0/N transactions (cosim deadlock).
//
// Runs BEFORE injectApScalar so the subsequent SpecInterface call uses
// the renamed (bridged) arg. Bridges via GEP through field 0 at fn entry.
static bool renameApScalarArgsToSsdm(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Function *Marker = M.getFunction("__vxx_ap_scalar");
  if (!Marker) return false;

  // Legacy gate: SpecDataflowPipeline call with kind=1. This is currently
  // unreachable (kind is never emitted as 1, and at Phase 3 the call doesn't
  // exist yet — injectDataflow runs in Phase 5), but kept for safety in case
  // a future pass emits it pre-Phase-3.
  SmallPtrSet<Function *, 4> KpnFns;
  if (Function *Dfp = M.getFunction("_ssdm_op_SpecDataflowPipeline")) {
    for (User *U : Dfp->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() < 2) continue;
      auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (!KindC || KindC->getZExtValue() != 1) continue;
      KpnFns.insert(CI->getParent()->getParent());
    }
  }
  // Dataflow context detection via still-live `__vxx_dataflow` /
  // `__vxx_dataflow_kpn` markers (the proc-macro emits these into the top fn;
  // injectDataflow consumes them in Phase 5). Used ONLY to enable the ap_hs
  // (mode 4) directio retype below: an ap_hs scalar read N times inside a
  // dataflow region is the hls::task `ap_hs<int>` pattern
  // (using_directio_hs_in_tasks). cosim TB must feed N handshake transactions on
  // that port, which requires the `class.hls::directio<T,0>*` type. Without it
  // cosim TB emits a single onebyonecpy → the kernel's 2nd ap_hs_read blocks on
  // n_ap_vld forever → cosim hang.
  //
  // NOTE: deliberately NOT used to gate modes 0/2/3/5. Those (e.g. ap_none in
  // using_directio_none_in_tasks, which is also dataflow_kpn) already work
  // with plain i32* args; widening the retype to them would change their
  // cosim TB shape and break them. Only ap_hs needs this.
  SmallPtrSet<Function *, 4> DataflowFns;
  for (const char *Mk : {"__vxx_dataflow", "__vxx_dataflow_kpn"}) {
    if (Function *Df = M.getFunction(Mk)) {
      for (User *U : Df->users()) {
        auto *CI = dyn_cast<CallInst>(U);
        if (!CI) continue;
        if (Function *F = CI->getFunction()) DataflowFns.insert(F);
      }
    }
  }

  // `__vxx_aphs_stream`: route a LOOPED ap_hs (mode 4) scalar through the retype
  // below even when it's directly written (output ap_hs) and/or outside a
  // dataflow region (a plain PIPELINE loop). Needed for the cpp_proxy cosim of
  // Task_level_Parallelism/Control_driven/directio/ap_hs (reset_value output /
  // reset_myCounter input, read/written N times in one loop). Under the wrap the
  // scalar xlx_ap_hs path collapses them to a 1-value ap_vld/ap_none port, so N
  // beats can't flow. INPUT ap_hs is retyped to `class.hls::stream<int>*` (apatb
  // resolves stream direction correctly through the wrap — the directio wrapc
  // codegen mishandles an INPUT directio and references an undefined
  // WRAPC_DIRECTIO_SIZE_OUT, HLS 212-317); OUTPUT ap_hs stays
  // `class.hls::directio<int,0>*` (SIZE_OUT is defined, so it compiles).
  const bool ApHsStream = hlsrs::vxx::markerUsed(M, "__vxx_aphs_stream");
  struct Cand {
    Function *F;
    unsigned ArgIdx;
    Type *ElemTy;
    unsigned Mode;  // 0=ap_none, 1=ap_ack, 2=ap_vld, 3=ap_ovld, 4=ap_hs, 5=ap_stable
    bool IsOutput;  // arg is written (store) — an output ap_hs
  };
  SmallVector<Cand, 4> Cands;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Argument *Arg = resolveMarkerArg(CI->getArgOperand(0));
    if (!Arg) continue;
    auto *PT = dyn_cast<PointerType>(Arg->getType());
    if (!PT) continue;
    auto *IT = dyn_cast<IntegerType>(PT->getElementType());
    if (!IT) continue;
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    unsigned Mode = ModeC ? (unsigned)ModeC->getZExtValue() : 0;
    bool HasWrite = false;
    for (User *AU : Arg->users()) {
      if (isa<StoreInst>(AU)) { HasWrite = true; break; }
      if (auto *AI = dyn_cast<Instruction>(AU)) {
        for (User *AU2 : AI->users()) {
          if (isa<StoreInst>(AU2)) { HasWrite = true; break; }
        }
        if (HasWrite) break;
      }
    }
    // Output ap_hs (written) normally skips this pass; `__vxx_aphs_stream` keeps it
    // so a directly-written ap_hs output becomes a directio (per-beat) port.
    if (HasWrite && !(ApHsStream && Mode == 4)) continue;
    // Rename only when the arg needs the directio/stream cosim TB:
    //   (a) legacy KPN context (SpecDataflowPipeline kind=1), or
    //   (b) ap_hs (mode 4) read-only scalar inside a dataflow region, or
    //   (c) `__vxx_aphs_stream`: any ap_hs (mode 4) scalar (looped in/out).
    // Everything else keeps i32* → cosim TB picks the plain Register wrapper,
    // the shape for `volatile T*` scalar args.
    bool InKpn = KpnFns.count(Arg->getParent());
    bool ApHsInDataflow = (Mode == 4) && DataflowFns.count(Arg->getParent());
    bool ApHsForced = ApHsStream && (Mode == 4);
    if (!InKpn && !ApHsInDataflow && !ApHsForced) continue;
    Cands.push_back({Arg->getParent(), Arg->getArgNo(), IT, Mode, HasWrite});
  }
  // Don't erase markers here — injectApScalar still needs them.

  if (Cands.empty()) return false;

  DenseMap<Function *, SmallVector<Cand, 4>> ByFn;
  for (auto &C : Cands) ByFn[C.F].push_back(C);

  bool Changed = false;
  for (auto &KV : ByFn) {
    Function *F = KV.first;
    auto &Group = KV.second;
    DenseMap<unsigned, Cand *> ByIdx;
    for (auto &C : Group) ByIdx[C.ArgIdx] = &C;

    FunctionType *OldFT = F->getFunctionType();
    unsigned NParams = OldFT->getNumParams();
    SmallVector<Type *, 8> NewParamTys;
    SmallVector<StructType *, 4> StTys(NParams, nullptr);
    for (unsigned i = 0; i < NParams; ++i) {
      auto It = ByIdx.find(i);
      if (It == ByIdx.end()) {
        NewParamTys.push_back(OldFT->getParamType(i));
        continue;
      }
      Cand *C = It->second;
      auto *IT = cast<IntegerType>(C->ElemTy);
      // mode 4 (ap_hs): lower `hls::ap_hs<int>&` as
      // `hls::directio<T, 0>&` → `class.hls::directio<T, 0>*`. The HLS
      // back-end auto-emits ap_hs SpecInterface (no duplicate ap_auto).
      // mode 0 (ap_none) / 2 (ap_vld): keep `class.hls::stream<T>*` (the
      // pattern that axilite_with_directio uses, which works for both
      // modes via the KpnDirectIO cosim TB path). For other modes, fall back
      // to `struct.ssdm_int<W, false>*`.
      std::string StName;
      std::string TName;
      switch (IT->getBitWidth()) {
        case 8:  TName = "char"; break;
        case 16: TName = "short"; break;
        case 32: TName = "int"; break;
        case 64: TName = "long"; break;
        default: TName = "i" + std::to_string(IT->getBitWidth()); break;
      }
      if (C->Mode == 4 && ApHsStream && !C->IsOutput) {
        // INPUT looped ap_hs → stream: apatb resolves stream direction through
        // the cpp_proxy wrap (the directio wrapc path mishandles an INPUT and
        // emits an undefined WRAPC_DIRECTIO_SIZE_OUT_<name>, HLS 212-317).
        StName = "class.hls::stream<" + TName + ">";
      } else if (C->Mode == 4) {
        StName = "class.hls::directio<" + TName + ", 0>";
      } else if (C->Mode == 0 || C->Mode == 2) {
        StName = "class.hls::stream<" + TName + ">";
      } else {
        StName = "struct.ssdm_int<" + std::to_string(IT->getBitWidth()) +
                 ", false>";
      }
      StructType *ST = M.getTypeByName(StName);
      if (!ST)
        ST = StructType::create(Ctx, {C->ElemTy}, StName, false);
      StTys[i] = ST;
      NewParamTys.push_back(PointerType::get(ST, 0));
    }

    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParamTys, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       F->getName() + ".ssdm_tmp",
                                       F->getParent());
    NewF->copyAttributesFrom(F);
    {
      AttributeList AL = NewF->getAttributes();
      for (auto &KV2 : ByIdx)
        AL = AL.removeParamAttributes(Ctx, KV2.first);
      NewF->setAttributes(AL);
    }
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);

    BasicBlock &Entry = NewF->getEntryBlock();
    Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
    IRBuilder<> B(InsertBefore);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);

    Function *Sideeffect =
        Intrinsic::getDeclaration(NewF->getParent(), Intrinsic::sideeffect);
    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    SmallVector<Argument *, 4> DirectioBundleArgs;
    SmallVector<Argument *, 4> StreamBundleArgs;
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (StTys[i] == nullptr) {
        AttributeSet AS = F->getAttributes().getParamAttributes(i);
        for (Attribute A : AS) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Bridge: GEP through ssdm_int struct's field 0 to get T*.
      Value *Idxs[] = {
          ConstantInt::get(I64Ty, 0),
          ConstantInt::get(I32Ty, 0),
      };
      Value *Bridge = B.CreateInBoundsGEP(StTys[i], &*NewA, Idxs,
                                           NewA->getName() + ".inner");
      OldA->replaceAllUsesWith(Bridge);
      // INPUT looped ap_hs → stream: the per-iteration scalar read must become a
      // `llvm.fpga.fifo.pop` so reflow synthesizes an ap_fifo input port. A plain
      // volatile load on the stream's field collapses to `ap_none` (a single
      // wire, no per-beat advance) → the RTL never consumes the TV beats → cosim
      // deadlock (0/1). Rewrite each volatile load of the bridged `T*` into a pop.
      {
        Cand *SC = ByIdx[i];
        if (SC && SC->Mode == 4 && ApHsStream && !SC->IsOutput) {
          Function *Pop = hlsrs::vxx::getOrInsertFifoPopAny(M, SC->ElemTy);
          SmallVector<LoadInst *, 4> Loads;
          for (User *BU : Bridge->users())
            if (auto *LD = dyn_cast<LoadInst>(BU))
              if (LD->getPointerOperand() == Bridge) Loads.push_back(LD);
          for (LoadInst *LD : Loads) {
            IRBuilder<> LB(LD);
            CallInst *Popped = LB.CreateCall(Pop, {Bridge});
            LD->replaceAllUsesWith(Popped);
            LD->eraseFromParent();
          }
        }
      }
      // Emit an interface op-bundle so reflow/cosim classify the port.
      // INPUT looped ap_hs (`__vxx_aphs_stream`, retyped to
      // `class.hls::stream<int>*`) → `stream_interface` (a real stream, so apatb
      // resolves its direction through the wrap). All other modes (0/2 ap_none/
      // ap_vld, OUTPUT ap_hs directio) → `directio_interface`.
      Cand *C = ByIdx[i];
      if (C && C->Mode == 4 && ApHsStream && !C->IsOutput)
        StreamBundleArgs.push_back(&*NewA);
      else if (C && (C->Mode == 0 || C->Mode == 2 || C->Mode == 4))
        DirectioBundleArgs.push_back(&*NewA);
    }
    for (Argument *DA : DirectioBundleArgs) {
      OperandBundleDef DB("directio_interface", ArrayRef<Value *>(DA));
      B.CreateCall(Sideeffect, ArrayRef<Value *>(), {DB});
    }
    for (Argument *SA : StreamBundleArgs) {
      OperandBundleDef SB("stream_interface", ArrayRef<Value *>(SA));
      auto *SC = B.CreateCall(Sideeffect, ArrayRef<Value *>(), {SB});
      SC->setOnlyAccessesInaccessibleMemory();
      SC->setDoesNotThrow();
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: renamed ap_scalar arg(s)\n";
  return Changed;
}












static void setupVitisModuleMetadata(Module &M) {
  LLVMContext &Ctx = M.getContext();

  // Set a canonical ModuleID + source_filename for the IR header.
  M.setModuleIdentifier("<stdin>");
  M.setSourceFileName("proj/s/.autopilot/db/a.g.0.bc");

  // triple `fpga{32,64}-unknown-unknown` → `fpga{32,64}-xilinx-none`
  StringRef Triple = M.getTargetTriple();
  if (Triple.startswith("fpga64") && Triple != "fpga64-xilinx-none")
    M.setTargetTriple("fpga64-xilinx-none");
  else if (Triple.startswith("fpga32") && Triple != "fpga32-xilinx-none")
    M.setTargetTriple("fpga32-xilinx-none");

  // Order matters: emit `llvm.ident`, then `blackbox_cfg`, then
  // `llvm.module.flags`, in that order so they serialize consistently.
  if (auto *Ident = M.getOrInsertNamedMetadata("llvm.ident")) {
    if (Ident->getNumOperands() == 0) {
      Ident->addOperand(MDNode::get(
          Ctx, {MDString::get(Ctx, "AMD/Xilinx clang version 16.0.6")}));
      // The ident list also pins a `clang version 7.0.0` node; we add a
      // single entry so downstream metadata indices
      // (`!fpga.pragma.source`, `!map`, ...) get consistent numbering.
      Ident->addOperand(MDNode::get(
          Ctx, {MDString::get(Ctx, "clang version 7.0.0 ")}));
    }
  }
  if (auto *BBCfg = M.getOrInsertNamedMetadata("blackbox_cfg")) {
    if (BBCfg->getNumOperands() == 0)
      BBCfg->addOperand(MDNode::get(Ctx, {}));
  }
  // Flag order: `wchar_size` first, then the full-lowering flag, so the
  // metadata index numbers stay consistent.
  if (M.getModuleFlag("wchar_size") == nullptr)
    M.addModuleFlag(Module::Error, "wchar_size", 4);
  // Module flag the Vitis HLS backend reads to enable its full HLS-lowering
  // pass on this module (interface synthesis, signature splitting, scheduling).
  // The backend stamps it on C++-derived modules itself; we set the same flag
  // so it treats our module identically. The key name is fixed by the backend.
  // Emit with `Module::Max` (severity 7), not Warning, so the bitcode merger
  // keeps the highest level when linking.
  if (M.getModuleFlag("reflow.full.lowering") == nullptr)
    M.addModuleFlag(Module::Max, "reflow.full.lowering", 1);
}

// For every `__vxx_aggregate(ptr, compact)` marker call, inject
//   call void @llvm.sideeffect() [ "aggregate"(<typed_ptr>, i64 compact) ]
// at the marker site. Mirrors `EmitAggregateIntrinsic`
// (`HLS/llvm/clang/lib/CodeGen/CGXlxAttr.cpp:1497`). The `compact` value
// rustc's `struct_llfields` (compiler/rustc_codegen_llvm/src/type_of.rs)
// always pushes a `type_padding_filler` before each field and at the
// trailing edge — even when the padding size is zero. With size=0 the
// filler emits as `[0 x iN]`. Naturally-aligned multi-field `#[repr(C)]`
// structs therefore emit as `{ [0 x iN], T1, [0 x iN], T2, ..., [0 x iN] }`
// — alternating zero-size pad fillers around real fields.
//
// Vitis HLS' csynth_design rejects `[0 x T]` struct fields with
// `[HLS 214-309] Detected unsupported array/vector as field with size 0`.
// We strip them in-place via `StructType::setBody` (LLVM allows replacing
// the body of an identified struct), and rewrite every GEP that indexed
// through a removed field. The byte layout is unchanged (zero-size fields
// occupy zero bytes), so values typed as the struct keep working as long
// as their GEPs are remapped.
//
// Two GEP shapes need rewriting:
//   1. `gep %S, 0, i32 K` where K is a real field's old index — remap to
//      the new index after stripping (`mapping[K]`).
//   2. `gep %S, 0, i32 P, i64 J` where P is a stripped `[0 x T]` field —
//      this is rustc's idiom for `&p.field0` (degenerate access via leading
//      pad). Collapse to `gep %S, 0, i32 mapping[next_real]`, dropping the
//      trailing `i64 J`.
// Defined later (after substituteTypeInFunction): LLVM-rule-compliant
// whole-module replacement of identified struct types.



// Recursively remap a type by substituting any occurrences of types in
// the `TypeMap`. Pointer/array types pass through, but their element
// types are recursed. Struct types are NOT recursed (would invalidate
// type identity); the caller is expected to pre-populate `TypeMap` for
// every struct type it wants substituted.

// Walk a function's body and substitute `OldT → NewT` (and any
// pointer/array wrappers thereof) in instruction types. For
// instructions whose result type changes (alloca, GEP, load, bitcast),
// recreate the instruction in place. For other instructions, only the
// operand types are remapped (which happens automatically when uses
// are RAUW'd to the new instructions).

// Rebuild a constant so it references remapped types. LLVM struct types are
// immutable once bodied, so when we replace a struct type we must also rebuild
// any ConstantStruct/Array/AggregateZero/Null that names the old type.

// LLVM-rule-compliant whole-module struct replacement.
//
// LLVM identified-struct bodies are IMMUTABLE once set (StructType::setBody
// asserts isOpaque()). So the legal way to "reshape" a named struct is NOT to
// mutate it in place, but to create a NEW struct type and rewrite every use of
// the old type across the module — function signatures, instruction types,
// global value-types and initializers — so nothing references the old type.
// `TypeMap` maps each old identified struct → its already-built replacement.



// =====================================================================
// Helpers
// =====================================================================

// traceToAllocaOrGlobal moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// Same as resolveMarkerArg but returns the underlying Value (alloca, global,
// or argument), not strictly an Argument*. Stays here: depends on the in-file
// resolveMarkerArg (a domain helper kept in VXXPrep.cpp).
static Value *resolveToKernelArg(Value *V) {
  if (auto *A = resolveMarkerArg(V)) return A;
  return hlsrs::vxx::traceToAllocaOrGlobal(V);
}

// Extract a literal NUL-terminated C string from a constant pointer V
// (typically a GEP into a `@.str = constant [N x i8] c"...\00"` global).
static StringRef extractStaticString(Value *V) {
  V = V->stripPointerCasts();
  GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  if (!GV) {
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::GetElementPtr)
        GV = dyn_cast<GlobalVariable>(CE->getOperand(0)->stripPointerCasts());
  }
  if (!GV || !GV->hasInitializer()) return "";
  if (auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer()))
    if (CDA->isCString()) return CDA->getAsCString();
  return "";
}

static StringRef extractSpecModeStr(Value *V) { return extractStaticString(V); }

static Loop *findEnclosingLoop(Instruction *I, LoopInfo &LI) {
  if (!I) return nullptr;
  return LI.getLoopFor(I->getParent());
}

// appendXilinxAttribute moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// Attach loop metadata to a call site (typically a sideeffect at loop hdr).
static MDNode *attachLoopMetadata(CallInst *CI, MDNode *Existing) {
  if (!CI || !Existing) return Existing;
  CI->setMetadata("llvm.loop", Existing);
  return Existing;
}

// =====================================================================
// Generic marker-erasure helpers
// =====================================================================
// dropMarkerDefinition / dropMarkerCallsAndDefinition / eraseUnimplementedMarkers
// moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// =====================================================================
// Generic fn-attribute injector. For marker calls with no positional ptr
// arg, attach AttrKey=AttrVal to the function containing the call.
// =====================================================================
// injectFnAttr moved to vxx_common.cpp (hlsrs::vxx:: namespace).

// =====================================================================
// Specific pragma injectors (each `__vxx_*` → ssdm op or sideeffect bundle)
// =====================================================================

static bool injectInline(Module &M) {
  Function *F = M.getFunction("__vxx_inline");
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  for (CallInst *CI : Calls) {
    Function *Parent = CI->getFunction();
    if (Parent) {
      Parent->removeFnAttr(Attribute::NoInline);
      Parent->addFnAttr(Attribute::AlwaysInline);
    }
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}

// latency() token chain → legacy `__vxx_latency(min, max)`, reassembled at the
// chain's begin site (which anchors the constrained region). Runs before
// injectLatency, which lowers the legacy marker to _ssdm_op_SpecLatency. Both
// forms accepted during the phase-2 pragma-builder migration.
static bool injectLatencyChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_latency_begin");
  if (!Begin) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Function *MinMk = M.getFunction("__vxx_latency_min");
  Function *MaxMk = M.getFunction("__vxx_latency_max");
  FunctionCallee LatFn = M.getOrInsertFunction(
      "__vxx_latency",
      FunctionType::get(Type::getVoidTy(Ctx), {I32, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(LatFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI) continue;
    ToErase.push_back(BeginCI);
    int64_t Min = 0, Max = 0;
    Value *Tok = BeginCI;
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == MinMk || Callee == MaxMk) { Next = CI; break; }
      }
      if (!Next) break;
      if (Next->arg_size() >= 2)
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1))) {
          if (Next->getCalledFunction() == MinMk) Min = C->getSExtValue();
          else Max = C->getSExtValue();
        }
      ToErase.push_back(Next);
      Tok = Next;
    }
    IRBuilder<> B(BeginCI);
    CallInst *NewCI = B.CreateCall(
        LatFn, {ConstantInt::get(I32, Min), ConstantInt::get(I32, Max)});
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    Changed = true;
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty()) Begin->eraseFromParent();
  if (MinMk && MinMk->use_empty()) MinMk->eraseFromParent();
  if (MaxMk && MaxMk->use_empty()) MaxMk->eraseFromParent();
  return Changed;
}

static bool injectLatency(Module &M) {
  Function *F = M.getFunction("__vxx_latency");
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    auto *MinC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    auto *MaxC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (MinC && MaxC) {
      // C++ pre-reflow form (free_running a.pp.bc):
      //   call void (...) @_ssdm_op_SpecLatency(i64 2, i64 2, [1 x i8]* @1)
      // — i64 operands plus the empty pragma-source string. Emitted in place,
      // so a marker inside a loop body scopes the constraint to that region,
      // exactly like the C++ `#pragma HLS LATENCY` written inside the loop.
      FunctionCallee SpecLat = hlsrs::vxx::getSsdmOp(M, "_ssdm_op_SpecLatency");
      GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
      Value *Args[] = {
        ConstantInt::get(Type::getInt64Ty(M.getContext()), MinC->getZExtValue()),
        ConstantInt::get(Type::getInt64Ty(M.getContext()), MaxC->getZExtValue()),
        EmptyStr
      };
      B.CreateCall(SpecLat, Args);
    }
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}

static bool injectVarReset(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_reset");
}





// __vxx_alias / __vxx_alias_pair: SpecAliasInst intrinsic isn't recognized
// by HLS clang-3.9-csynth (gives "Unknown intrinsic op" warning then fails).
// Drop the markers — HLS will lose the alias hint but synthesis proceeds.
// __vxx_fence(a, b) -> `llvm.fpga.fence(a, b, i32 -1, a, b, i32 1)` — the
// exact varargs shape the OSS clang-16 frontend emits for hls::fence(a, b)
// (aliasing_axi_master_ports test.bc: `call void (...) @llvm.fpga.fence(ptr,
// ptr, i32 -1, ptr, ptr, i32 1)`). Do NOT emit `_ssdm_op_Fence` here: that is
// the POST-split form. The closed flow first splits an m_axi arg into
// (bundle `i32 addrspace(1)*`, offset i64) and THEN lowers llvm.fpga.fence
// to _ssdm_op_Fence with the split values — the fence consumer (a.g.lto →
// a.g.ld.0 boundary) only recognises those direct port references. A
// pre-split _ssdm_op_Fence keeps the raw pointer alive through the split via
// a ptrtoint/inttoptr reconstruction, which the consumer cannot see through;
// the un-consumed fence then survives to RTL generation and SIGSEGVs
// fsmd::RtlGen::addRtlSignalAssign (verified 2026-07-10, aliasing).
static bool injectFence(Module &M) {
  Function *Marker = M.getFunction("__vxx_fence");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee FenceFn = M.getOrInsertFunction("llvm.fpga.fence", FT);
  if (auto *F = dyn_cast<Function>(FenceFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    Value *A = CI->getArgOperand(0)->stripPointerCasts();
    Value *B = CI->getArgOperand(1)->stripPointerCasts();
    IRBuilder<> Bld(CI);
    Bld.SetCurrentDebugLocation(DebugLoc());
    Bld.CreateCall(FenceFn, {A, B, ConstantInt::get(I32, (uint64_t)-1, true),
                             A, B, ConstantInt::get(I32, 1)});
    CI->eraseFromParent();
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_fence");
  return !Calls.empty();
}

static bool injectAlias(Module &M) { return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_alias"); }
static bool injectAliasPair(Module &M) { return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_alias_pair"); }






















// Unified s_axilite pass — single IRBuilder per function, walks
// `__vxx_s_axilite(port_str, ..., bundle_str, ..., offset)` markers in
// source-call order. Emission order:
//   for each port marker (port != "return"):
//     SpecBitsMap(arg) + SpecInterface(arg, s_axilite, bundle, offset) +
//     SpecInterface(arg, ap_auto)
//   then once: SpecInterface(i32 0, s_axilite, bundle) for port == "return"
// vxxParseRustStrSlice moved to vxx_common.cpp (hlsrs::vxx:: namespace).




// __vxx_array_reshape(ptr, type, factor, dim): emit SpecArrayReshape.
// Physically pack a local array alloca per ARRAY_RESHAPE (block/cyclic, factor
// F):
//   `[N x iW]` alloca  →  `[N/F x i(F*W)]` alloca (align 512)
// and every element load/store is rewritten to PartSelect/PartSet on the packed
// word — with NO SpecArrayReshape directive emitted. Emitting the directive on
// an *un-packed* local alloca instead makes HLS fail with
// `200-70 Failed building synthesis data model` (ecc_flags: the uram_ecc
// BIND_STORAGE needs the 64-bit packed word). block: word=i%(N/F), pos=i/(N/F);
// cyclic: word=i/F, pos=i%F. Returns true if the alloca was packed; false (no
// change) if the shape is unexpected so the caller can fall back to the directive.
[[maybe_unused]] static bool physicallyReshapeAlloca(Module &M, AllocaInst *AI,
                                                     uint64_t Factor,
                                                     uint64_t Dim, bool Block) {
  if (Dim != 1 || Factor < 2) return false;
  auto *ArrTy = dyn_cast<ArrayType>(AI->getAllocatedType());
  if (!ArrTy) return false;
  auto *ElemTy = dyn_cast<IntegerType>(ArrTy->getElementType());
  if (!ElemTy) return false;
  uint64_t N = ArrTy->getNumElements();
  unsigned W = ElemTy->getBitWidth();
  if (N == 0 || (N % Factor) != 0) return false;
  uint64_t NF = N / Factor;
  unsigned FW = W * (unsigned)Factor;
  if (FW > 4096) return false;

  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  IntegerType *WordTy = IntegerType::get(Ctx, FW);
  ArrayType *PackedTy = ArrayType::get(WordTy, NF);

  // Classify uses: element GEPs `(0, idx)` feeding load/store, vs non-deref ptr
  // uses (bind_storage / reshape markers, bitcasts, lifetime, dbg). Bail on
  // anything else so we never corrupt an unexpected access shape.
  SmallVector<GetElementPtrInst *, 16> ElemGEPs;
  SmallVector<Use *, 8> OtherUses;
  for (Use &U : AI->uses()) {
    User *Usr = U.getUser();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
      if (GEP->getPointerOperand() == AI && GEP->getNumIndices() == 2) {
        if (auto *I0 = dyn_cast<ConstantInt>(GEP->getOperand(1)))
          if (I0->isZero()) { ElemGEPs.push_back(GEP); continue; }
      }
      return false;
    }
    OtherUses.push_back(&U);
  }
  // Every element GEP must only feed loads/stores of the element type.
  for (GetElementPtrInst *GEP : ElemGEPs)
    for (User *GU : GEP->users()) {
      if (auto *LD = dyn_cast<LoadInst>(GU)) {
        if (LD->getType() != ElemTy) return false;
      } else if (auto *ST = dyn_cast<StoreInst>(GU)) {
        if (ST->getValueOperand()->getType() != ElemTy ||
            ST->getPointerOperand() != GEP) return false;
      } else return false;
    }

  std::string Nm = AI->getName().str();
  IRBuilder<> AB(AI);
  AllocaInst *NA = AB.CreateAlloca(PackedTy, nullptr, "");
  NA->setAlignment(Align(512));

  std::string SelName = "_ssdm_op_PartSelect.i" + std::to_string(W) + ".i" +
                        std::to_string(FW) + ".i64";
  std::string SetName = "_ssdm_op_PartSet.i" + std::to_string(FW) + ".i" +
                        std::to_string(FW) + ".i" + std::to_string(W) + ".i64";
  FunctionCallee PSel = M.getOrInsertFunction(
      SelName, FunctionType::get(ElemTy, {WordTy, I64}, false));
  FunctionCallee PSet = M.getOrInsertFunction(
      SetName, FunctionType::get(WordTy, {WordTy, ElemTy, I64}, false));

  Value *Zero = ConstantInt::get(I64, 0);
  Value *NFc = ConstantInt::get(I64, NF);
  Value *Fc = ConstantInt::get(I64, Factor);
  Value *Wc = ConstantInt::get(I64, W);

  for (GetElementPtrInst *GEP : ElemGEPs) {
    IRBuilder<> GB(GEP);
    Value *Idx = GEP->getOperand(2);
    if (Idx->getType() != I64) Idx = GB.CreateZExtOrTrunc(Idx, I64);
    Value *Word = Block ? GB.CreateURem(Idx, NFc) : GB.CreateUDiv(Idx, Fc);
    Value *Pos = Block ? GB.CreateUDiv(Idx, NFc) : GB.CreateURem(Idx, Fc);
    Value *Lo = GB.CreateMul(Wc, Pos);
    Value *WGEP = GB.CreateGEP(PackedTy, NA, {Zero, Word});

    SmallVector<Instruction *, 4> GUsers;
    for (User *GU : GEP->users())
      if (auto *I = dyn_cast<Instruction>(GU)) GUsers.push_back(I);
    for (Instruction *GU : GUsers) {
      if (auto *LD = dyn_cast<LoadInst>(GU)) {
        IRBuilder<> LB(LD);
        Value *Wv = LB.CreateLoad(WordTy, WGEP);
        Value *Val = LB.CreateCall(PSel, {Wv, Lo});
        LD->replaceAllUsesWith(Val);
        LD->eraseFromParent();
      } else if (auto *ST = dyn_cast<StoreInst>(GU)) {
        IRBuilder<> SB(ST);
        Value *Wv = SB.CreateLoad(WordTy, WGEP);
        Value *W2 = SB.CreateCall(PSet, {Wv, ST->getValueOperand(), Lo});
        SB.CreateStore(W2, WGEP);
        ST->eraseFromParent();
      }
    }
    GEP->eraseFromParent();
  }

  // Point remaining non-deref uses (bind_storage/reshape markers, lifetime) at
  // NA via a bitcast to the old pointer type so downstream passes resolve to NA.
  if (!OtherUses.empty()) {
    IRBuilder<> CB(AI);
    Value *BC = CB.CreateBitCast(NA, AI->getType());
    for (Use *U : OtherUses) U->set(BC);
  }
  AI->eraseFromParent();
  NA->setName(Nm);
  return true;
}























static bool injectFunctionAllocation(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_function_allocation");
}

// injectFunctionInstantiate now lives in loop.cpp (emits the
// `fpga.func.instantiate` op-bundle so reflow specialises the function) and is
// declared in vxx_passes.h; it is called late in the pipeline (below).

static bool injectOccurrence(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_occurrence");
}






// Scope-pair markers (begin/end): used for protocol / array_view / etc.
// HLS handles these via region tagging; just drop the markers.
static bool injectScopePair(Module &M) {
  bool A = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_protocol_begin");
  bool B = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_protocol_end");
  return A || B;
}
static bool injectProtocolScope(Module &M) { return injectScopePair(M); }


// Catch-all for misc pragma sideeffects that have no specific lowering.
static bool injectPragmaSideeffects(Module &M) {
  bool Changed = false;
  // Token-chain dependence()/stable() builders reassemble their legacy markers
  // BEFORE the Spec consumers below (phase-2 pragma-builder migration).
  Changed |= injectDependenceChain(M);
  Changed |= injectStableChain(M);
  // __vxx_dependence → _ssdm_SpecDependence (then drop leftovers).
  Changed |= injectDependenceSpec(M);
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_dependence");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_expression_balance");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_loop_merge");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_shared");
  // __vxx_stable → SpecStableContent (then drop leftovers).
  Changed |= injectStableSpec(M);
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_stable");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_stable_content");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_disaggregate");
  // __vxx_bind_storage → SpecResource (handled separately above; drop any
  // leftover markers if SpecResource emit didn't fire).
  Changed |= injectBindStorageSpecResource(M);
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_bind_storage");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_top_dataflow");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_top_return");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_top_directio_param");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_directio_read");
  Changed |= hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_directio_write");
  return Changed;
}

// =====================================================================
// Strip / normalize passes (mostly no-op — the intact passes above
// already handle the heavy lifting for these areas)
// =====================================================================


// Packet trailing pad strip was tried but counterproductive: rustc SROA
// packs all Packet fields into i96 (12 bytes), which then OOBs a 10-byte
// post-strip Packet alloca. Real fix needs full AXIS struct disagg (Stream
// rewriter that prevents the i96 packing path). Revert to no-op.




// Remove `noalias` (+ `readonly`) from top kernel arg attributes — HLS
// 200-1986 "Could not apply TOP directive, invalid function" fires when
// these are on top kernel args. Detect top fn by `__vxx_top_kernel`
// marker call (runs before injectKernelTopAttribute, so fpga.top.func
// attr isn't set yet).
static bool stripTopKernelNoalias(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> Tops;
  if (Function *Marker = M.getFunction("__vxx_top_kernel"))
    for (User *U : Marker->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *Parent = CI->getFunction())
          Tops.insert(Parent);
  // Also handle case where fpga.top.func already set (later passes / re-run).
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasFnAttribute("fpga.top.func"))
      Tops.insert(&F);
  for (Function *F : Tops) {
    // Strip `unnamed_addr` from the function itself — Rust adds it but
    // HLS scheduler treats top kernels as named-addr.
    if (F->hasAtLeastLocalUnnamedAddr()) {
      F->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
      Changed = true;
    }
    // Strip `target-cpu="generic"` — rustc adds it on every function but
    // HLS prints `'generic' is not a recognized processor for this target`
    // warnings AND its scheduler's CoreAgent::configDelayBudget segfaults
    // when no matching platform agent exists for the cpu string. Synthesized
    // functions should have no `target-cpu` attribute at all.
    if (F->hasFnAttribute("target-cpu")) {
      F->removeFnAttr("target-cpu");
      Changed = true;
    }
    // Strip Rust-specific !noalias / !alias.scope metadata from kernel body
    // load/store ops. The HLS scheduler's AxiInterfaceAccessDelay lookup
    // segfaults when it encounters these on s_axilite-bound i8* args, so
    // load/store ops must carry no alias metadata.
    // Strip dead `bitcast i8* %a to {}*` debris (rustc emits these for
    // closure capture analysis; HLS shouldn't see them).
    SmallVector<Instruction *, 8> DeadBC;
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (isa<LoadInst>(&I) || isa<StoreInst>(&I)) {
          I.setMetadata(LLVMContext::MD_noalias, nullptr);
          I.setMetadata(LLVMContext::MD_alias_scope, nullptr);
          I.setMetadata(LLVMContext::MD_tbaa, nullptr);
          Changed = true;
        }
        if (auto *BC = dyn_cast<BitCastInst>(&I)) {
          if (BC->use_empty()) DeadBC.push_back(BC);
        }
      }
    }
    for (Instruction *I : DeadBC) { I->eraseFromParent(); Changed = true; }
    for (Argument &A : F->args()) {
      // `__vxx_axis_packed`: a packed `class.hls::stream<...>*` top arg needs
      // `nonnull align N dereferenceable(M)` and **no `noalias`**. Adding
      // `noalias` here makes the HLS back-end's
      // CorrelatedValuePropagation / LazyValueInfo SIGSEGV on the
      // flattened-inline AXIS top (causes a deadlock). The disagg
      // form DOES carry noalias on its 14 scalar channel args — but
      // that's a different representation. Detect the packed stream arg by its
      // `class.hls::stream` pointee struct name and give it nonnull, not noalias.
      bool PackedStreamArg = false;
      if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed") && A.getType()->isPointerTy()) {
        if (auto *ST = dyn_cast<StructType>(
                A.getType()->getPointerElementType()))
          if (ST->hasName() &&
              ST->getName().startswith("class.hls::stream"))
            PackedStreamArg = true;
      }
      if (PackedStreamArg) {
        if (A.hasAttribute(Attribute::NoAlias)) {
          A.removeAttr(Attribute::NoAlias);
          Changed = true;
        }
        if (!A.hasAttribute(Attribute::NonNull)) {
          A.addAttr(Attribute::NonNull);
          Changed = true;
        }
      } else
      // ENSURE noalias on every ptr arg. Rust's `&mut T` doesn't always get
      // noalias from rustc (historical LLVM-bug workaround). Without
      // noalias on the write target arg, the HLS scheduler can't reason
      // about pointer aliasing → 200-1986 TOP reject + scheduler segfault.
      if (A.getType()->isPointerTy() && !A.hasAttribute(Attribute::NoAlias)) {
        A.addAttr(Attribute::NoAlias);
        Changed = true;
      }
      // Strip `readonly` from ptr args unconditionally: the HLS back-end
      // recomputes access_type from the SpecInterface direction and ignores
      // the LLVM readonly attr, so it has no effect on cosim.
      if (A.hasAttribute(Attribute::ReadOnly)) {
        A.removeAttr(Attribute::ReadOnly);
        Changed = true;
      }
      // dereferenceable strip + nocapture add — narrow gate for SCALAR
      // pointer args (pointee is primitive integer/float, not array/struct).
      // Rust emits `i32* noalias align 4 dereferenceable(4) %dout`; the
      // dereferenceable attr triggers Vitis HLS to emit an extra `ap_clk`
      // port (10 vs 9 for using_C++_templates). Stripping dereferenceable +
      // adding nocapture yields `i32* noalias nocapture align 4 %dout`.
      // SAFE for AXIS/array args because those have ArrayType pointee (gate
      // skips them).
      bool IsScalarPtrArg = false;
      if (auto *PT = dyn_cast<PointerType>(A.getType())) {
        Type *PointeeTy = PT->getElementType();
        if (PointeeTy->isIntegerTy() || PointeeTy->isFloatingPointTy())
          IsScalarPtrArg = true;
      }
      if (IsScalarPtrArg) {
        if (A.hasAttribute(Attribute::Dereferenceable)) {
          A.removeAttr(Attribute::Dereferenceable);
          Changed = true;
        }
        if (A.hasAttribute(Attribute::DereferenceableOrNull)) {
          A.removeAttr(Attribute::DereferenceableOrNull);
          Changed = true;
        }
        // Don't strip alignment — C++ keeps `align 4`.
        if (!A.hasAttribute(Attribute::NoCapture)) {
          A.addAttr(Attribute::NoCapture);
          Changed = true;
        }
      }
      // Add `nocapture` — HLS uses it to confirm the pointer doesn't escape,
      // enabling cleaner address-space inference.
      // EXCEPT: skip args that have `_ssdm_SpecStream` emitted — stream
      // args (e.g. auto_disaggregation_of_struct d_s_in) carry `noalias`
      // only, no nocapture. With nocapture, HLS raises 201-504 "Interface
      // type specification conflicts" on the stream port.
      bool HasSpecStream = false;
      for (Instruction &II : F->getEntryBlock()) {
        auto *CIS = dyn_cast<CallInst>(&II);
        if (!CIS || !CIS->getCalledFunction()) continue;
        if (CIS->getCalledFunction()->getName() != "_ssdm_SpecStream") continue;
        if (CIS->arg_size() < 1) continue;
        Value *V = CIS->getArgOperand(0);
        while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
        if (V == &A) { HasSpecStream = true; break; }
      }
      // Also skip nocapture for packed-int array args (`[N x iK]*` where
      // K > 64) — these are packAggregateByteKernelSig outputs. Adding
      // nocapture triggers the HLS back-end to auto-add SpecInterface(ap_memory)
      // even when explicit bram marker is present (203-801 conflict on
      // aggregation_of_nested_structs port `c`).
      bool IsPackedIntArray = false;
      if (auto *PT = dyn_cast<PointerType>(A.getType())) {
        if (auto *AT = dyn_cast<ArrayType>(PT->getElementType())) {
          if (auto *IT = dyn_cast<IntegerType>(AT->getElementType())) {
            if (IT->getBitWidth() > 64) IsPackedIntArray = true;
          }
        }
      }
      // Strip nocapture for stream-like args. Do NOT add nocapture: it causes
      // the HLS back-end to layer SpecInterface(ap_auto) on top of explicit
      // SpecInterface(axis/bram/etc), breaking cosim TB TV capture on AXIS
      // array-stream cosim (using_axis_array_stream_no_side_channel_data:
      // 212-361).
      if (HasSpecStream || IsPackedIntArray) {
        if (A.hasAttribute(Attribute::NoCapture)) {
          A.removeAttr(Attribute::NoCapture);
          Changed = true;
        }
      }
      // Strip nocapture for all ptr args at phase-10 final cleanup EXCEPT
      // scalar pointer args (where we WANT nocapture to prevent an extra
      // ap_clk port in IP-XACT).
      if (A.getType()->isPointerTy() && A.hasAttribute(Attribute::NoCapture)
          && !IsScalarPtrArg) {
        A.removeAttr(Attribute::NoCapture);
        Changed = true;
      }
    }
  }
  return Changed;
}


// rustc lowers `extern "C" fn f(a: BitInt<N>, ..., out: &mut BitInt<N>)`
// to `declare void @f(i128, ..., i128* align 16 dereferenceable(16))`.
// `BitInt<N>` = `#[rustc_apint(N)]` struct over u128 storage; the C ABI
// normalises to the storage type i128, losing N. Callers always
// `sext/zext iN to i128` for value args and use `alloca i128 align 16`
// + `load i128` for ptr args. HLS RTL-blackbox bind
// (`add_files -blackbox <json>`) then fails because the JSON expects
// ap_int<N> port widths but the kernel.bc declares i128.
//
// Narrow each candidate extern declare from i128(*) to iN(*) by
// inferring N from caller patterns. Strictly conservative: only fires
// when every callsite agrees on the same N for each arg index. No
// hazard on libcore extern decls (they don't have all-i128 signatures).
// Runs in VXXEarlyPrep so LLVM mid-end opts clean up the residual
// ext/trunc chains automatically.

static bool normalizeApScalarSpecInterface(Module &M) { (void)M; return false; }
static bool renameAxisDisabledToCanonical(Module &M) { (void)M; return false; }
static bool replaceFptosiWithHlsHelper(Module &M)    { (void)M; return false; }

// Vitis HLS clang-3.9-csynth (LLVM 7) crashes in ADCE::markLive when
// multiple `_ssdm_op_SpecInterface(ptr addrspace(1) null, ...)` calls
// have the same null gmem bundle key — that's what `injectMAxiSpecInterface`
// emits for each m_axi port using the default "gmem" bundle. Dedupe so
// only one survives per call site.
// Multiple passes (injectSAxilitePortSpec, m_axi auto-emit, etc.) each
// emit SpecBitsMap + SpecInterface for the same kernel arg with
// overlapping mode strings (e.g. two "ap_auto" calls for the same %a).
// HLS scheduler's `getAxiInterfaceAccessDelay` segfaults when a single
// arg has two SpecInterface entries for the same mode — it can't decide
// which one's delay budget to honor.
//
// Walk each kernel function; group SpecInterface/SpecBitsMap calls by
// (arg SSA value, mode-string for SpecInterface; arg only for SpecBitsMap).
// Keep the first, drop subsequent duplicates.
static bool dedupePerArgSpecInterface(Module &M) {
  bool Changed = false;
  Function *SpecF = M.getFunction("_ssdm_op_SpecInterface");
  Function *BMapF = M.getFunction("_ssdm_op_SpecBitsMap");
  if (!SpecF && !BMapF) return false;
  for (Function &Fn : M) {
    if (Fn.isDeclaration()) continue;
    std::set<std::string> SeenSI, SeenBM;
    SmallVector<CallInst *, 16> Dead;
    for (BasicBlock &BB : Fn) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == SpecF) {
          if (CI->arg_size() < 2) continue;
          // arg 0: target ptr/value (only dedupe when it's an Argument or
          // ConstantInt(0)/ConstantPointerNull — i.e. addressable target).
          Value *A0 = CI->getArgOperand(0);
          std::string ArgKey;
          if (auto *Arg = dyn_cast<Argument>(A0))
            ArgKey = "%a" + std::to_string(Arg->getArgNo());
          else continue;  // i32 0 / null = return spec, leave alone
          // arg 1: mode string global. Walk through any GEP cast.
          Value *A1 = CI->getArgOperand(1)->stripPointerCasts();
          std::string Mode;
          if (auto *GV = dyn_cast<GlobalVariable>(A1)) {
            if (auto *Init = dyn_cast<ConstantDataArray>(GV->getInitializer()))
              if (Init->isString()) Mode = Init->getAsString().str();
          }
          if (Mode.empty()) continue;
          std::string Key = ArgKey + "|" + Mode;
          if (!SeenSI.insert(Key).second) Dead.push_back(CI);
        } else if (Callee == BMapF) {
          if (CI->arg_size() < 1) continue;
          Value *A0 = CI->getArgOperand(0);
          if (auto *Arg = dyn_cast<Argument>(A0)) {
            std::string Key = "%a" + std::to_string(Arg->getArgNo());
            if (!SeenBM.insert(Key).second) Dead.push_back(CI);
          }
        }
      }
    }
    for (CallInst *CI : Dead) { CI->eraseFromParent(); Changed = true; }
  }
  return Changed;
}

static bool dedupeSpecInterfaceNullGmem(Module &M) {
  bool Changed = false;
  for (StringRef SpecName : {"_ssdm_op_SpecInterface", "_ssdm_op_SpecBitsMap"}) {
    Function *F = M.getFunction(SpecName);
    if (!F) continue;
    for (Function &Fn : M) {
      if (Fn.isDeclaration()) continue;
      std::set<std::string> Seen;
      SmallVector<CallInst *, 8> Dead;
      for (BasicBlock &BB : Fn) {
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || CI->getCalledFunction() != F) continue;
          if (CI->arg_size() < 1) continue;
          auto *A0 = dyn_cast<ConstantPointerNull>(CI->getArgOperand(0));
          if (!A0) continue;
          std::string Key = SpecName.str() + "|null";
          if (CI->arg_size() >= 2) {
            if (auto *GV = dyn_cast<GlobalVariable>(CI->getArgOperand(1)->stripPointerCasts()))
              Key += "|" + GV->getName().str();
          }
          if (CI->arg_size() >= 7) {
            if (auto *CD = dyn_cast<ConstantInt>(CI->getArgOperand(6)))
              Key += "|d=" + std::to_string(CD->getZExtValue());
          }
          if (!Seen.insert(Key).second) Dead.push_back(CI);
        }
      }
      for (CallInst *CI : Dead) { CI->eraseFromParent(); Changed = true; }
    }
  }
  return Changed;
}

// Vitis HLS clang-3.9-csynth (LLVM 7) crashes in ADCE::markLive when it
// sees `@llvm.assume`. The intrinsic carries no semantics needed for
// HLS — drop all calls + the declaration.

// `<T as From<T>>::from` is the identity blanket impl. rustc sometimes
// leaves it as an external call. HLS rejects "Undefined function".
// Replace each call with its arg, then drop the declaration.

// LLVM 7's verifier rejects `store T %val, U* %ptr` when T != pointee(U)
// — LLVM 11 is more permissive. Insert a bitcast `U* → T*` before the
// store. Same for `%val = load T, U* %ptr` when T != pointee(U).
static bool normalizeStoreLoadPointerTypes(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<Instruction *, 16> Targets;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isa<StoreInst>(&I) || isa<LoadInst>(&I)) Targets.push_back(&I);
    for (Instruction *I : Targets) {
      Value *Ptr; Type *ValTy;
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        Ptr = SI->getPointerOperand();
        ValTy = SI->getValueOperand()->getType();
      } else {
        auto *LI = cast<LoadInst>(I);
        Ptr = LI->getPointerOperand();
        ValTy = LI->getType();
      }
      auto *PT = dyn_cast<PointerType>(Ptr->getType());
      if (!PT) continue;
      if (PT->getElementType() == ValTy) continue;
      // Mismatch — insert bitcast.
      IRBuilder<> B(I);
      PointerType *NewPT = PointerType::get(ValTy, PT->getAddressSpace());
      Value *NewPtr = B.CreateBitCast(Ptr, NewPT);
      if (auto *SI = dyn_cast<StoreInst>(I))
        SI->setOperand(1, NewPtr);
      else
        cast<LoadInst>(I)->setOperand(0, NewPtr);
      Changed = true;
    }
  }
  return Changed;
}

// LLVM 7's verifier (in clang-3.9-csynth) rejects @llvm.memcpy with
// mismatched src/dst alignments ("source and dest alignments must be the
// same"). LLVM 11 allows it. Take the min of the two so the IR survives
// LLVM 7 llvm-as during the .ll → kernel.bc convert step.
static bool normalizeMemcpyAlignments(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<MemTransferInst *, 8> MTIs;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *MT = dyn_cast<MemTransferInst>(&I)) MTIs.push_back(MT);
    for (MemTransferInst *MT : MTIs) {
      MaybeAlign DA = MT->getDestAlign();
      MaybeAlign SA = MT->getSourceAlign();
      unsigned D = DA ? DA->value() : 1;
      unsigned S = SA ? SA->value() : 1;
      unsigned Min = std::min(D, S);
      if (D != Min) { MT->setDestAlignment(Align(Min)); Changed = true; }
      if (S != Min) { MT->setSourceAlignment(Align(Min)); Changed = true; }
    }
  }
  return Changed;
}


// Rust's `for i in 0..N { dst[i] = src[i] }` gets vectorized by LLVM into
// `llvm.memcpy(dst, src, N*sizeof(T))`. When `src` and `dst` are kernel
// ap_fifo args, HLS scheduler errors out with SYNCHK 200-91 "Port has
// both read and write" — HLS memcpy expansion uses wider-than-element
// accesses that include reading the dst pointer for partial-store
// preservation. Detect memcpy where both src and dst are bitcasts of
// `[N x T]*` arg pointers (the decay shape used in our kernels) and
// expand to a clean per-element load+store loop.
static bool lowerKernelMemcpyToLoop(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    // Note: `fpga.top.func` attribute is set later in Phase 5; we run in
    // Phase 3 so we cannot gate on it. The pass keys on per-memcpy
    // shape (both src and dst are kernel-arg-shaped `[N x T]*`) which is
    // sufficient — non-top functions almost never have this shape.
    SmallVector<MemCpyInst *, 4> Memcpys;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *MC = dyn_cast<MemCpyInst>(&I))
          if (isa<ConstantInt>(MC->getLength()))
            Memcpys.push_back(MC);
    // Walk bitcasts towards the kernel-arg-shaped pointer (`[N x T]*`).
    // The body shape we want is:
    //   %arr_p = bitcast i8* %i8_ptr to [N x T]*  (or via a chain)
    // We accept any intermediate bitcasts and stop at the `[N x T]*` form.
    auto resolveArrayPtr = [](Value *V) -> std::pair<Value *, ArrayType *> {
      Value *Cur = V;
      ArrayType *AT = nullptr;
      while (true) {
        if (auto *PT = dyn_cast<PointerType>(Cur->getType()))
          if ((AT = dyn_cast<ArrayType>(PT->getPointerElementType())))
            return {Cur, AT};
        if (auto *BC = dyn_cast<BitCastOperator>(Cur)) {
          Cur = BC->getOperand(0);
          continue;
        }
        break;
      }
      return {nullptr, nullptr};
    };
    for (MemCpyInst *MC : Memcpys) {
      auto DstPair = resolveArrayPtr(MC->getRawDest());
      auto SrcPair = resolveArrayPtr(MC->getRawSource());
      Value *Dst = DstPair.first;
      Value *Src = SrcPair.first;
      ArrayType *DstAT = DstPair.second;
      ArrayType *SrcAT = SrcPair.second;
      if (!Dst || !Src || !DstAT || !SrcAT) continue;
      if (DstAT->getElementType() != SrcAT->getElementType()) continue;
      // At least one side must trace (through bitcasts only) to a top arg.
      auto stripBC = [](Value *V) {
        while (auto *BC = dyn_cast<BitCastOperator>(V))
          V = BC->getOperand(0);
        return V;
      };
      bool DstIsArg = isa<Argument>(stripBC(Dst));
      bool SrcIsArg = isa<Argument>(stripBC(Src));
      if (!DstIsArg && !SrcIsArg) continue;
      Type *ElemTy = DstAT->getElementType();
      uint64_t ElemSize = M.getDataLayout().getTypeStoreSize(ElemTy);
      if (ElemSize == 0) continue;
      uint64_t Len = cast<ConstantInt>(MC->getLength())->getZExtValue();
      if (Len % ElemSize) continue;
      uint64_t N = Len / ElemSize;
      if (N == 0) continue;
      if (DstAT->getNumElements() < N || SrcAT->getNumElements() < N) continue;
      // Build the loop. We splice the original block at MC into:
      //   entry: ... br loop
      //   loop:  i = phi [0,entry],[i.next,loop]
      //          load src[i]; store dst[i]; i.next = i+1
      //          br (i.next < N) loop, exit
      //   exit:  ... rest of orig BB
      LLVMContext &Ctx = M.getContext();
      IRBuilder<> B(MC);
      Type *I64 = Type::getInt64Ty(Ctx);
      BasicBlock *Entry = MC->getParent();
      BasicBlock *Exit = Entry->splitBasicBlock(MC, Entry->getName() + ".memcpy.exit");
      BasicBlock *Loop = BasicBlock::Create(Ctx, Entry->getName() + ".memcpy.loop", &F, Exit);
      // Replace the unconditional branch from Entry to Exit with a branch to Loop.
      Entry->getTerminator()->eraseFromParent();
      B.SetInsertPoint(Entry);
      B.CreateBr(Loop);
      B.SetInsertPoint(Loop);
      PHINode *I = B.CreatePHI(I64, 2, "memcpy.i");
      I->addIncoming(ConstantInt::get(I64, 0), Entry);
      Value *SrcGep = B.CreateInBoundsGEP(SrcAT, Src, {ConstantInt::get(I64, 0), I});
      Value *V = B.CreateAlignedLoad(ElemTy, SrcGep, MC->getSourceAlign().getValueOr(Align(1)));
      Value *DstGep = B.CreateInBoundsGEP(DstAT, Dst, {ConstantInt::get(I64, 0), I});
      B.CreateAlignedStore(V, DstGep, MC->getDestAlign().getValueOr(Align(1)));
      Value *INext = B.CreateAdd(I, ConstantInt::get(I64, 1), "memcpy.i.next");
      I->addIncoming(INext, Loop);
      Value *Cond = B.CreateICmpULT(INext, ConstantInt::get(I64, N), "memcpy.cmp");
      B.CreateCondBr(Cond, Loop, Exit);
      MC->eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: lowered top-kernel memcpy to per-element loop\n";
  return Changed;
}

// rustc OpenCL/SPIR named metadata leaks: rustc's LLVM 11 codegen attaches
// !opencl.ocl.version + !opencl.spir.version on every TU which leak through
// to the kernel.bc. The HLS back-end's clock-gen path interprets opencl
// markers as "this is an OpenCL kernel, needs ap_clk wrapping". For purely
// combinational kernels (using_C++_templates, etc.) Rust would emit a
// spurious ap_clk solely because of these leaked named metadata. Strip them.

}  // end anonymous namespace

// =====================================================================
// C ABI entry points called via dlopen from rustc / opt plugin loader.
// =====================================================================

// LLVM 11's `freeze` instruction (rustc emits it for div/rem and other
// poison-guarding lowerings, e.g. the `idx/W`,`idx%W` of a flattened loop)
// is not understood by Vitis' LLVM-7 `llvm-as` ("expected instruction
// opcode"). `freeze X` produces a non-poison version of X; HLS has no poison
// semantics, so replacing each `freeze X` with `X` is sound. Strip them so
// the emitted .ll round-trips through the LLVM-7 assembler.

// LLVM 7 verifier compat: `llvm.lifetime.start/end.p0i8` is declared as
// `void (i64, i8*)`; rustc emits the pointer arg through an i8* bitcast, but
// VXXPrep type-rewrite passes (AxisDisabled strip / struct retype) can RAUW
// that operand with a typed pointer, leaving e.g.
//   call void @llvm.lifetime.start.p0i8(i64 10, %"barista_hls::AxisDisabled"* %p)
// which llvm-as rejects ("defined with type 'void (i64, i8*)*'",
// custom_side_2). Re-insert the canonical i8* bitcast on any mistyped arg.
static bool normalizeLifetimeIntrinsicArgs(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    StringRef N = F.getName();
    if (!N.startswith("llvm.lifetime.") && !N.startswith("llvm.memcpy.") &&
        !N.startswith("llvm.memmove.") && !N.startswith("llvm.memset."))
      continue;
    FunctionType *FT = F.getFunctionType();
    for (User *U : F.users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      for (unsigned i = 0, e = std::min((unsigned)CI->arg_size(),
                                        FT->getNumParams());
           i < e; ++i) {
        Value *Op = CI->getArgOperand(i);
        Type *Want = FT->getParamType(i);
        if (Op->getType() == Want) continue;
        if (!Op->getType()->isPointerTy() || !Want->isPointerTy()) continue;
        IRBuilder<> B(CI);
        CI->setArgOperand(i, B.CreateBitCast(Op, Want));
        Changed = true;
      }
    }
  }
  return Changed;
}

// vxxLowerVitis: the Vitis lowering stage. Runs after vxxShapeIR (which has
// already produced clean, synthesizable fpga64 IR with no Xilinx intrinsics).
// This stage adds all the HLS interface/pragma semantics + signature
// disaggregation + DSP/FFT IP — i.e. what makes the IR actually synthesize.
// On its own, vxxShapeIR output gives HLS 200-70 "Cannot find any design unit
// to elaborate" (no SpecTopModule); this stage emits that and the rest.

// Normalize volatile load/store to the clang form: C++ emits volatile ops
// ONLY where the source signature says `volatile T*` (stream_better: 6 ops);
// everywhere else the a.pp.bc has plain load/store even for ap_hs/directio
// handshake ports (the protocol comes from the interface spec, not the
// volatile flag). Rust's barista APIs (ap_hs_read/write, directio reads)
// always use read/write_volatile so LLVM keeps distinct loads (one load =
// one handshake) — so at the END of lowering, strip volatile everywhere
// EXCEPT pointers rooted at a port marked `barista_hls::volatile(&p)`
// (`__vxx_volatile`, the mirror of the C++ signature qualifier), propagated
// through GEP/bitcast and into callee params (task/helper functions).
static void normalizeVolatileToCpp(Module &M) {
  SmallPtrSet<Value *, 16> Keep;
  SmallVector<CallInst *, 4> VCalls;
  if (Function *VM = M.getFunction("__vxx_volatile")) {
    for (User *VU : VM->users())
      if (auto *VC = dyn_cast<CallInst>(VU))
        VCalls.push_back(VC);
    SmallVector<Value *, 8> Work;
    for (CallInst *VC : VCalls)
      Work.push_back(VC->getArgOperand(0)->stripPointerCasts());
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      if (!Keep.insert(V).second)
        continue;
      for (User *VU : V->users()) {
        if (isa<BitCastInst>(VU) || isa<GetElementPtrInst>(VU) ||
            isa<BitCastOperator>(VU) || isa<GEPOperator>(VU)) {
          Work.push_back(VU);
        } else if (auto *CB = dyn_cast<CallInst>(VU)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee || Callee->isDeclaration())
            continue;
          for (unsigned ai = 0; ai < CB->arg_size() &&
                                ai < Callee->arg_size(); ++ai)
            if (CB->getArgOperand(ai) == V)
              Work.push_back(Callee->getArg(ai));
        }
      }
    }
    for (CallInst *VC : VCalls)
      VC->eraseFromParent();
    if (VM->use_empty())
      VM->eraseFromParent();
  }
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->isVolatile() && !Keep.count(LI->getPointerOperand()) &&
              !Keep.count(LI->getPointerOperand()->stripPointerCasts()))
            LI->setVolatile(false);
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->isVolatile() && !Keep.count(SI->getPointerOperand()) &&
              !Keep.count(SI->getPointerOperand()->stripPointerCasts()))
            SI->setVolatile(false);
        }
      }
    }
  }
}

extern "C" void vxxLowerVitis(LLVMModuleRef MRef) {
  Module &M = *llvm::unwrap(MRef);

  // Phase 3: kernel param shape (array decay, rename for HLS conventions).
  decayKernelArrayParams(M);
  // Lower llvm.memcpy(arg, arg, K) -> per-element loop. Must run after
  // decayKernelArrayParams (we need the `[N x T]*` shape on args) and
  // before any ap_fifo SpecInterface injection so HLS scheduler sees
  // a clean write-only loop.
  lowerKernelMemcpyToLoop(M);
  // lowerApFifoLoadStoreToIfStream — disabled: even with DATAFLOW pragma
  // + helper-call pattern, the HLS back-end still auto-emits
  // `SpecInterface(arg, "ap_auto")` on top stream args → SYN 201-504
  // conflict with our `SpecInterface(arg, "ap_fifo")`. The auto-emit is
  // avoided for hls::fft<>, which is a HLS-known library call (registered
  // with HLS's interface inference); our generic helper isn't recognized.
  // True fix: implement `barista-hls::fft` IP-wrap with HLS-recognized
  // call signature. (definition removed — reference-only, never called.)
  renameArrayOfStreamArgs(M);
  narrowFftComplexArgs(M);
  // FIR IP: retype ApFixed/ApUint args to the C++ proxy plain-iW form.
  narrowFirApFixedArgs(M);
  renameFftComplexArgs(M);
  renameFftSsrArgs(M);
  renameDirectioArgs(M);
  renameApScalarArgsToSsdm(M);
  renameAndStripBurstMaxi(M);
  renameAxisDisabledToCanonical(M);
  renameStreamWrapperToCanonical(M);
  retypeAxisStreamToNested(M);
  stripTopKernelNoalias(M);

  // Phase 4: Stream<T> read/write lowering to FIFO intrinsics.
  // hls::merge/split components first: their struct locals become the
  // per-channel stream allocas + SpecNPortChannel of the C++ pre-reflow form.
  injectNPortChannel(M);
  injectStreamOfBlocks(M);
  // cpp_proxy cosim (`__vxx_axis_packed`): globally strip the AXIS packet struct's
  // `[0 x T]` ZSA fields BEFORE the stream rewrites. This replaces `%Packet`
  // everywhere at once (stream element + local `tmp` alloca + GEPs) with a
  // consistent pad-free struct, so the HLS backend neither rejects the
  // zero-size fields (214-309) nor sees a per-alloca type that diverges from
  // the stream element (which would break the prototype check, 214-136).
  if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    stripZeroSizeArrayStructFields(M);
  rewriteStreamReads(M);
  rewriteStreamWrites(M);
  splitMergedFifoPushes(M);
  rewriteFifoStructPtrBitcasts(M);
  // cpp_proxy cosim (`__vxx_axis_packed`): retype the now-finalized fifo.pop/push.iN
  // on AXIS/struct streams to struct-typed, matching the C++ hls::stream<Struct>
  // element (clears HLS 214-136 at cosim_top -> rust_<top>). No-op otherwise.
  retypeAxisPackedFifo(M);
  // Must run AFTER all fifo.pop/push are created AND finalized
  // (splitMergedFifoPushes / rewriteFifoStructPtrBitcasts) so BOTH read and
  // write are caught — wraps them in C++-faithful axis.pop/push helper fns
  // (`__vxx_axis_packed` side-channel emission).
  wrapAxisReadWriteInHelpers(M);
  // cpp_proxy cosim: wrapAxisReadWriteInHelpers leaves per-channel 14-arg
  // `llvm.fpga.axis.pop/push` (one src/dst pointer per side-channel) inside the
  // hlsaxis_read/write helpers. The cosim_top port is a whole packed
  // `hls::stream<ap_axis>`, so a per-channel read trips XFORM 203-801
  // ("data pack only on source/destination"). Collapse the per-channel form
  // back to a single whole-struct `fifo.pop/push.<axis>` matching the packed
  // port. Must run AFTER wrapAxisReadWriteInHelpers.
  rewriteAxisPopToWholeStruct(M);
  // cpp_proxy cosim: plain repr(C) packet streams (ap_axis_user!) reach the
  // generic struct-fifo path with `[1 x i8]` align-pad arrays inside the stream
  // element; reflow's array-to-stream check SIGSEGVs on them. Re-pop/push as an
  // array-free clone. Skips canonical hls::axis streams (the ap_axis! path).
  cleanStructFifoArrays(M);
  wrapKpnStreamArgs(M);
  wrapKpnTaskCalls(M);
  injectKpnLocalStream(M);
  // Eliminate Rust's intermediate `let` memcpy-bridge allocas before
  // they reach HLS SROA (Pkt-typed bridges otherwise trigger HLS 214-211
  // OOB on field 0 when source memcpy uses iN packed form).
  collapseMemcpyBridgeAllocas(M);
  // Replace `store iN V, %iN_alloca ; memcpy(%pkt, %iN_alloca, N)` with
  // field-wise stores into %pkt. HLS 214-211 fires on any memcpy whose
  // source is an iN alloca (HLS SROA can't reconcile the byte-level
  // memcpy with the struct field layout). Field-wise stores avoid the
  // iN aliasing entirely.
  inlineIntMemcpyAsFieldStores(M);

  // Phase 5: top function attributes.
  injectKernelTopAttribute(M);
  // Re-pack rustc ScalarPair by-value struct args (`i64 %x.0, i64 %x.1`) into
  // the single packed integer C++ clang emits (`i128 %x`) — must run before
  // the Phase 6 SpecInterface/SpecBitsMap emitters see the args. Uncalled top
  // kernels only (cpp_proxy adapter calls keep the ScalarPair ABI).
  packScalarPairByValueArgs(M);
  // `#pragma HLS disaggregate` on struct-array ports: split into per-field
  // args named `<arg>_<field>` (the C++ pre-reflow form) and re-point the
  // per-port `__vxx_axis` markers — before Phase 6 consumes them.
  disaggStructArrayKernelSig(M);
  // AUTOMATIC struct-ref port disaggregation (pragma-less, like the C++
  // frontend): split `A*` into per-field args; stream members become
  // element pointers + a `__vxx_top_stream_param` for renameTopStreamArgs.
  autoDisaggStructRefKernelSig(M);
  // array-of-streams params (`&mut [Stream<T>; M]`): flatten to M
  // per-stream args like the C++ frontend does for `hls::stream<T> p[M]`.
  splitStreamArrayKernelSig(M);
  injectTopPipelineAttribute(M);
  injectDataflowAttribute(M);
  injectDataflow(M);
  injectMtTask(M);
  // decayDataflowHelperFns: turned out unnecessary — DATAFLOW kernels can
  // keep `[N x T]*` args and fastcc helpers; the HLS back-end does the
  // decay. The real missing pieces are SpecTopModule + per-arg
  // SpecInterface + SpecBitsMap + a 3-arg SpecDataflowPipeline. Kept the
  // helper pass around as no-op (its dso_local/linkage strip doesn't
  // regress anything).
  decayDataflowHelperFns(M);
  // token-chain s_axilite port(Return) builder → legacy return-control markers,
  // BEFORE injectAutorestart / injectSAxiliteSideeffect consume them.
  injectSAxiliteReturnChain(M);
  injectApCtrlNoneAttribute(M);
  injectApCtrlChainAttribute(M);
  injectAutorestart(M);
  injectInline(M);
  injectLatencyChain(M); // token-chain latency() builder → legacy __vxx_latency
  injectLatency(M);
  injectFunctionAllocation(M);
  // injectFunctionInstantiate is called later (emits the op-bundle late so no
  // signature-rewrite pass strips it before reflow).
  injectOccurrence(M);

  // Stamp fpga.decayed.dim.hint on undecayed [N x T]* m_axi array args
  // (e.g. coeff `[256 x i8]*` → "256"), which the HLS back-end needs for
  // burst inference on the typed arg.
  stampUndecayedArrayMAxiDimHint(M);

  // Phase 6: m_axi + AXIS + stream + array interface specs.
  injectMAxiSideeffect(M);
  injectMAxiSpecInterface(M);
  injectMAxiCache(M);  // cluster: m_axi cache pragma → SpecMAXICache
  injectCacheChain(M); // token-chain cache() builder → same xlx_cache bundle
  injectMaxiConfig(M);
  injectMaxiIntrinsics(M);
  injectMemoryInterface(M);
  injectSimplePtrBundle(M);
  injectStreamInterface(M);
  injectStreamChain(M); // token-chain stream() builder → legacy __vxx_stream_depth
  injectStreamDepth(M);
  // Stream::empty() → inverted llvm.fpga.fifo.not.empty (after the stream
  // args are retyped so the probe lands on the canonical stream object).
  injectStreamEmpty(M);
  // Retype aggregate!+ap_fifo struct-array ports to the C++ ld.5 shape
  // BEFORE injectApFifo (which skips struct arrays) consumes the markers.
  retypeAggregateFifoStructPort(M);
  injectApFifo(M);
  fixFftSsrOvfloSpecStreamDepth(M);
  injectApScalar(M);
  injectApHsValid(M);
  // Emit ap_auto on unmarked integer scalar args of TOP kernels with
  // stream-marker args + void return. 4-gate narrow:
  //   1. has at least one marked arg (FFT pattern, not rtl_as_blackbox)
  //   2. is void return (not auto_disagg's `dut -> i64`)
  //   3. integer scalar arg type (not pointer)
  //   4. arg not already marked
  injectDefaultApAutoSpec(M);
  normalizeApScalarSpecInterface(M);
  injectSAxiliteSideeffect(M);
  injectArrayPartition(M);
  injectArrayReshape(M);
  injectArrayAxis(M);
  splitCombinedArrayGeps(M); // dst[y][x] combined GEP -> clang 2-step (m_axi write burst)
  injectArrayStencil(M);
  injectArrayViewScope(M);
  // injectArrayAxisSpec handles single-channel `__vxx_axis(arr)` (paired with
  // addrspace 0 decay in decayKernelArrayParams). Same path as multi-channel
  // disagg markers. Emitted as an op-bundle (xlx_axis 6-arg op-bundle, matches
  // OSS Clang CGXlxAttr.cpp:1786): the op-bundle is the canonical form and
  // the HLS back-end converts it to SpecInterface(axis) natively. Direct
  // SpecInterface(axis) emit instead would layer on top of the back-end's
  // auto-added SpecInterface(ap_auto), creating 4 SpecInterface entries →
  // cosim TB generates an extra input-side capture port for "ap_auto" → PC
  // stage reads non-existent rtl.<top>.autotvout_A.dat for an input-only arg →
  // 212-361 "Error on TV file".
  injectArrayAxisSpec(M);
  injectAxisDisaggSpecInterface(M);
  injectAxisUserDisaggSpecInterface(M);
  injectAxis7chDisaggSpecInterface(M);
  // Drop any SpecInterface(ap_memory/...) left on an xlx_axis arg
  // (order-independent conflict cleanup → fixes XFORM 203-801).
  stripSpecInterfaceOnAxisArgs(M);
  injectAggregate(M);
  injectFence(M);
  injectAlias(M);
  injectAliasPair(M);
  // injectSpecTopModuleNarrow disabled — the HLS back-end ALWAYS emits
  // SpecTopModule based on the `set_top <name>` TCL command. Emitting it
  // here creates a duplicate (using_C++_templates). Letting the back-end
  // handle it alone is the correct path.
  // (definition removed — reference-only, never called.)

  // Phase 7: numerical/IP intrinsics.
  injectApFloatIntrinsics(M);
  // Fully unroll DSP-cascade tap loops FIRST (the C++ input arrives unrolled),
  // so each tap becomes its own marker call site with its own state alloca.
  unrollDspCascadeLoops(M);
  injectDspCplxIntrinsics(M);
  injectDsp58Intrinsics(M);
  // Dissolve the shared R out-alloca the unroller left behind (see the pass
  // comment): must run AFTER the marker lowering removed the escaping uses.
  cleanupDspCascadeAllocas(M);
  // Pack {i32,i32}/{i64,i64} DSPCPLX complex ports to the C++ pp i36/i116
  // form (Stage A port widths). After the marker lowering so the DSP
  // intrinsic gate is in place; only touches uncalled top functions.
  narrowDspCplxPortTypes(M);
  injectFftSsrIp(M);
  replaceFptosiWithHlsHelper(M);

  // Phase 8: loop pragmas.
  injectLoopUnroll(M);
  injectApWait(M);
  injectLoopPipelineMetadata(M);
  injectLoopPipelineChain(M); // token-chain pipeline() builder → same metadata
  // User loop labels ('SUM_LOOP: in the Rust source, via __vxx_loop_name)
  // BEFORE the auto-namer so the label wins over synthetic VITIS_LOOP_N.
  injectLoopUserName(M);
  injectAutoLoopName(M);
  injectLoopFlattenChain(M); // token-chain loop_flatten() builder → legacy marker
  injectLoopFlatten(M);
  // Explicit `#pragma HLS LOOP_TRIPCOUNT` (barista_hls::loop_tripcount) MUST run
  // LAST. Empirically, when it ran before injectAutoLoopName the user's explicit
  // trip count on a *runtime-bounded* loop got replaced by an SCEV-style
  // max-trip-count (i32-size/BUFFER = 2097152 on lmem_2rw) by the loop-ID rebuild
  // in injectAutoLoopName; constant-bounded loops (perfect_loop) were unaffected.
  // Running last makes the user's explicit min/max the final word — which is the
  // entire point of the pragma — and verified-correct on lmem_2rw (5 loops) +
  // perfect_loop. The loop NAME that injectAutoLoopName added is preserved
  // because injectLoopTripCount copies the existing !llvm.loop operands.
  injectLoopTripCount(M);
  injectLoopTripCountChain(M); // token-chain loop_tripcount() builder → same metadata
  // `#pragma HLS performance target_ti=` — emit SpecPerformance in the loop body.
  injectPerformance(M);
  // `#pragma HLS FUNCTION_INSTANTIATE variable=` — emit the fpga.func.instantiate
  // op-bundle so reflow specialises the function per constant call-site value.
  injectFunctionInstantiate(M);
  // After all loop-md emitters have built the final !llvm.loop nodes, append
  // llvm.loop.mustprogress on stencil loops (the emitters rebuild the node, so
  // this must run last). See stampStencilLoopMustProgress for rationale.
  stampStencilLoopMustProgress(M);
  // NOTE: synthesizeStencilLoopDebugLoc (debug source-loc synthesis) was tried
  // here to make the HLS back-end keep the outlined stencil pipeline leaf as a
  // module, but rustc strips all debug metadata after VXXPrep returns for the
  // fpga64 target (even -C debuginfo=2 yields 0 DILocations; the synthesized
  // CU/SP/locations are dropped, only the module flag survives). The flow is
  // also debug-hostile downstream (the LLVM-11→7 downgrader strips
  // spFlags/retainedNodes and the back-end ADCE segfaults on DI scope chains —
  // see the SetCurrentDebugLocation(DebugLoc()) call elsewhere). So the loop
  // source-location path is blocked without a Rustc-HLS change. Left disabled;
  // see tickets/_crashers_remaining.md.
  injectVarReset(M);

  // Un-merge select-of-stream-ptr fifo pops/pushes into per-case branches
  // (C++ pp keeps one IfRead per case; HLS cannot trace a FIFO through a
  // select). After all stream rewrites so every fifo op is final.
  // fixpoint: a >2-way dynamic stream selection lowers to a NESTED select
  // chain (array-of-streams) — each round peels one level.
  for (int SSI = 0; SSI < 8 && splitSelectPtrFifoOps(M); ++SSI) {
  }

  // Phase 9: scope-pair / protocol / pragma-only sideeffects.
  injectScopePair(M);
  injectProtocolScope(M);
  injectPragmaSideeffects(M);

  // Phase 10: final cleanup.
  // Strip `[0 x i32]` ZSA padding from any surviving local
  // `class.hls::stream<hls::axis<...>>` wrapper alloca (ap_hs datamover /
  // side_channel_data) — runs after all rename + stream-rewrite passes, so the
  // wrapper carries its canonical name and nothing still needs the padded
  // 3-field shape. Without this the HLS back-end rejects the local with
  // `214-309 unsupported array/vector field with size 0`.
  hlsrs::vxx::stripAxisStreamWrapperZsa(M);
  // volatile normalize BEFORE eraseUnimplementedMarkers (which would erase
  // the __vxx_volatile markers this pass consumes).
  normalizeVolatileToCpp(M);
  hlsrs::vxx::eraseUnimplementedMarkers(M);
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_dummy");
  dedupeSpecInterfaceNullGmem(M);
  dedupePerArgSpecInterface(M);
  // Re-run top-kernel cleanup after all marker passes have erased their
  // marker calls. This removes dead bitcasts left behind by marker arg
  // type-erasure (e.g. `arg as *const T as *const ()` → bitcast i8* to {}*).
  stripTopKernelNoalias(M);
  setupVitisModuleMetadata(M);

  // Phase 11: LLVM 7 verifier compat (run last so all prior passes can
  // assume LLVM 11 semantics).
  normalizeMemcpyAlignments(M);
  normalizeStoreLoadPointerTypes(M);
  normalizeLifetimeIntrinsicArgs(M);


  // NOTE: a late re-strip of OpenCL named metadata (rustc re-emits
  // opencl.ocl/spir.version after phase 0) was tried to stop the HLS back-end
  // treating the kernel as OpenCL (suspected force-inline of the stencil
  // pipeline leaf). It successfully removed the metadata but did NOT change the
  // stencil inline/crash behavior — so OpenCL metadata is not the keep-as-module
  // trigger. Reverted to avoid a global IR change with no benefit.

  // rustc strips ALL debug info after VXXPrep returns for the fpga64 target —
  // a bare DICompileUnit emitted here does not survive into the rustc-emitted
  // .ll. The back-end's keep-as-module decision for the stencil pipeline leaf
  // depends on debug source-locations, so debug must be injected POST-rustc
  // (build-script step) or Rustc-HLS must stop stripping it. Cannot be done
  // from this in-rustc hook.

}

// Public C ABI, called directly by rustc codegen (back/write.rs). The whole
// pipeline is compiled statically into rustc (no .so, no dlopen).

// LLVMRustVitisEarlyPrep = pre-opt rustc -O0 cleanup.
extern "C" void LLVMRustVitisEarlyPrep(LLVMModuleRef M) { vxxEarlyShape(M); }

// LLVMRustVitisPrep = the full post-opt pipeline: IR normalization (vxxShapeIR)
// then Vitis lowering (vxxLowerVitis).
extern "C" void LLVMRustVitisPrep(LLVMModuleRef M) { vxxShapeIR(M); vxxLowerVitis(M); }
