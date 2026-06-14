// vxx_passes.h — domain-pass declarations.
//
// The cohesive domain-pass groups that used to live in the monolithic
// VXXPrep.cpp were moved out into per-domain module .cpp files (maxi.cpp,
// loop.cpp, dsp_fft.cpp, dataflow_kpn.cpp, ...). Because the pass functions
// are called from the vxxLowerVitis / VXXEarlyPrep entries (in VXXPrep.cpp /
// vxx_prep.cpp) and, in a few cases, from sibling modules, their definitions
// can no longer be `static`/anonymous-namespace. They live in the `hlsrs::vxx`
// namespace and are declared here.

#ifndef VXX_PASSES_H
#define VXX_PASSES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace hlsrs { namespace vxx {

// --- maxi.cpp -----------------------------------------------------------
bool injectMAxiSideeffect(llvm::Module &M);
bool injectMAxiCache(llvm::Module &M);
bool injectCacheChain(llvm::Module &M);
bool injectMAxiSpecInterface(llvm::Module &M);
bool injectMAxiReadWriteReq(llvm::Module &M);
bool injectMaxiConfig(llvm::Module &M);
bool injectMaxiIntrinsics(llvm::Module &M);
bool injectSimplePtrBundle(llvm::Module &M);
bool renameAndStripBurstMaxi(llvm::Module &M);
bool stampUndecayedArrayMAxiDimHint(llvm::Module &M);

// --- loop.cpp -----------------------------------------------------------
bool injectLoopUnroll(llvm::Module &M);
bool injectApWait(llvm::Module &M);
bool injectLoopPipelineMetadata(llvm::Module &M);
bool injectLoopPipelineChain(llvm::Module &M);
bool injectLoopTripCount(llvm::Module &M);
bool injectLoopTripCountChain(llvm::Module &M);
bool injectLoopUserName(llvm::Module &M);
bool injectPerformance(llvm::Module &M);
bool injectFunctionInstantiate(llvm::Module &M);
bool injectDependenceSpec(llvm::Module &M);
bool injectDependenceChain(llvm::Module &M);
bool injectStableSpec(llvm::Module &M);
bool injectStableChain(llvm::Module &M);
bool injectAutoLoopName(llvm::Module &M);
bool injectLoopFlatten(llvm::Module &M);
bool injectLoopFlattenChain(llvm::Module &M);
bool injectArrayPartitionChain(llvm::Module &M);
bool injectArrayReshapeChain(llvm::Module &M);
bool injectBindStorageChain(llvm::Module &M);
bool injectArrayStencilChain(llvm::Module &M);
bool stampStencilLoopMustProgress(llvm::Module &M);
bool injectTopPipelineAttribute(llvm::Module &M);
bool sinkIntoStencilInnerLoop(llvm::Function &F, llvm::BasicBlock *StencilBB);
bool splitCombinedArrayGeps(llvm::Module &M);
bool injectArrayStencil(llvm::Module &M);
void flattenArrayGEPs(llvm::Value *Old, llvm::Value *New);

// --- dsp_fft.cpp --------------------------------------------------------
bool narrowFftComplexArgs(llvm::Module &M);
bool renameFftComplexArgs(llvm::Module &M);
bool renameFftSsrArgsImpl(llvm::Module &M, llvm::StringRef ProxyName);
bool renameFftSsrArgs(llvm::Module &M);
bool fixFftSsrOvfloSpecStreamDepth(llvm::Module &M);
bool injectFftSsrIp(llvm::Module &M);
bool injectApFloatIntrinsics(llvm::Module &M);
bool injectDspCplxIntrinsics(llvm::Module &M);
bool injectDsp58Intrinsics(llvm::Module &M);
bool narrowFirApFixedArgs(llvm::Module &M);
bool unrollDspCascadeLoops(llvm::Module &M);
bool cleanupDspCascadeAllocas(llvm::Module &M);
bool narrowDspCplxPortTypes(llvm::Module &M);
bool retypeAggregateFifoStructPort(llvm::Module &M);
bool splitSelectPtrFifoOps(llvm::Module &M);

// --- dataflow_kpn.cpp ---------------------------------------------------
bool injectDataflow(llvm::Module &M);
bool injectDataflowAttribute(llvm::Module &M);
bool decayDataflowHelperFns(llvm::Module &M);
bool injectMtTask(llvm::Module &M);
bool injectKpnLocalStream(llvm::Module &M);
bool wrapKpnStreamArgs(llvm::Module &M);
bool wrapKpnTaskCalls(llvm::Module &M);
bool injectAutorestart(llvm::Module &M);
bool injectApCtrlNoneAttribute(llvm::Module &M);
bool injectApCtrlChainAttribute(llvm::Module &M);

// --- still defined in VXXPrep.cpp, but needed by the moved passes -------
// `getOrCreateCStrGlobal` (private-string-constant helper) and
// `resolveMarkerArg` (marker first-arg → kernel Argument resolver) are generic
// helpers used by BOTH the moved domain passes and many passes that remain in
// VXXPrep.cpp. Their definitions were moved to maxi.cpp (`namespace hlsrs::vxx`)
// — they could not be re-namespaced in place because they sat in VXXPrep.cpp's
// outer anonymous namespace (which would make them `(anonymous)::hlsrs::vxx`). The
// unqualified callers that remain in VXXPrep.cpp resolve to the single
// definition via the file-scope `using namespace hlsrs::vxx;`.
llvm::GlobalVariable *getOrCreateCStrGlobal(llvm::Module &M, llvm::StringRef Text);
llvm::Argument *resolveMarkerArg(llvm::Value *V);

// ===========================================================================
// remaining domain-pass modules carved out of VXXPrep.cpp.
//
// Only the functions that are referenced from OUTSIDE their home module (the
// `vxxLowerVitis`/`VXXEarlyPrep` entries in VXXPrep.cpp, or a sibling module via the
// entangled `injectStreamInterface` orchestrator) are declared here. Purely
// file-local helpers (e.g. findLoadAfter / packStructStreamAlloca /
// getApIntWrapper / collectLeafFields / emitAxisDisaggSpec) keep their
// definitions private to their .cpp and are NOT declared. As above, the
// definitions are no longer `static` — they live in the `hlsrs::vxx` namespace and
// resolve across translation units via this header + `using namespace hlsrs::vxx;`.
// ===========================================================================

// --- axilite_mem.cpp ----------------------------------------------------
bool injectApFifo(llvm::Module &M);
bool injectApScalar(llvm::Module &M);
bool injectBindStorageSpecResource(llvm::Module &M);
bool injectDefaultApAutoSpec(llvm::Module &M);
bool injectMemoryInterface(llvm::Module &M);
bool injectSAxiliteSideeffect(llvm::Module &M);
// phase-2 B3: token-chain interface().mode(s_axilite/...) reassembly.
bool injectSAxilitePortChain(llvm::Module &M);
bool injectSAxiliteReturnChain(llvm::Module &M);
// injectSpecTopModuleNarrow / lowerApFifoLoadStoreToIfStream were
// never-called disabled passes; definitions removed.

// --- struct_layout.cpp --------------------------------------------------
bool collapseMemcpyBridgeAllocas(llvm::Module &M);
bool decayKernelArrayParams(llvm::Module &M);
bool disaggCompletePartitionKernelSig(llvm::Module &M);
bool disaggStructArrayKernelSig(llvm::Module &M);
bool dissolveSingleFieldStructs(llvm::Module &M);
bool autoDisaggStructRefKernelSig(llvm::Module &M);
bool splitStreamArrayKernelSig(llvm::Module &M);
bool packScalarPairByValueArgs(llvm::Module &M);
bool injectAggregate(llvm::Module &M);
bool injectArrayPartition(llvm::Module &M);
bool injectArrayReshape(llvm::Module &M);
bool injectArrayViewScope(llvm::Module &M);
bool inlineIntMemcpyAsFieldStores(llvm::Module &M);

// --- stream_fifo.cpp ----------------------------------------------------
bool fnHasAxisRMWBody(llvm::Function &F);
bool injectApHsValid(llvm::Module &M);
bool injectStreamChain(llvm::Module &M);
bool injectStreamDepth(llvm::Module &M);
bool injectStreamEmpty(llvm::Module &M);
bool injectNPortChannel(llvm::Module &M);
bool injectStreamOfBlocks(llvm::Module &M);
bool injectStreamInterface(llvm::Module &M);
bool rewriteFifoStructPtrBitcasts(llvm::Module &M);
bool rewriteStreamReads(llvm::Module &M);
bool rewriteStreamWrites(llvm::Module &M);
bool splitMergedFifoPushes(llvm::Module &M);
// cpp_proxy cosim only (`__vxx_axis_packed` marker): retype `fifo.pop/push.iN` on a
// `class.hls::stream<StructT>*` to struct-typed, so the synthesized arg matches
// the C++ `hls::stream<Struct>` element (clears HLS 214-136). Marker-independent.
bool retypeAxisPackedFifo(llvm::Module &M);
// cpp_proxy cosim only (`__vxx_axis_packed`): convert per-channel
// `llvm.fpga.axis.pop/push` to a whole-struct `fifo.pop/push.<axis>` so the
// read/write matches the cosim_top AXIS port (clears XFORM 203-801).
bool rewriteAxisPopToWholeStruct(llvm::Module &M);
// cpp_proxy cosim only (`__vxx_axis_packed`): re-pop/push plain `#[repr(C)]` packet
// streams (e.g. ap_axis_user! `%Packet`) as an array-free clone (`[1 x i8]` ->
// `i8`), so reflow's array-to-stream EntiretyAccessCheck does not SIGSEGV
// (HLS 200-1715). Skips canonical `hls::axis<...>` streams.
bool cleanStructFifoArrays(llvm::Module &M);
// Global ZSA strip: replace rustc repr(C) structs carrying `[0 x T]`
// alignment-marker / trailing-pad fields with pad-free equivalents (alloca +
// GEP + type uses), so the HLS backend does not reject the zero-size array
// fields (HLS 214-309). Defined in vxx_prep.cpp.
bool stripZeroSizeArrayStructFields(llvm::Module &M);

// --- axis.cpp -----------------------------------------------------------
bool collapseScalarCopyBridge(llvm::Module &M);
bool convertPaddedAxisAllocasToNoPad(llvm::Module &M);
bool disaggAxisStructKernelSig(llvm::Module &M);
llvm::StructType *getNoPadAxisStruct(llvm::Module &M, llvm::StructType *AxisST);
bool injectArrayAxis(llvm::Module &M);
// injectAxisBodyRewrite / injectAxisSpecInterfaceTop were never-called
// disabled passes; definitions removed.
bool injectArrayAxisSpec(llvm::Module &M);
bool injectAxis7chDisaggSpecInterface(llvm::Module &M);
bool injectAxisDisaggSpecInterface(llvm::Module &M);
bool injectAxisUserDisaggSpecInterface(llvm::Module &M);
bool renameStreamWrapperToCanonical(llvm::Module &M);
bool retypeAxisStreamToNested(llvm::Module &M);
bool stripAxisStreamWrapperZsa(llvm::Module &M);
bool rewriteAxisPopPushToLoadStore(llvm::Module &M);
bool scalarizeAxisWideLoads(llvm::Module &M);
bool simplifyAxisI96Body(llvm::Module &M);
bool stripSpecInterfaceOnAxisArgs(llvm::Module &M);
bool wrapAxisReadWriteInHelpers(llvm::Module &M);

} } // namespace hlsrs::vxx

#endif // VXX_PASSES_H
