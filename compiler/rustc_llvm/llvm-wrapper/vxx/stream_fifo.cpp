//===----------------------------------------------------------------------===//
//
// stream_fifo.cpp — hls::stream<T> FIFO read/write rewrite + stream interface passes.
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <string>

using namespace llvm;
using namespace hlsrs::vxx;

namespace hlsrs { namespace vxx {

// Collapse a scalar (iN) copy-bridge alloca: `%v = load iN, %P ; store iN %v,
// %Q` where %Q is an alloca written ONLY by that store. Then *%Q == *%P for all
// later reads (P = an axis.pop dest temp, written before the load and not
// after), so RAUW %Q → %P and drop the store + (now-dead) load. This kills the
// rustc read-result i96 bridge that InstCombine can't forward (the source is
// behind the opaque axis.pop). Gated by `__vxx_axis_packed`.
// Returns true if F carries an AXIS read/modify/write body whose i96 round-trip
// we want to collapse: either the packed `llvm.fpga.axis.*` form, or the default
// disagg `_ssdm_op_IfRead.Stream` / `IfWrite.Stream` form. General predicate,
// shared by the i96-cracking helpers below (no per-example matching).
bool fnHasAxisRMWBody(Function &F) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *C = CI->getCalledFunction()) {
          StringRef N = C->getName();
          if (N.startswith("llvm.fpga.axis.") ||
              N.startswith("_ssdm_op_IfRead.Stream") ||
              N.startswith("_ssdm_op_IfWrite.Stream"))
            return true;
        }
  return false;
}

// Walk forward from each marker call, looking for the next load/store in
// the same basic block — this is what the Stream::read / write helper
// emits immediately after the marker once inlined into a caller.
// Operating per-call-site is essential once Stream methods are
// `#[inline(always)]`: each inlined copy has its own marker + load/store
// pair, and a "first load in function" heuristic would only touch one of
// them.
// Walk forward from `Marker` through the linear control-flow chain
// (current BB + any single-successor descendants) looking for the first
// load. `read_volatile` and the `__vxx_stream_*` markers are MIR-level
// function calls, so rustc emits a separate continuation BB after each;
// pre-opt the marker's BB ends with a `br` and the volatile load lives
// in the next BB. We stop at branches/returns since past that point any
// load is no longer guaranteed to come from the stream pointer.
LoadInst *findLoadAfter(CallInst *Marker) {
  BasicBlock *BB = Marker->getParent();
  Instruction *Cur = Marker->getNextNode();
  while (true) {
    for (; Cur; Cur = Cur->getNextNode())
      if (auto *LI = dyn_cast<LoadInst>(Cur))
        return LI;
    Instruction *Term = BB->getTerminator();
    if (!Term || Term->getNumSuccessors() != 1)
      return nullptr;
    BB = Term->getSuccessor(0);
    Cur = &BB->front();
  }
}

StoreInst *findStoreAfter(CallInst *Marker) {
  BasicBlock *BB = Marker->getParent();
  Instruction *Cur = Marker->getNextNode();
  while (true) {
    for (; Cur; Cur = Cur->getNextNode())
      if (auto *SI = dyn_cast<StoreInst>(Cur))
        return SI;
    Instruction *Term = BB->getTerminator();
    if (!Term || Term->getNumSuccessors() != 1)
      return nullptr;
    BB = Term->getSuccessor(0);
    Cur = &BB->front();
  }
}

// `core::intrinsics::volatile_store(dst, value)` lowers to either a
// `store volatile` instruction (for primitive types) or a volatile
// `llvm.memcpy` (for composite types passed indirectly). Walk the linear
// successor chain after `Marker` and return the first match of either
// shape — the rewriter handles them uniformly.
Instruction *findStoreOrVolatileMemcpyAfter(CallInst *Marker) {
  BasicBlock *BB = Marker->getParent();
  Instruction *Cur = Marker->getNextNode();
  while (true) {
    for (; Cur; Cur = Cur->getNextNode()) {
      if (isa<StoreInst>(Cur))
        return Cur;
      if (auto *MC = dyn_cast<MemCpyInst>(Cur))
        if (MC->isVolatile())
          return MC;
    }
    Instruction *Term = BB->getTerminator();
    if (!Term || Term->getNumSuccessors() != 1)
      return nullptr;
    BB = Term->getSuccessor(0);
    Cur = &BB->front();
  }
}

// Convert a non-integer FIFO element type (struct/array/etc.) into an iN
// of the same bit width. The Stream<T>::read/write rewrites have to use an
// integer-typed FIFO intrinsic (`llvm.fpga.fifo.{pop,push}.iN.p0iN`); any
// caller-visible struct value is recovered by bitcasting the iN through an
// alloca. SROA can then collapse the alloca round-trip when safe.
// Pre-scan: collect AXIS-marked stream pointers (kernel-arg ptrs passed to
// `__vxx_axis(stream_ptr)` marker). Used by `rewriteStreamReads`/`Writes`
// to choose between fifo.pop/push (default) and the per-field axis.pop/push
// intrinsic. Bitcasts are unwound so the same alloca/argument is keyed
// regardless of intermediate type casts.
DenseSet<Value *> collectAxisMarkedStreams(Module &M) {
  DenseSet<Value *> Marked;
  Function *AxisMarker = M.getFunction("__vxx_axis");
  if (!AxisMarker) return Marked;
  for (User *U : AxisMarker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Value *P = CI->getArgOperand(0);
    while (auto *BC = dyn_cast<BitCastInst>(P)) P = BC->getOperand(0);
    while (auto *CE = dyn_cast<ConstantExpr>(P)) {
      if (CE->getOpcode() != Instruction::BitCast) break;
      P = CE->getOperand(0);
    }
    Marked.insert(P);
  }
  // Transitively propagate AXIS-ness across call-site arg passing.
  // If `f(axis_arg)` is called and `axis_arg ∈ Marked`, then the
  // callee's corresponding Argument is also AXIS-marked. Repeat until
  // fixpoint to handle helper chains: @example → __axis_helper_read →
  // Stream<T>::read. Without this, the load inside Stream::read sees
  // its own `%self` Argument which is never directly marked, only the
  // kernel's `%a` is — and the existing matching is per-function.
  bool ChangedTrans = true;
  while (ChangedTrans) {
    ChangedTrans = false;
    SmallVector<Value *, 8> ToAdd;
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI) continue;
          Function *Callee = CI->getCalledFunction();
          if (!Callee || Callee->isDeclaration()) continue;
          unsigned NArgs = std::min<unsigned>(CI->arg_size(), Callee->arg_size());
          auto ArgIt = Callee->arg_begin();
          for (unsigned i = 0; i < NArgs; ++i, ++ArgIt) {
            Value *Op = CI->getArgOperand(i);
            while (auto *BC = dyn_cast<BitCastInst>(Op)) Op = BC->getOperand(0);
            while (auto *CE = dyn_cast<ConstantExpr>(Op)) {
              if (CE->getOpcode() != Instruction::BitCast) break;
              Op = CE->getOperand(0);
            }
            if (Marked.count(Op) && !Marked.count(&*ArgIt)) {
              ToAdd.push_back(&*ArgIt);
            }
          }
        }
      }
    }
    if (!ToAdd.empty()) {
      for (Value *V : ToAdd) Marked.insert(V);
      ChangedTrans = true;
    }
  }
  return Marked;
}

// For AXIS-marked struct streams, emit `llvm.fpga.axis.pop`/`push` with
// per-field pointer args (stream-side GEPs + temp-side GEPs) instead of the
// single struct-typed `fpga.fifo.pop`/`push`. Returns true on successful
// rewrite. The struct must have ≥2 fields (per-AXIS-channel decomposition).
//
// Layout: pop(stream.f0, stream.f1, ..., stream.fN, temp.f0, ..., temp.fN)
// — first N args are read-from (stream-side), last N are written (temp-side).
// The intrinsic is overloaded by the N field pointer types. Mirrors
// `__fpga_axis_pop` builtin (CGBuiltin.cpp:976-998).

// Compute the AXIS axis.pop/push per-field argument layout for a stream
// element struct ElemST: for each real channel, its packed pointer type
// (appended to FieldPtrTys) and its struct field index (appended to RealIdxs).
// Skips fields that carry no AXIS sub-channel:
//   - rustc's `[0 x ?]` zero-size-array padding and trailing `[N x i8]`
//     alignment pads (embedding them in the intrinsic name makes the decl
//     signature unstable across strip-padding ordering);
//   - `hls::axis_disabled_signal` 0-width channels (disagg already drops them
//     from the kernel args, and the HLS backend's CDFG SIGSEGVs on
//     disabled-signal operands).
// Scalar fields (int/ptr/float) pass through as-is; struct/array fields are
// packed into an iN matching their alloc size (e.g. ComplexShort -> i32).
// Shared by the read / write / memcpy stream-rewrite paths.
static void computeAxisPackedFieldPtrs(Module &M, StructType *ElemST,
                                       SmallVectorImpl<Type *> &FieldPtrTys,
                                       SmallVectorImpl<unsigned> &RealIdxs) {
  const DataLayout &DL = M.getDataLayout();
  for (unsigned i = 0; i < ElemST->getNumElements(); ++i) {
    Type *FT = ElemST->getElementType(i);
    Type *PackedTy = nullptr;
    if (FT->isIntegerTy() || FT->isPointerTy() || FT->isFloatingPointTy()) {
      PackedTy = FT;
    } else if (auto *InnerST = dyn_cast<StructType>(FT)) {
      if (hlsrs::vxx::isAxisDisabledType(InnerST))
        continue;
      PackedTy = Type::getIntNTy(M.getContext(),
                                 DL.getTypeAllocSizeInBits(InnerST));
    } else if (auto *AT = dyn_cast<ArrayType>(FT)) {
      if (AT->getNumElements() == 0) continue;
      if (AT->getElementType()->isIntegerTy(8)) continue;
      PackedTy = Type::getIntNTy(M.getContext(), DL.getTypeAllocSizeInBits(AT));
    } else {
      continue;
    }
    FieldPtrTys.push_back(PointerType::get(PackedTy, 0));
    RealIdxs.push_back(i);
  }
}

// Nested mode: when ElemST has a pad-stripped nested clone (a
// `struct.hls::axis<ap_int<W>,...>.nest`), the axis.pop/push args must be
// the per-field STRUCT pointers (%"struct.ap_int<W>"* etc.), matching
// `llvm.fpga.axis.pop.p0s_struct.ap_int<W>s...`. Bitcasting them down to
// leaf i32*/i8* erases the struct-type info the HLS backend's AXISProcess
// solver (visitAllocaInst) reads → SIGSEGV. When such a clone exists, override
// FieldPtrTys/RealIdxs to the nested field struct ptrs. Returns the nested
// clone (== ElemST when there is no trailing-pad to strip; idempotent).
static StructType *applyAxisNestedOverride(Module &M, StructType *ElemST,
                                           SmallVectorImpl<Type *> &FieldPtrTys,
                                           SmallVectorImpl<unsigned> &RealIdxs) {
  StructType *NestST = getNoPadAxisStruct(M, ElemST);
  if (NestST != ElemST) {
    FieldPtrTys.clear();
    RealIdxs.clear();
    for (unsigned i = 0; i < NestST->getNumElements(); ++i) {
      FieldPtrTys.push_back(PointerType::get(NestST->getElementType(i), 0));
      RealIdxs.push_back(i);
    }
  }
  return NestST;
}

// Emit one per-channel GEP into BasePtr (a BaseST*) for each real field index,
// bitcast to the matching FieldPtrTys entry when needed, and append it to Args.
// Used for both the stream-side and temp/src-side argument halves of an
// axis.pop/push call.
static void appendAxisChannelGeps(IRBuilder<> &B, StructType *BaseST,
                                  Value *BasePtr, ArrayRef<unsigned> RealIdxs,
                                  ArrayRef<Type *> FieldPtrTys,
                                  SmallVectorImpl<Value *> &Args) {
  for (size_t k = 0; k < RealIdxs.size(); ++k) {
    Value *GEP = B.CreateConstGEP2_32(BaseST, BasePtr, 0, RealIdxs[k]);
    if (GEP->getType() != FieldPtrTys[k])
      GEP = B.CreateBitCast(GEP, FieldPtrTys[k]);
    Args.push_back(GEP);
  }
}

bool rewriteStreamReadAxis(LoadInst *LI, StructType *ElemST,
                                  Value *StreamPtr, Module &M) {
  if (ElemST->getNumElements() < 2) return false;
  // Stream pointer: bitcast to ElemST* if needed (post-strip Stream<T> is
  // single-field { T } so the inner T pointer == stream pointer modulo cast).
  IRBuilder<> B(LI);
  Type *ElemPtrTy = PointerType::get(ElemST, 0);
  Value *StreamElemPtr = StreamPtr;
  if (StreamElemPtr->getType() != ElemPtrTy)
    StreamElemPtr = B.CreateBitCast(StreamElemPtr, ElemPtrTy);
  // Allocate temp ElemST in entry block.
  Function *F = LI->getParent()->getParent();
  IRBuilder<> EntryB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
  // Use a pad-stripped (7-field) clone for the LOCAL temp alloca so the HLS
  // backend's AXISProcess solver doesn't SIGSEGV on the trailing [N x i8] pad
  // field (visitAllocaInst). First 7 fields are identical to ElemST so the
  // temp-side GEP indices and axis.pop pointer types are unchanged.
  StructType *TempST = getNoPadAxisStruct(M, ElemST);
  AllocaInst *TempT = EntryB.CreateAlloca(TempST, nullptr, "axis.tmp");
  // Per-field GEPs: stream-side and temp-side. Skip rustc's `[0 x ?]`
  // zero-size-array padding fields — they have no AXIS sub-channel
  // semantics and embedding them in the intrinsic name (via FieldPtrTys)
  // makes the decl signature unstable across the Strip-Padding-Before-
  // vs-After-rewriteStreamReadAxis ordering.
  SmallVector<Value *, 14> Args;
  SmallVector<Type *, 7> FieldPtrTys;
  SmallVector<unsigned, 7> RealIdxs;
  computeAxisPackedFieldPtrs(M, ElemST, FieldPtrTys, RealIdxs);
  // Override to nested-mode field-struct ptrs when ElemST has a pad-stripped
  // clone (== TempST). Stream-side GEPs then bitcast i32*/i8* up; temp-side
  // GEPs into TempST yield the field struct ptr directly.
  // The temp-side GEPs below use TempST, which equals the NestST this call
  // computes internally: getNoPadAxisStruct is cached/idempotent, so the
  // caller's earlier `TempST = getNoPadAxisStruct(M, ElemST)` is the same type.
  applyAxisNestedOverride(M, ElemST, FieldPtrTys, RealIdxs);
  appendAxisChannelGeps(B, ElemST, StreamElemPtr, RealIdxs, FieldPtrTys, Args);
  appendAxisChannelGeps(B, TempST, TempT, RealIdxs, FieldPtrTys, Args);
  Function *Pop = hlsrs::vxx::getOrInsertAxisPopOrPush(M, "pop", FieldPtrTys);
  B.CreateCall(Pop, Args);
  // Load result from temp. The original LI may have been struct-typed
  // (matches ElemST directly) OR integer-typed (rustc PassMode::Cast
  // collapsed the struct to a wide iN for the function-return ABI). For
  // the latter, bitcast the temp pointer to iN* and load iN — preserves
  // the original load's value type so all downstream uses still type-check.
  Value *NewVal;
  if (LI->getType() == ElemST || LI->getType() == TempST) {
    Value *Cast = B.CreateBitCast(TempT, PointerType::get(LI->getType(), 0));
    NewVal = B.CreateLoad(LI->getType(), Cast);
  } else if (auto *IntTy = dyn_cast<IntegerType>(LI->getType())) {
    Value *AsInt = B.CreateBitCast(TempT, PointerType::get(IntTy, 0));
    NewVal = B.CreateLoad(IntTy, AsInt);
  } else {
    // Other type: just bitcast through to LI's type.
    Value *Cast = B.CreateBitCast(TempT, PointerType::get(LI->getType(), 0));
    NewVal = B.CreateLoad(LI->getType(), Cast);
  }
  LI->replaceAllUsesWith(NewVal);
  LI->eraseFromParent();
  return true;
}

// `__vxx_axis_packed`: rewrite a struct/i96 stream load into a struct-typed
// `llvm.fpga.fifo.pop.<axisStruct>` on the packed stream arg, so the HLS
// backend sees a real stream read returning the AXIS element struct (and
// disaggregates the 7 channels itself). Bridges the result back to the load's
// original type (rustc collapses the struct return to i96).
bool rewriteStreamReadAxisPacked(LoadInst *LI, StructType *ElemST,
                                        Value *StreamPtr, Module &M) {
  IRBuilder<> B(LI);
  Type *ElemPtrTy = PointerType::get(ElemST, 0);
  Value *SElemPtr = StreamPtr;
  if (SElemPtr->getType() != ElemPtrTy)
    SElemPtr = B.CreateBitCast(SElemPtr, ElemPtrTy);
  Function *Pop = hlsrs::vxx::getOrInsertFifoPopAny(M, ElemST);
  CallInst *Val = B.CreateCall(Pop, {SElemPtr});  // ElemST by value
  Value *NewVal;
  if (LI->getType() == ElemST) {
    NewVal = Val;
  } else {
    Function *F = LI->getFunction();
    IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
    AllocaInst *Tmp = EB.CreateAlloca(ElemST);
    B.CreateStore(Val, Tmp);
    Value *C = B.CreateBitCast(Tmp, PointerType::get(LI->getType(), 0));
    NewVal = B.CreateLoad(LI->getType(), C);
  }
  LI->replaceAllUsesWith(NewVal);
  LI->eraseFromParent();
  return true;
}

// Trace a stream access pointer back to its `class.hls::stream<Struct>`
// wrapper and return the element struct type, or null. Mirrors the inline
// logic in the AXIS read/write paths (NF==1 bare `{ T }` or NF==3
// ZSA-sandwich `{ [0 x T'], T, [0 x T'] }` shapes). Used by the cpp_proxy
// cosim packed-struct fifo.pop/push (`__vxx_axis_packed`) path.
static StructType *streamElemStructOf(Value *Ptr) {
  Value *Trace = Ptr;
  while (true) {
    if (auto *BC = dyn_cast<BitCastInst>(Trace)) { Trace = BC->getOperand(0); continue; }
    if (auto *G = dyn_cast<GetElementPtrInst>(Trace)) { Trace = G->getPointerOperand(); continue; }
    break;
  }
  auto *PT = dyn_cast<PointerType>(Trace->getType());
  if (!PT) return nullptr;
  auto *StreamST = dyn_cast<StructType>(PT->getElementType());
  if (!StreamST) return nullptr;
  unsigned NF = StreamST->getNumElements();
  if (NF == 1) return dyn_cast<StructType>(StreamST->getElementType(0));
  if (NF == 3) {
    auto *F0 = dyn_cast<ArrayType>(StreamST->getElementType(0));
    auto *F2 = dyn_cast<ArrayType>(StreamST->getElementType(2));
    if (F0 && F0->getNumElements() == 0 && F2 && F2->getNumElements() == 0)
      return dyn_cast<StructType>(StreamST->getElementType(1));
  }
  return nullptr;
}


bool rewriteStreamReads(Module &M) {
  Function *Marker = M.getFunction("__vxx_stream_read_marker");
  if (!Marker)
    return false;

  const DataLayout &DL = M.getDataLayout();
  DenseSet<Value *> AxisMarked = collectAxisMarkedStreams(M);

  SmallVector<CallInst *, 8> Markers;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Markers.push_back(CI);

  vxxDbg() << "vxx: rewriteStreamReads — module=" << M.getName()
         << " AxisMarked=" << AxisMarked.size()
         << " markers=" << Markers.size() << "\n";

  bool Changed = false;
  for (CallInst *MC : Markers) {
    LoadInst *LI = findLoadAfter(MC);
    if (!LI)
      continue;
    Type *LoadTy = LI->getType();
    // AXIS struct stream path: if the stream pointer is AXIS-marked AND
    // the original (un-bitcasted) stream's inner element type is a multi-
    // field struct, emit per-field axis.pop. The actual load may be
    // integer-typed (i80, i96, ...) due to rustc's PassMode::Cast on small
    // struct returns — we detect the struct via the source Stream<T>*
    // pointee type, not the load type.
    if (!AxisMarked.empty()) {
      Value *Trace = LI->getPointerOperand();
      // Unwind bitcasts AND GEPs — the load pointer typically goes
      // through `getelementptr Stream<T>* %arg, 0, 0` to reach the
      // inner T* (Stream::read body's `&self.value`), then a bitcast
      // to iN* under PassMode::Cast. We want the original Stream<T>*.
      while (true) {
        if (auto *BC = dyn_cast<BitCastInst>(Trace)) { Trace = BC->getOperand(0); continue; }
        if (auto *G = dyn_cast<GetElementPtrInst>(Trace)) { Trace = G->getPointerOperand(); continue; }
        break;
      }
      if (AxisMarked.count(Trace)) {
        if (auto *PT = dyn_cast<PointerType>(Trace->getType())) {
          if (auto *StreamST = dyn_cast<StructType>(PT->getElementType())) {
            // Stream<T> shape can be:
            // - single-field `{ T }` (repr(transparent) collapse)
            // - ZSA-sandwich `{ [0 x T'], T, [0 x T'] }` (raw rustc kernel-arg)
            StructType *ElemST = nullptr;
            unsigned NF = StreamST->getNumElements();
            if (NF == 1) {
              ElemST = dyn_cast<StructType>(StreamST->getElementType(0));
            } else if (NF == 3) {
              // Check ZSA-sandwich shape and extract middle field.
              auto *F0 = dyn_cast<ArrayType>(StreamST->getElementType(0));
              auto *F2 = dyn_cast<ArrayType>(StreamST->getElementType(2));
              if (F0 && F0->getNumElements() == 0 &&
                  F2 && F2->getNumElements() == 0) {
                ElemST = dyn_cast<StructType>(StreamST->getElementType(1));
              }
            }
            // AXIS read lowering. Two forms:
            //  - per-field `llvm.fpga.axis.pop` (rewriteStreamReadAxis) →
            //    later lowered to per-channel struct-field load/store by
            //    rewriteAxisPopPushToLoadStore. In PACKED mode (single stream
            //    arg) the HLS backend REJECTS that with HLS 214-244 ("each array
            //    element must be accessed only once / whole element in one op")
            //    because the 7 field accesses look like 7 sub-accesses of one
            //    stream element. So per-field is only valid in disagg-arg mode.
            //  - whole-element struct `fifo.pop` (rewriteStreamReadAxisPacked):
            //    one read of the whole hls::axis element. With the
            //    `class.hls::stream<hls::axis<ap_int<W>,...>>` type now in
            //    place (renameStreamWrapperToCanonical), the HLS backend
            //    recognises the AXIS element type and disaggregates the side
            //    channels itself.
            if (ElemST) {
              if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
                // cpp_proxy cosim: the read was SROA-packed to an integer iN
                // (LoadTy), but the stream element is a struct. Emit a
                // *struct-typed* `fifo.pop.<ElemST>` so the synthesized arg
                // matches the C++ `hls::stream<Struct>` element — the iN form
                // mismatches and trips HLS 214-136 at the cosim_top ->
                // rust_<top> call. Convert the popped struct back to iN via a
                // temp alloca for the downstream integer users. Gated on
                // `__vxx_axis_packed` (cosim build only) so the Stage-A per-field
                // AXIS path and scalar streams are untouched.
                if (auto *ILT = dyn_cast<IntegerType>(LoadTy)) {
                  if (DL.getTypeAllocSizeInBits(ElemST) == ILT->getBitWidth()) {
                    IRBuilder<> B(LI);
                    Value *STPtr = B.CreateBitCast(
                        LI->getPointerOperand(), PointerType::get(ElemST, 0));
                    Function *Pop =
                        hlsrs::vxx::getOrInsertFifoPopAny(M, ElemST);
                    CallInst *PopCall = B.CreateCall(Pop, {STPtr});
                    Function *F = LI->getParent()->getParent();
                    IRBuilder<> EntryB(&F->getEntryBlock(),
                                       F->getEntryBlock().getFirstInsertionPt());
                    AllocaInst *Tmp =
                        EntryB.CreateAlloca(ElemST, nullptr, "axis.packed.read");
                    B.CreateStore(PopCall, Tmp);
                    Value *AsInt =
                        B.CreateBitCast(Tmp, PointerType::get(ILT, 0));
                    Value *IntVal = B.CreateLoad(ILT, AsInt);
                    LI->replaceAllUsesWith(IntVal);
                    LI->eraseFromParent();
                    Changed = true;
                    continue;
                  }
                }
                if (rewriteStreamReadAxisPacked(LI, ElemST, Trace, M)) {
                  Changed = true;
                  continue;
                }
              } else if (rewriteStreamReadAxis(LI, ElemST, Trace, M)) {
                Changed = true;
                continue;
              }
            }
          }
        }
      }
    }
    IntegerType *ElemTy = dyn_cast<IntegerType>(LoadTy);
    if (ElemTy) {
      IRBuilder<> B(LI);
      Function *Pop = hlsrs::vxx::getOrInsertFifoPopAny(M, ElemTy);
      // A single-field struct stream element (C++ `dat_t<T> { T data; }`)
      // loads its scalar through a still-struct-typed pointer; the C++
      // frontend flattens such wrappers to the scalar (aliasing pre-reflow:
      // `i32 as(128)* %inputStream`) — cast the pointer to match.
      Value *P = LI->getPointerOperand();
      Type *WantPT = PointerType::get(
          ElemTy, cast<PointerType>(P->getType())->getAddressSpace());
      if (P->getType() != WantPT)
        P = B.CreateBitCast(P, WantPT);
      CallInst *Call = B.CreateCall(Pop, {P});
      LI->replaceAllUsesWith(Call);
      LI->eraseFromParent();
    } else {
      // Struct-typed stream element: emit `llvm.fpga.fifo.pop.<T>.p0<T>`
      // directly with the struct type (no bitcast to iN). The earlier
      // bitcast-to-iN path triggered the HLS backend's "type conversion
      // operator" rejection on AGGREGATE-eligible streams (HLS 214-171);
      // the struct-typed intrinsic form is what the HLS backend's stream
      // analysis recognises.
      IRBuilder<> B(LI);
      Function *Pop = hlsrs::vxx::getOrInsertFifoPopAny(M, LoadTy);
      CallInst *Call = B.CreateCall(Pop, {LI->getPointerOperand()});
      LI->replaceAllUsesWith(Call);
      LI->eraseFromParent();
    }
    Changed = true;
  }
  for (CallInst *CI : Markers)
    CI->eraseFromParent();
  return Changed;
}

// AXIS counterpart to rewriteStreamReadAxis for the write side.
// Emits per-field `llvm.fpga.axis.push` (stream-side ptrs +
// temp-side ptrs holding the values being written). The intrinsic
// reads from the temp ptrs and pushes onto the stream's per-channel
// FIFOs.
bool rewriteStreamWriteAxis(StoreInst *SI, StructType *ElemST,
                                   Value *StreamPtr, Module &M) {
  if (ElemST->getNumElements() < 2) return false;
  IRBuilder<> B(SI);
  Type *ElemPtrTy = PointerType::get(ElemST, 0);
  Value *StreamElemPtr = StreamPtr;
  if (StreamElemPtr->getType() != ElemPtrTy)
    StreamElemPtr = B.CreateBitCast(StreamElemPtr, ElemPtrTy);

  // Allocate a temp ElemST in entry block and copy the stored value into it.
  Function *F = SI->getParent()->getParent();
  IRBuilder<> EntryB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
  AllocaInst *TempT = EntryB.CreateAlloca(ElemST, nullptr, "axis.tmp.write");
  // Stage the value: bitcast-or-store into the temp at SI's insertion point.
  Value *StoredVal = SI->getValueOperand();
  if (StoredVal->getType() == ElemST) {
    B.CreateStore(StoredVal, TempT);
  } else if (auto *IntTy = dyn_cast<IntegerType>(StoredVal->getType())) {
    Value *AsInt = B.CreateBitCast(TempT, PointerType::get(IntTy, 0));
    B.CreateStore(StoredVal, AsInt);
  } else {
    Value *Cast = B.CreateBitCast(TempT, PointerType::get(StoredVal->getType(), 0));
    B.CreateStore(StoredVal, Cast);
  }

  SmallVector<Value *, 14> Args;
  SmallVector<Type *, 7> FieldPtrTys;
  SmallVector<unsigned, 7> RealIdxs;
  computeAxisPackedFieldPtrs(M, ElemST, FieldPtrTys, RealIdxs);
  appendAxisChannelGeps(B, ElemST, StreamElemPtr, RealIdxs, FieldPtrTys, Args);
  appendAxisChannelGeps(B, ElemST, TempT, RealIdxs, FieldPtrTys, Args);
  Function *Push = hlsrs::vxx::getOrInsertAxisPopOrPush(M, "push", FieldPtrTys);
  B.CreateCall(Push, Args);
  SI->eraseFromParent();
  return true;
}

// Try to rewrite one AXIS stream write into a per-field axis.push: detect that
// PtrOp traces back to an AxisMarked single-field stream whose element is a
// struct, stage StoredVal through a pad-stripped temp, emit the axis.push, and
// erase ToReplace. Returns true on success. Caller supplies the right
// (ptr, instruction, value) triple for the StoreInst or fifo.push-call sink.
static bool tryRewriteAxisWrite(Module &M, const DenseSet<Value *> &AxisMarked,
                                Value *PtrOp, Instruction *ToReplace,
                                Value *StoredVal) {
  Value *Trace = PtrOp;
  while (true) {
    if (auto *BC = dyn_cast<BitCastInst>(Trace)) { Trace = BC->getOperand(0); continue; }
    if (auto *G = dyn_cast<GetElementPtrInst>(Trace)) { Trace = G->getPointerOperand(); continue; }
    break;
  }
  if (!AxisMarked.count(Trace)) return false;
  auto *PT = dyn_cast<PointerType>(Trace->getType());
  if (!PT) return false;
  auto *StreamST = dyn_cast<StructType>(PT->getElementType());
  if (!StreamST) return false;
  // The Stream wrapper is either a bare 1-field `{ axis }` or a rustc-emitted
  // ZSA sandwich `{ [0 x i32], axis, [0 x i32] }`. The read path
  // (rewriteStreamReadAxis) already handles both; mirror it here so the AXIS
  // write disaggregates to the out_r side-channel ports instead of falling
  // through to a generic fifo.push on a dead local i96 temp (the regression
  // that left out_r unconnected → HLS 214-211).
  StructType *ElemST = nullptr;
  unsigned NF = StreamST->getNumElements();
  if (NF == 1) {
    ElemST = dyn_cast<StructType>(StreamST->getElementType(0));
  } else if (NF == 3) {
    auto *F0 = dyn_cast<ArrayType>(StreamST->getElementType(0));
    auto *F2 = dyn_cast<ArrayType>(StreamST->getElementType(2));
    if (F0 && F0->getNumElements() == 0 && F2 && F2->getNumElements() == 0)
      ElemST = dyn_cast<StructType>(StreamST->getElementType(1));
  }
  if (!ElemST) return false;
  // Build axis.push at ToReplace's location, using StoredVal as the value
  // (same shape as the load case but staged through a temp so per-field GEPs
  // work).
  IRBuilder<> B(ToReplace);
  Type *ElemPtrTy = PointerType::get(ElemST, 0);
  Value *StreamElemPtr = Trace;
  if (StreamElemPtr->getType() != ElemPtrTy)
    StreamElemPtr = B.CreateBitCast(StreamElemPtr, ElemPtrTy);
  Function *F = ToReplace->getParent()->getParent();
  IRBuilder<> EntryB(&F->getEntryBlock(),
                     F->getEntryBlock().getFirstInsertionPt());
  // Pad-stripped temp so the HLS backend's AXISProcess solver doesn't SIGSEGV
  // on the trailing [N x i8] array field (visitAllocaInst).
  StructType *TempST = getNoPadAxisStruct(M, ElemST);
  AllocaInst *TempT = EntryB.CreateAlloca(TempST, nullptr, "axis.tmp.write");
  if (StoredVal->getType() == ElemST || StoredVal->getType() == TempST) {
    Value *Cast = B.CreateBitCast(TempT,
                                  PointerType::get(StoredVal->getType(), 0));
    B.CreateStore(StoredVal, Cast);
  } else if (auto *IntTy = dyn_cast<IntegerType>(StoredVal->getType())) {
    Value *AsInt = B.CreateBitCast(TempT, PointerType::get(IntTy, 0));
    B.CreateStore(StoredVal, AsInt);
  } else {
    Value *Cast = B.CreateBitCast(TempT, PointerType::get(StoredVal->getType(), 0));
    B.CreateStore(StoredVal, Cast);
  }
  SmallVector<Value *, 14> Args;
  SmallVector<Type *, 7> FieldPtrTys;
  SmallVector<unsigned, 7> RealIdxs;
  computeAxisPackedFieldPtrs(M, ElemST, FieldPtrTys, RealIdxs);
  // The temp-side GEPs below use TempST, which equals the NestST this call
  // computes internally: getNoPadAxisStruct is cached/idempotent, so the
  // caller's earlier `TempST = getNoPadAxisStruct(M, ElemST)` is the same type.
  applyAxisNestedOverride(M, ElemST, FieldPtrTys, RealIdxs);
  appendAxisChannelGeps(B, ElemST, StreamElemPtr, RealIdxs, FieldPtrTys, Args);
  appendAxisChannelGeps(B, TempST, TempT, RealIdxs, FieldPtrTys, Args);
  Function *Push = hlsrs::vxx::getOrInsertAxisPopOrPush(M, "push", FieldPtrTys);
  B.CreateCall(Push, Args);
  ToReplace->eraseFromParent();
  return true;
}

bool rewriteStreamWrites(Module &M) {
  Function *Marker = M.getFunction("__vxx_stream_write_marker");
  if (!Marker)
    return false;

  const DataLayout &DL = M.getDataLayout();
  DenseSet<Value *> AxisMarked = collectAxisMarkedStreams(M);

  SmallVector<CallInst *, 8> Markers;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Markers.push_back(CI);

  vxxDbg() << "vxx: rewriteStreamWrites — module=" << M.getName()
         << " AxisMarked=" << AxisMarked.size()
         << " markers=" << Markers.size() << "\n";

  // Unwind bitcasts/GEPs/addrspacecasts to the underlying base pointer.
  auto TraceBase = [](Value *V) -> Value * {
    while (true) {
      if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
      if (auto *AC = dyn_cast<AddrSpaceCastInst>(V)) { V = AC->getOperand(0); continue; }
      if (auto *G = dyn_cast<GetElementPtrInst>(V)) { V = G->getPointerOperand(); continue; }
      if (auto *CE = dyn_cast<ConstantExpr>(V)) {
        if (CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::GetElementPtr) { V = CE->getOperand(0); continue; }
      }
      return V;
    }
  };

  bool Changed = false;
  for (CallInst *MC : Markers) {
    Instruction *SinkInst = findStoreOrVolatileMemcpyAfter(MC);
    if (!SinkInst)
      continue;
    // Struct-element `write_volatile` goes through a stack temp: rustc first
    // stores the value into an SROA'd alloca (`_5.sroa.*`), then volatile
    // load+store copies it into the port. `findStoreOrVolatileMemcpyAfter`
    // returns that FIRST temp store — pushing onto the temp leaves the port
    // as a plain volatile store (an RTL stream with no producer → cosim
    // deadlock). Prefer the store whose dest traces to the marker hint's
    // base (`&self.value` — the real stream); fall back to the first store.
    if (auto *SI0 = dyn_cast<StoreInst>(SinkInst)) {
      if (MC->arg_size() > 0) {
        Value *HintBase = TraceBase(MC->getArgOperand(0));
        if (isa<Argument>(HintBase) || isa<GlobalValue>(HintBase) ||
            isa<CallInst>(HintBase)) {
          if (TraceBase(SI0->getPointerOperand()) != HintBase) {
            BasicBlock *BB = SI0->getParent();
            Instruction *Cur = SI0->getNextNode();
            Instruction *Better = nullptr;
            while (!Better) {
              for (; Cur; Cur = Cur->getNextNode()) {
                if (auto *SI = dyn_cast<StoreInst>(Cur)) {
                  if (TraceBase(SI->getPointerOperand()) == HintBase) {
                    Better = SI;
                    break;
                  }
                }
              }
              if (Better) break;
              Instruction *Term = BB->getTerminator();
              if (!Term || Term->getNumSuccessors() != 1) break;
              BB = Term->getSuccessor(0);
              Cur = &BB->front();
            }
            if (Better)
              SinkInst = Better;
          }
        }
      }
    }
    // AXIS struct stream path: same shape detection as in rewriteStreamReads.
    // SinkInst can be either a StoreInst (initial run, EarlyPrep) OR a
    // `fpga.fifo.push` call (LatePrep, EarlyPrep already converted on
    // a prior pass). Handle both — for the call case, undo the fifo.push
    // and emit per-field axis.push instead.
    // PACKED mode skips the per-field axis.push (it lowers to per-channel
    // load/store which the HLS backend rejects with HLS 214-244 on a single
    // packed stream arg). Packed write falls through to the whole-element
    // struct fifo.push below, matching the whole-element read.
    if (!AxisMarked.empty() &&
        (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))) {
      // The write marker's hint argument is `&self.value` (a bitcast/GEP of the
      // out_r stream pointer), so it reliably identifies the AXIS stream even
      // when SROA + inlining of Stream::write/write_volatile has rewritten the
      // actual store's pointer to a dead local i96 temp (the regression that
      // left out_r unconnected → generic fifo.push on a local → HLS 214-211).
      // Recover the stream from the marker and use it as the axis.push target;
      // the sink only supplies the value.
      Value *MarkerStream = MC->arg_size() > 0 ? MC->getArgOperand(0) : nullptr;
      // Case 1: SinkInst is StoreInst (initial run).
      if (auto *SI = dyn_cast<StoreInst>(SinkInst)) {
        if (tryRewriteAxisWrite(M, AxisMarked, SI->getPointerOperand(), SI,
                                SI->getValueOperand())) {
          Changed = true;
          continue;
        }
        if (MarkerStream &&
            tryRewriteAxisWrite(M, AxisMarked, MarkerStream, SI,
                                SI->getValueOperand())) {
          Changed = true;
          continue;
        }
      }
      // Case 2: SinkInst is `llvm.fpga.fifo.push` call (already
      // EarlyPrep-converted, but we now have AXIS info → re-rewrite).
      if (auto *CI = dyn_cast<CallInst>(SinkInst)) {
        if (Function *Callee = CI->getCalledFunction()) {
          if (Callee->getName().startswith("llvm.fpga.fifo.push")) {
            if (CI->arg_size() >= 2) {
              if (tryRewriteAxisWrite(M, AxisMarked, CI->getArgOperand(1), CI,
                                      CI->getArgOperand(0))) {
                Changed = true;
                continue;
              }
              if (MarkerStream &&
                  tryRewriteAxisWrite(M, AxisMarked, MarkerStream, CI,
                                      CI->getArgOperand(0))) {
                Changed = true;
                continue;
              }
            }
          }
        }
      }
      // Case 3: SinkInst is `llvm.memcpy` (rustc emits memcpy for
      // struct-value writes that survive SROA). The dst is the stream
      // ptr, src is a temp holding the value. We can re-purpose the
      // struct as the temp directly since both src and dst point at
      // ElemST-shaped memory.
      if (auto *MCpy = dyn_cast<MemCpyInst>(SinkInst)) {
        Value *DstPtr = MCpy->getDest();
        Value *SrcPtr = MCpy->getSource();
        Value *Trace = DstPtr;
        while (true) {
          if (auto *BC = dyn_cast<BitCastInst>(Trace)) { Trace = BC->getOperand(0); continue; }
          if (auto *G = dyn_cast<GetElementPtrInst>(Trace)) { Trace = G->getPointerOperand(); continue; }
          break;
        }
        if (AxisMarked.count(Trace)) {
          if (auto *PT = dyn_cast<PointerType>(Trace->getType())) {
            if (auto *StreamST = dyn_cast<StructType>(PT->getElementType())) {
              // Same ZSA-sandwich handling as rewriteStreamReads.
              StructType *ElemST = nullptr;
              unsigned NF = StreamST->getNumElements();
              if (NF == 1) {
                ElemST = dyn_cast<StructType>(StreamST->getElementType(0));
              } else if (NF == 3) {
                auto *F0 = dyn_cast<ArrayType>(StreamST->getElementType(0));
                auto *F2 = dyn_cast<ArrayType>(StreamST->getElementType(2));
                if (F0 && F0->getNumElements() == 0 &&
                    F2 && F2->getNumElements() == 0) {
                  ElemST = dyn_cast<StructType>(StreamST->getElementType(1));
                }
              }
              if (ElemST) {
                if (ElemST->getNumElements() >= 2) {
                    IRBuilder<> B(MCpy);
                    Type *ElemPtrTy = PointerType::get(ElemST, 0);
                    Value *DstElem = DstPtr;
                    if (DstElem->getType() != ElemPtrTy)
                      DstElem = B.CreateBitCast(DstElem, ElemPtrTy);
                    Value *SrcElem = SrcPtr;
                    if (SrcElem->getType() != ElemPtrTy)
                      SrcElem = B.CreateBitCast(SrcElem, ElemPtrTy);
                    SmallVector<Value *, 14> Args;
                    SmallVector<Type *, 7> FieldPtrTys;
                    SmallVector<unsigned, 7> RealIdxs;
                    computeAxisPackedFieldPtrs(M, ElemST, FieldPtrTys, RealIdxs);
                    // Nested-mode override; the bare ElemST GEPs below produce
                    // i32*/i8* which appendAxisChannelGeps then bitcasts up to
                    // the struct ptr types.
                    applyAxisNestedOverride(M, ElemST, FieldPtrTys, RealIdxs);
                    appendAxisChannelGeps(B, ElemST, DstElem, RealIdxs, FieldPtrTys, Args);
                    appendAxisChannelGeps(B, ElemST, SrcElem, RealIdxs, FieldPtrTys, Args);
                    Function *Push = hlsrs::vxx::getOrInsertAxisPopOrPush(M, "push", FieldPtrTys);
                    B.CreateCall(Push, Args);
                    MCpy->eraseFromParent();
                    Changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }
      }
    if (auto *SI = dyn_cast<StoreInst>(SinkInst)) {
      Type *StoreTy = SI->getValueOperand()->getType();
      // cpp_proxy cosim (`__vxx_axis_packed`): symmetric to the read side. The
      // struct write was SROA-packed to an integer store; emit a struct-typed
      // `fifo.push.<ElemST>` so the synthesized arg matches the C++
      // `hls::stream<Struct>` element (the iN form trips HLS 214-136). Stage A
      // (no env) is untouched.
      if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
        if (auto *IST = dyn_cast<IntegerType>(StoreTy)) {
          StructType *ElemST = streamElemStructOf(SI->getPointerOperand());
          if (ElemST && DL.getTypeAllocSizeInBits(ElemST) == IST->getBitWidth()) {
            IRBuilder<> B(SI);
            Function *F = SI->getParent()->getParent();
            IRBuilder<> EntryB(&F->getEntryBlock(),
                               F->getEntryBlock().getFirstInsertionPt());
            AllocaInst *Tmp =
                EntryB.CreateAlloca(ElemST, nullptr, "axis.packed.write");
            Value *AsInt = B.CreateBitCast(Tmp, PointerType::get(IST, 0));
            B.CreateStore(SI->getValueOperand(), AsInt);
            Value *StructVal = B.CreateLoad(ElemST, Tmp);
            Value *STPtr = B.CreateBitCast(SI->getPointerOperand(),
                                           PointerType::get(ElemST, 0));
            Function *Push = hlsrs::vxx::getOrInsertFifoPushAny(M, ElemST);
            B.CreateCall(Push, {StructVal, STPtr});
            SI->eraseFromParent();
            Changed = true;
            continue;
          }
        }
      }
      IntegerType *ElemTy = dyn_cast<IntegerType>(StoreTy);
      if (ElemTy) {
        IRBuilder<> B(SI);
        Function *Push = hlsrs::vxx::getOrInsertFifoPushAny(M, ElemTy);
        // Single-field struct wrapper elements (C++ dat_t<T>) store the
        // scalar through a struct-typed pointer — cast to the scalar ptr
        // (the C++ frontend flattens the wrapper; see the pop side).
        Value *P = SI->getPointerOperand();
        Type *WantPT = PointerType::get(
            ElemTy, cast<PointerType>(P->getType())->getAddressSpace());
        if (P->getType() != WantPT)
          P = B.CreateBitCast(P, WantPT);
        B.CreateCall(Push, {SI->getValueOperand(), P});
        SI->eraseFromParent();
      } else {
        // Struct-typed stream element: emit `llvm.fpga.fifo.push.<T>.p0<T>`
        // directly with the struct type. No bitcast to iN*, no temp alloca
        // staging — just pass the struct value and pointer to the intrinsic.
        IRBuilder<> B(SI);
        Function *Push = hlsrs::vxx::getOrInsertFifoPushAny(M, StoreTy);
        B.CreateCall(Push,
                     {SI->getValueOperand(), SI->getPointerOperand()});
        SI->eraseFromParent();
      }
    } else {
      // Volatile memcpy form (struct path): rewrite to a single iN FIFO push
      // of the bytes in question.
      auto *MCpy = cast<MemCpyInst>(SinkInst);
      auto *LenC = dyn_cast<ConstantInt>(MCpy->getLength());
      if (!LenC)
        continue;
      // cpp_proxy cosim (`__vxx_axis_packed`): if the memcpy destination is a
      // struct-element stream, push the whole struct (`fifo.push.<ElemST>`)
      // so the synthesized arg matches the C++ `hls::stream<Struct>` element
      // instead of an iN bag (which trips HLS 214-136). Stage A is untouched.
      if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
        StructType *ElemST = streamElemStructOf(MCpy->getDest());
        if (ElemST &&
            DL.getTypeAllocSizeInBits(ElemST) == LenC->getZExtValue() * 8) {
          IRBuilder<> B(MCpy);
          Type *ElemPtrTy = PointerType::get(ElemST, 0);
          Value *SrcST = B.CreateBitCast(MCpy->getSource(), ElemPtrTy);
          Value *DstST = B.CreateBitCast(MCpy->getDest(), ElemPtrTy);
          Value *StructVal = B.CreateLoad(ElemST, SrcST);
          Function *Push = hlsrs::vxx::getOrInsertFifoPushAny(M, ElemST);
          B.CreateCall(Push, {StructVal, DstST});
          MCpy->eraseFromParent();
          Changed = true;
          continue;
        }
      }
      uint64_t Bits = LenC->getZExtValue() * 8;
      IntegerType *IntTy = IntegerType::get(M.getContext(), Bits);
      Type *IntPtrTy = PointerType::get(IntTy, MCpy->getDestAddressSpace());
      IRBuilder<> B(MCpy);
      Value *SrcAsInt = B.CreateBitCast(MCpy->getSource(), IntPtrTy);
      Value *DstAsInt = B.CreateBitCast(MCpy->getDest(), IntPtrTy);
      LoadInst *Bag = B.CreateLoad(IntTy, SrcAsInt);
      Function *Push = hlsrs::vxx::getOrInsertFifoPushAny(M, IntTy);
      B.CreateCall(Push, {Bag, DstAsInt});
      MCpy->eraseFromParent();
    }
    Changed = true;
  }
  for (CallInst *CI : Markers)
    CI->eraseFromParent();
  return Changed;
}

// Undo LLVM tail-merge that produced
//   call llvm.fpga.fifo.push(val, select(cond, p1, p2))
// patterns. Vitis HLS rejects select-typed stream pointers
// (`SYNCHK 200-43 use or assignment of a non-static pointer`). Split
// the parent BB into two branches, each pushing to a distinct stream
// pointer. Required for if/else stream-write patterns common in
// KPN dataflow helpers (e.g. simple_data_driven's splitter).
//
// Runs unconditionally — must NOT be gated on `__vxx_stream_write_marker`
// presence because that function is DCE'd after EarlyPrep, but the
// merge can survive into post-prep optimisation.
bool splitMergedFifoPushes(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<CallInst *, 8> ToSplit;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee) continue;
        if (!Callee->getName().startswith("llvm.fpga.fifo.push")) continue;
        if (CI->arg_size() < 2) continue;
        if (isa<SelectInst>(CI->getArgOperand(1)))
          ToSplit.push_back(CI);
      }
    for (CallInst *CI : ToSplit) {
      auto *Sel = cast<SelectInst>(CI->getArgOperand(1));
      Value *Cond = Sel->getCondition();
      Value *TrueP = Sel->getTrueValue();
      Value *FalseP = Sel->getFalseValue();
      Value *Val = CI->getArgOperand(0);
      Function *Push = CI->getCalledFunction();

      BasicBlock *PreBB = CI->getParent();
      BasicBlock *PostBB =
          PreBB->splitBasicBlock(CI, PreBB->getName() + ".write.post");
      // splitBasicBlock created an unconditional br PreBB->PostBB; replace it.
      Instruction *Br = PreBB->getTerminator();
      Br->eraseFromParent();
      BasicBlock *TrueBB = BasicBlock::Create(
          F.getContext(), PreBB->getName() + ".write.true", &F, PostBB);
      BasicBlock *FalseBB = BasicBlock::Create(
          F.getContext(), PreBB->getName() + ".write.false", &F, PostBB);
      IRBuilder<> PreB(PreBB);
      PreB.CreateCondBr(Cond, TrueBB, FalseBB);
      IRBuilder<> TrueB(TrueBB);
      TrueB.CreateCall(Push, {Val, TrueP});
      TrueB.CreateBr(PostBB);
      IRBuilder<> FalseB(FalseBB);
      FalseB.CreateCall(Push, {Val, FalseP});
      FalseB.CreateBr(PostBB);
      CI->eraseFromParent();
      if (Sel->use_empty())
        Sel->eraseFromParent();
      Changed = true;
      vxxDbg() << "vxx: split merged push into 2 branches in "
             << F.getName() << "\n";
    }
  }
  return Changed;
}

// Rewrite `bitcast %struct* → iN*` followed by integer-typed
// `llvm.fpga.fifo.push.iN` to use the struct-typed intrinsic with a
// short alloca round-trip for the value:
//
//   %p = bitcast %struct* %src to iN*
//   call void @llvm.fpga.fifo.push.iN.p0iN(iN %v, iN* %p)
// →
//   %tmp = alloca %struct
//   %tmp_iN = bitcast %struct* %tmp to iN*
//   store iN %v, iN* %tmp_iN
//   %sv = load %struct, %struct* %tmp
//   call void @llvm.fpga.fifo.push.<struct>.p0<struct>(%struct %sv, %struct* %src)
//
// The alloca-temp bitcast is on a fresh local (not on the lifted stream
// alloca), so it does not block the auto-AGGREGATE pragma that Vitis HLS
// applies to local hls::stream<T>. The typed intrinsic is the form
// hls::stream<T>::write expects; this pass brings the Rust-emitted
// shape in line. Required for hls::stream<hls::vector<...>> (using_fifos),
// where the bitcast on the stream-derived ptr triggers HLS 214-319.
bool rewriteFifoStructPtrBitcasts(Module &M) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<CallInst *, 8> Pushes;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee) continue;
        if (!Callee->getName().startswith("llvm.fpga.fifo.push.i")) continue;
        if (CI->arg_size() < 2) continue;
        Pushes.push_back(CI);
      }
    for (CallInst *CI : Pushes) {
      Value *PtrArg = CI->getArgOperand(1);
      auto *BC = dyn_cast<BitCastInst>(PtrArg);
      if (!BC) continue;
      auto *DstPtrTy = dyn_cast<PointerType>(BC->getOperand(0)->getType());
      if (!DstPtrTy) continue;
      auto *DstStruct = dyn_cast<StructType>(DstPtrTy->getElementType());
      if (!DstStruct || !DstStruct->hasName()) continue;
      Value *FifoPtr = BC->getOperand(0);
      // Only rewrite when the struct is one of the named hls::* wrappers
      // that Vitis HLS recognises as a FIFO element type.
      StringRef N = DstStruct->getName();
      // `__vxx_axis_packed`: a `class.hls::stream<hls::axis<...>>` ptr is the
      // STREAM, not the FIFO element — descend to the inner axis element
      // struct so the push moves the axis packet (not the whole stream).
      if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed") &&
          N.startswith("class.hls::stream<hls::axis<")) {
        StructType *ElemST = nullptr;
        unsigned NF = DstStruct->getNumElements();
        if (NF == 1) ElemST = dyn_cast<StructType>(DstStruct->getElementType(0));
        else if (NF == 3) ElemST = dyn_cast<StructType>(DstStruct->getElementType(1));
        if (ElemST) {
          IRBuilder<> CB0(CI);
          FifoPtr = CB0.CreateBitCast(FifoPtr, PointerType::get(ElemST, 0));
          DstStruct = ElemST;
          N = DstStruct->getName();
        }
      }
      if (!N.startswith("class.hls::vector<") &&
          !N.startswith("class.hls::stream<") &&
          !N.startswith("struct.hls::axis<"))
        continue;
      Value *Val = CI->getArgOperand(0);
      auto *IntTy = dyn_cast<IntegerType>(Val->getType());
      if (!IntTy) continue;
      // Bit-width sanity: struct's storage width must equal iN.
      if (DL.getTypeSizeInBits(DstStruct) != IntTy->getBitWidth()) continue;
      Function *TypedPush = hlsrs::vxx::getOrInsertFifoPushAny(M, DstStruct);
      // Place the alloca in entry BB to avoid stack growth in loops.
      BasicBlock &Entry = CI->getFunction()->getEntryBlock();
      IRBuilder<> AB(&*Entry.getFirstInsertionPt());
      AllocaInst *Tmp = AB.CreateAlloca(DstStruct, nullptr, "fifo.push.tmp");
      IRBuilder<> CB(CI);
      Value *TmpIN = CB.CreateBitCast(Tmp,
                                       PointerType::getUnqual(IntTy),
                                       "fifo.push.tmp.iN");
      CB.CreateStore(Val, TmpIN);
      Value *StructVal = CB.CreateLoad(DstStruct, Tmp, "fifo.push.struct");
      CallInst *NewCI = CB.CreateCall(
          TypedPush, {StructVal, FifoPtr});
      NewCI->setCallingConv(CI->getCallingConv());
      CI->eraseFromParent();
      if (BC->use_empty()) BC->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// `hls::stream_of_blocks` — declaration + lock markers to the C++ pre-reflow
// ops (a.pp.bc of using_stream_of_blocks): the channel (a `[B x T]` alloca —
// StreamOfBlocks is repr(transparent)) gets `_ssdm_op_SpecChannel(name, 0,
// "", "", depth, bytes, ptr, ptr)` + `_ssdm_op_SpecPipoDepth(ptr, depth)`;
// each `read_lock`/`write_lock` acquisition becomes
// `_ssdm_op_ReadReq.volatile(ptr, n)` / `_ssdm_op_WriteReq.volatile(ptr, n)`.
bool injectStreamOfBlocks(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  bool Changed = false;

  // The channel struct (`StreamOfBlocks<T, B>` = { [B x T] }) does not
  // collapse under repr(transparent) at the LLVM level — remap it to its
  // block array module-wide first (same machinery as
  // dissolveSingleFieldStructs), so allocas/params/GEPs take the C++
  // `[B x T]` shape before the specs are emitted.
  if (M.getFunction("__vxx_sob_channel")) {
    DenseMap<Type *, Type *> TypeMap;
    for (StructType *ST : M.getIdentifiedStructTypes()) {
      if (ST->isOpaque()) continue;
      if (!ST->getName().startswith(
              "barista_hls::stream_of_blocks::StreamOfBlocks<"))
        continue;
      if (ST->getNumElements() == 1 &&
          isa<ArrayType>(ST->getElementType(0)))
        TypeMap[ST] = ST->getElementType(0);
    }
    if (!TypeMap.empty())
      hlsrs::vxx::remapStructsInModule(M, TypeMap, /*FieldMaps=*/nullptr);
  }

  if (Function *ChM = M.getFunction("__vxx_sob_channel")) {
    FunctionCallee SpecChannel = M.getOrInsertFunction(
        "_ssdm_op_SpecChannel", FunctionType::get(Type::getVoidTy(Ctx), true));
    FunctionCallee SpecPipo = M.getOrInsertFunction(
        "_ssdm_op_SpecPipoDepth", FunctionType::get(Type::getVoidTy(Ctx), true));
    SmallVector<CallInst *, 4> Calls;
    for (User *U : ChM->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *MC : Calls) {
      Value *P = MC->getArgOperand(0);
      while (auto *BC = dyn_cast<BitCastInst>(P)) P = BC->getOperand(0);
      auto *AI = dyn_cast<AllocaInst>(P);
      auto *DepthC = dyn_cast<ConstantInt>(MC->getArgOperand(1));
      auto *BytesC = dyn_cast<ConstantInt>(MC->getArgOperand(2));
      if (!AI || !DepthC || !BytesC) {
        vxxDbg() << "vxx: sob_channel marker skipped (shape)\n";
        continue;
      }
      // Vitis channel objects carry align 512 (every C++ pp channel alloca).
      AI->setAlignment(Align(512));
      // C++-style channel name from the alloca (strip inline suffixes).
      std::string Nm = AI->getName().str();
      while (Nm.size() > 2 && Nm.compare(Nm.size() - 2, 2, ".i") == 0)
        Nm.resize(Nm.size() - 2);
      GlobalVariable *NameG = hlsrs::vxx::getOrCreateCStrGlobal(M, Nm);
      GlobalVariable *Empty = hlsrs::vxx::getOrCreateCStrGlobal(M, "");
      IRBuilder<> B(MC);
      B.CreateCall(SpecChannel,
                   {NameG, ConstantInt::get(I32, 0), Empty, Empty,
                    ConstantInt::get(I32, DepthC->getZExtValue()),
                    ConstantInt::get(I32, BytesC->getZExtValue()), AI, AI});
      B.CreateCall(SpecPipo, {AI,
                              ConstantInt::get(I32, DepthC->getZExtValue()),
                              Empty});
      // The channel's InterfaceSpec — C++ pp: SpecInterface(ptr, "mem_fifo",
      // 0, 0, "", 0, 0, "", "", "", 2, 2, 16, 16, "", "", -1, 0, -1, 0).
      {
        FunctionCallee SpecIf = M.getOrInsertFunction(
            "_ssdm_op_SpecInterface",
            FunctionType::get(Type::getVoidTy(Ctx), true));
        GlobalVariable *MemFifo =
            hlsrs::vxx::getOrCreateCStrGlobal(M, "mem_fifo");
        B.CreateCall(
            SpecIf,
            {AI, MemFifo, ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
             Empty, ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), Empty,
             Empty, Empty, ConstantInt::get(I32, 2), ConstantInt::get(I32, 2),
             ConstantInt::get(I32, 16), ConstantInt::get(I32, 16), Empty,
             Empty, ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
             ConstantInt::get(I32, -1), ConstantInt::get(I32, 0)});
      }
      MC->eraseFromParent();
      // Erase the constructor zero-init (a raw write into the channel makes
      // the backend synthesize a writer process — same rule as the other
      // channel components).
      SmallVector<Instruction *, 8> Dead;
      for (User *U : AI->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (isa<Constant>(SI->getValueOperand())) Dead.push_back(SI);
          continue;
        }
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          for (User *BU : BC->users()) {
            if (auto *II = dyn_cast<IntrinsicInst>(BU)) {
              if (II->getIntrinsicID() == Intrinsic::memset) Dead.push_back(II);
            }
            if (auto *MCp = dyn_cast<MemCpyInst>(BU))
              if (MCp->getRawDest() == BC) Dead.push_back(MCp);
            if (auto *SI2 = dyn_cast<StoreInst>(BU))
              if (isa<Constant>(SI2->getValueOperand())) Dead.push_back(SI2);
          }
          continue;
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // ctor zero-init unrolled into per-element constant stores; the
          // channel local's only top-level GEP users are that init (the
          // helpers receive the pointer through calls).
          bool OnlyInit = true;
          SmallVector<Instruction *, 4> Sub;
          for (User *GU : GEP->users()) {
            if (auto *SI3 = dyn_cast<StoreInst>(GU)) {
              if (isa<Constant>(SI3->getValueOperand())) {
                Sub.push_back(SI3);
                continue;
              }
            }
            OnlyInit = false;
            break;
          }
          if (OnlyInit) {
            for (Instruction *I : Sub) Dead.push_back(I);
            Dead.push_back(GEP);
          }
          continue;
        }
      }
      for (Instruction *I : Dead) I->eraseFromParent();
      vxxDbg() << "vxx: stream_of_blocks channel " << Nm << " (depth="
               << DepthC->getZExtValue() << ")\n";
      Changed = true;
    }
  }

  auto LowerReq = [&](const char *MarkerName, const char *SsdmName) {
    Function *F = M.getFunction(MarkerName);
    if (!F) return;
    FunctionCallee Ssdm = M.getOrInsertFunction(
        SsdmName, FunctionType::get(Type::getVoidTy(Ctx), true));
    SmallVector<CallInst *, 8> Calls;
    // direct calls AND calls through a ConstantExpr bitcast of the marker
    // (cross-CGU declaration type skew — e.g. the guard-drop release calls)
    SmallVector<Value *, 4> Roots;
    Roots.push_back(F);
    for (User *U : F->users())
      if (isa<ConstantExpr>(U)) Roots.push_back(U);
    for (Value *R : Roots)
      for (User *U : R->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledOperand()->stripPointerCasts() == F)
            Calls.push_back(CI);
    for (CallInst *MC : Calls) {
      Value *P = MC->getArgOperand(0);
      while (auto *BC = dyn_cast<BitCastInst>(P)) P = BC->getOperand(0);
      auto *NC = dyn_cast<ConstantInt>(MC->getArgOperand(1));
      IRBuilder<> B(MC);
      B.CreateCall(Ssdm,
                   {P, ConstantInt::get(I32, NC ? NC->getZExtValue() : 2)});
      MC->eraseFromParent();
      Changed = true;
    }
  };
  LowerReq("__vxx_sob_read_req", "_ssdm_op_ReadReq.volatile");
  LowerReq("__vxx_sob_write_req", "_ssdm_op_WriteReq.volatile");
  LowerReq("__vxx_sob_read_release", "_ssdm_op_Read.volatile");
  LowerReq("__vxx_sob_write_release", "_ssdm_op_Write.volatile");

  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_sob_channel");
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_sob_read_req");
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_sob_write_req");
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_sob_read_release");
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_sob_write_release");
  return Changed;
}

// `__vxx_nport_channel(ptr, nin, nout, depth_out, kind, depth_in)` — the
// hls::merge / hls::split channel component. The Rust side declares ONE
// component struct local ({[N x T], T} for merge, {T, [N x T]} for split;
// Stream<T> is repr(transparent) so the fields are plain T slots) and marks
// it; this pass reproduces the C++ pre-reflow form (a.pp.bc of the
// merge_split baselines):
//   %s_in_0 = alloca T   + _ssdm_SpecStream + SpecInterface("ap_fifo")
//   ...                    (one per channel)
//   _ssdm_op_SpecNPortChannel(ins..., outs..., nin, nout, DEPTH_OUT, kind,
//                             DEPTH_IN)        ; kind 1=load_balance 2=rr
// then re-points every field GEP to its channel alloca, erases the
// constructor zero-init (a raw store into a stream channel makes the HLS
// backend synthesize a Block_start writer process) and drops the
// component struct.
bool injectNPortChannel(Module &M) {
  Function *Marker = M.getFunction("__vxx_nport_channel");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  bool Changed = false;

  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);

  for (CallInst *MC : Calls) {
    Value *P = MC->getArgOperand(0);
    while (auto *BC = dyn_cast<BitCastInst>(P)) P = BC->getOperand(0);
    auto *AI = dyn_cast<AllocaInst>(P);
    auto *NinC = dyn_cast<ConstantInt>(MC->getArgOperand(1));
    auto *NoutC = dyn_cast<ConstantInt>(MC->getArgOperand(2));
    auto *DoutC = dyn_cast<ConstantInt>(MC->getArgOperand(3));
    auto *KindC = dyn_cast<ConstantInt>(MC->getArgOperand(4));
    auto *DinC = dyn_cast<ConstantInt>(MC->getArgOperand(5));
    if (!AI || !NinC || !NoutC || !DoutC || !KindC || !DinC) {
      vxxDbg() << "vxx: nport_channel marker skipped (shape)\n";
      continue;
    }
    auto *ST = dyn_cast<StructType>(AI->getAllocatedType());
    if (!ST) {
      vxxDbg() << "vxx: nport_channel skipped (not struct alloca)\n";
      continue;
    }
    unsigned Nin = (unsigned)NinC->getZExtValue();
    unsigned Nout = (unsigned)NoutC->getZExtValue();
    bool IsMerge = Nin > 1;
    unsigned NMany = IsMerge ? Nin : Nout;

    // Locate the [N x T] (many) and T (one) fields, skipping zero-size pads.
    const DataLayout &DL = M.getDataLayout();
    int ManyIdx = -1, OneIdx = -1;
    Type *ElemT = nullptr;
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *FT = ST->getElementType(i);
      if (FT->isSized() && DL.getTypeAllocSize(FT) == 0) continue;
      if (auto *AT = dyn_cast<ArrayType>(FT)) {
        if (AT->getNumElements() == NMany && ManyIdx < 0) {
          ManyIdx = (int)i;
          ElemT = AT->getElementType();
          continue;
        }
      }
      if (OneIdx < 0) OneIdx = (int)i;
    }
    if (ManyIdx < 0 || OneIdx < 0 || !ElemT ||
        ST->getElementType(OneIdx) != ElemT) {
      vxxDbg() << "vxx: nport_channel skipped (field shape) on ";
      ST->print(vxxDbg());
      vxxDbg() << "\n";
      continue;
    }

    // Per-channel allocas + SpecStream + ap_fifo SpecInterface, C++-named.
    IRBuilder<> B(AI);
    StringRef Base = AI->getName();
    std::string BaseS = Base.empty() ? std::string("s") : Base.str();
    // strip rustc inlining suffixes for the C++-style channel names
    while (true) {
      size_t L = BaseS.size();
      if (L > 2 && BaseS.compare(L - 2, 2, ".i") == 0) { BaseS.resize(L - 2); continue; }
      break;
    }
    GlobalVariable *Empty = hlsrs::vxx::getOrCreateCStrGlobal(M, "");
    GlobalVariable *ApFifo = hlsrs::vxx::getOrCreateCStrGlobal(M, "ap_fifo");
    FunctionCallee SpecStream = M.getOrInsertFunction(
        "_ssdm_SpecStream",
        FunctionType::get(Type::getVoidTy(Ctx), true));
    FunctionCallee SpecIf = M.getOrInsertFunction(
        "_ssdm_op_SpecInterface",
        FunctionType::get(Type::getVoidTy(Ctx), true));
    FunctionCallee SpecNPort = M.getOrInsertFunction(
        "_ssdm_op_SpecNPortChannel",
        FunctionType::get(Type::getVoidTy(Ctx), true));

    auto MkChannel = [&](const Twine &Name) -> AllocaInst * {
      AllocaInst *CA = B.CreateAlloca(ElemT, nullptr, Name);
      B.CreateCall(SpecStream, {CA, ConstantInt::get(I32, 0),
                                ConstantInt::get(I32, 0), Empty});
      B.CreateCall(SpecIf,
                   {CA, ApFifo, ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), Empty, ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), Empty, Empty, Empty,
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), Empty,
                    Empty, ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                    ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)});
      return CA;
    };

    SmallVector<AllocaInst *, 8> Many;
    AllocaInst *One = nullptr;
    if (IsMerge) {
      for (unsigned k = 0; k < NMany; ++k)
        Many.push_back(MkChannel(BaseS + "_in_" + Twine(k)));
      One = MkChannel(BaseS + "_out");
    } else {
      One = MkChannel(BaseS + "_in");
      for (unsigned k = 0; k < NMany; ++k)
        Many.push_back(MkChannel(BaseS + "_out_" + Twine(k)));
    }
    SmallVector<Value *, 16> NPortArgs;
    if (IsMerge) {
      for (AllocaInst *A : Many) NPortArgs.push_back(A);
      NPortArgs.push_back(One);
    } else {
      NPortArgs.push_back(One);
      for (AllocaInst *A : Many) NPortArgs.push_back(A);
    }
    NPortArgs.push_back(ConstantInt::get(I32, Nin));
    NPortArgs.push_back(ConstantInt::get(I32, Nout));
    NPortArgs.push_back(ConstantInt::get(I32, DoutC->getSExtValue()));
    NPortArgs.push_back(ConstantInt::get(I32, KindC->getZExtValue()));
    NPortArgs.push_back(ConstantInt::get(I32, DinC->getSExtValue()));
    B.CreateCall(SpecNPort, NPortArgs);

    // Re-point the component struct's uses. TWO-PHASE: classify every use
    // first (no mutation), then apply — an abort must leave the IR unchanged.
    bool Aborted = false;
    SmallVector<std::pair<Instruction *, Value *>, 8> Repls; // gep -> channel
    SmallVector<Instruction *, 16> Dead;
    SmallVector<User *, 16> Users(AI->user_begin(), AI->user_end());
    for (User *U : Users) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        // gep [0, fieldIdx, (k, trailing zeros...)]
        if (GEP->getNumIndices() < 2) { Aborted = true; break; }
        auto It = GEP->idx_begin();
        auto *Zero = dyn_cast<ConstantInt>(It->get());
        auto *FieldC = dyn_cast<ConstantInt>((It + 1)->get());
        if (!Zero || !Zero->isZero() || !FieldC) { Aborted = true; break; }
        unsigned F = (unsigned)FieldC->getZExtValue();
        Value *Repl = nullptr;
        if ((int)F == OneIdx && GEP->getNumIndices() == 2) {
          Repl = One;
        } else if ((int)F == ManyIdx && GEP->getNumIndices() >= 3) {
          auto *KC = dyn_cast<ConstantInt>((It + 2)->get());
          bool RestZero = true;
          for (auto It2 = It + 3; It2 != GEP->idx_end(); ++It2) {
            auto *Z = dyn_cast<ConstantInt>(It2->get());
            if (!Z || !Z->isZero()) { RestZero = false; break; }
          }
          if (KC && RestZero && KC->getZExtValue() < Many.size())
            Repl = Many[(unsigned)KC->getZExtValue()];
        } else if ((int)F == ManyIdx && GEP->getNumIndices() == 2) {
          // Two-level access: `gep [0, manyIdx]` to the whole [N x T] array,
          // then per-element `gep [0, k]` sub-GEPs. Classify every sub-GEP
          // to its channel; the intermediate GEP dies with them.
          bool SubsOk = true;
          SmallVector<std::pair<Instruction *, Value *>, 4> SubRepls;
          for (User *GU : GEP->users()) {
            // ctor zero-init of the array field reaches through a cast
            // (memcpy from a zero temp / memset / lifetime) — classify the
            // whole cast chain as dead init.
            if (auto *SBC = dyn_cast<BitCastInst>(GU)) {
              bool CastOk = true;
              SmallVector<Instruction *, 4> Sub;
              for (User *CU : SBC->users()) {
                if (auto *II = dyn_cast<IntrinsicInst>(CU)) {
                  if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                      II->getIntrinsicID() == Intrinsic::lifetime_end ||
                      II->getIntrinsicID() == Intrinsic::memset) {
                    Sub.push_back(II);
                    continue;
                  }
                }
                if (auto *MCp = dyn_cast<MemCpyInst>(CU)) {
                  if (MCp->getRawDest() == SBC) { Sub.push_back(MCp); continue; }
                }
                if (auto *SI2 = dyn_cast<StoreInst>(CU)) {
                  if (isa<Constant>(SI2->getValueOperand())) {
                    Sub.push_back(SI2);
                    continue;
                  }
                }
                CastOk = false;
                break;
              }
              if (!CastOk) { SubsOk = false; break; }
              for (Instruction *I : Sub) Dead.push_back(I);
              Dead.push_back(SBC);
              continue;
            }
            auto *SG = dyn_cast<GetElementPtrInst>(GU);
            if (!SG || SG->getNumIndices() != 2) { SubsOk = false; break; }
            auto SIt = SG->idx_begin();
            auto *SZ = dyn_cast<ConstantInt>(SIt->get());
            auto *SK = dyn_cast<ConstantInt>((SIt + 1)->get());
            if (!SZ || !SZ->isZero() || !SK ||
                SK->getZExtValue() >= Many.size()) {
              SubsOk = false;
              break;
            }
            Value *Ch = Many[(unsigned)SK->getZExtValue()];
            if (Ch->getType() != SG->getType()) { SubsOk = false; break; }
            SubRepls.push_back({SG, Ch});
          }
          if (!SubsOk) {
            vxxDbg() << "vxx: nport abort many-subgep: ";
            GEP->print(vxxDbg());
            vxxDbg() << "\n";
            Aborted = true;
            break;
          }
          for (auto &SR : SubRepls) Repls.push_back(SR);
          Dead.push_back(GEP);
          continue;
        }
        if (!Repl || Repl->getType() != GEP->getType()) {
          vxxDbg() << "vxx: nport abort gep: ";
          GEP->print(vxxDbg());
          vxxDbg() << "\n";
          Aborted = true;
          break;
        }
        Repls.push_back({GEP, Repl});
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        // constructor zero-init of the whole struct
        if (isa<Constant>(SI->getValueOperand())) { Dead.push_back(SI); continue; }
        Aborted = true;
        break;
      }
      if (auto *BC = dyn_cast<BitCastInst>(U)) {
        // Through an i8*/()-cast: lifetime intrinsics, the ctor zero-init
        // (memset OR a memcpy from a zero-temp — rustc materialises small
        // Default arrays that way), constant stores, and the component
        // marker call itself.
        bool AllOk = true;
        SmallVector<Instruction *, 4> Sub;
        for (User *BU : BC->users()) {
          if (auto *II = dyn_cast<IntrinsicInst>(BU)) {
            if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                II->getIntrinsicID() == Intrinsic::lifetime_end ||
                II->getIntrinsicID() == Intrinsic::memset) {
              Sub.push_back(II);
              continue;
            }
          }
          if (auto *MCp = dyn_cast<MemCpyInst>(BU)) {
            if (MCp->getRawDest() == BC) { Sub.push_back(MCp); continue; }
          }
          if (auto *SI2 = dyn_cast<StoreInst>(BU)) {
            if (isa<Constant>(SI2->getValueOperand())) { Sub.push_back(SI2); continue; }
          }
          if (auto *CI2 = dyn_cast<CallInst>(BU)) {
            Function *Callee = CI2->getCalledFunction();
            if (CI2 == MC ||
                (Callee && Callee->getName() == "__vxx_nport_channel")) {
              continue; // dropped with the marker definition
            }
          }
          vxxDbg() << "vxx: nport abort cast-user: ";
          BU->print(vxxDbg());
          vxxDbg() << "\n";
          AllOk = false;
          break;
        }
        if (!AllOk) { Aborted = true; break; }
        for (Instruction *I : Sub) Dead.push_back(I);
        continue;
      }
      if (auto *CI = dyn_cast<CallInst>(U)) {
        Function *Callee = CI->getCalledFunction();
        if (CI == MC || (Callee && Callee->getName() == "__vxx_nport_channel"))
          continue; // the marker itself (erased below)
        vxxDbg() << "vxx: nport abort call-user: ";
        CI->print(vxxDbg());
        vxxDbg() << "\n";
        Aborted = true;
        break;
      }
      vxxDbg() << "vxx: nport abort other-user: ";
      U->print(vxxDbg());
      vxxDbg() << "\n";
      Aborted = true;
      break;
    }
    if (Aborted) {
      vxxDbg() << "vxx: injectNPortChannel aborted on " << BaseS
               << " (unrecognised component use)\n";
      // nothing was mutated; the fresh channel allocas are unused and the
      // backend DCEs them
      continue;
    }
    for (auto &RP : Repls) {
      RP.first->replaceAllUsesWith(RP.second);
      RP.first->eraseFromParent();
    }
    for (Instruction *I : Dead) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
    MC->eraseFromParent();
    if (AI->use_empty()) AI->eraseFromParent();
    vxxDbg() << "vxx: injectNPortChannel lowered " << BaseS << " (nin=" << Nin
             << " nout=" << Nout << " kind=" << KindC->getZExtValue() << ")\n";
    Changed = true;
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_nport_channel");
  return Changed;
}

// Rewrite `bitcast %"class.hls::stream<axis<T>>"* → %"struct.hls::axis<T>"*`
// into `GEP %stream, 0, 1` (the axis field of the Stream wrapper struct
// `{ [0 x T], axis<T>, [0 x T] }`). HLS rejects the bitcast as
// "type conversion operator" (HLS 214-319 "Cannot apply aggregate pragma
// due to the existence of one or more type conversion operators"). The GEP
// form is the idiomatic struct field access HLS expects.
bool rewriteStreamToAxisBitcastAsGEP(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<BitCastInst *, 8> Dead;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC) continue;
        auto *SrcPT = dyn_cast<PointerType>(BC->getSrcTy());
        auto *DstPT = dyn_cast<PointerType>(BC->getDestTy());
        if (!SrcPT || !DstPT) continue;
        auto *SrcST = dyn_cast<StructType>(SrcPT->getElementType());
        auto *DstST = dyn_cast<StructType>(DstPT->getElementType());
        if (!SrcST || !DstST || !SrcST->hasName() || !DstST->hasName()) continue;
        // Stream<T> wrapper is `{ [0 x T], T, [0 x T] }` — 3 fields, index 1 is T.
        if (SrcST->getNumElements() != 3) continue;
        if (SrcST->getElementType(1) != DstST) continue;
        // Replace bitcast with GEP.
        IRBuilder<> B(BC);
        Type *I32 = IntegerType::get(F.getContext(), 32);
        Value *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 1)};
        Value *GEP = B.CreateGEP(SrcST, BC->getOperand(0), Idxs);
        BC->replaceAllUsesWith(GEP);
        Dead.push_back(BC);
        Changed = true;
      }
    for (auto *BC : Dead) BC->eraseFromParent();
  }
  return Changed;
}

// Consume `__vxx_top_stream_param(idx)` markers: rename the marked top
// kernel arg from scalar `T*` (Rust `&mut barista_hls::Stream<T>`, collapsed
// to `T*` by repr(transparent) over `{T}` before VXXPrep ever sees it) to the
// named-struct pointer `class.hls::stream<T>*` + inject a `stream_interface`
// op-bundle, matching the `hls::stream<T>&` arg shape.
//
// Without this, the cosim TB generator sees raw `T*` and resolves the cosim wrapper to
// `onebyonecpy_hls.p0iN` (copies ONE element per port per call) instead of
// the `class.hls::stream<T>` overload (which calls `streamcpy_hls` to drain
// ALL queued tokens). The kernel's `llvm.fpga.fifo.pop` loop then blocks
// waiting for tokens cosim TB never delivers → RTL handshake stalls → xsim
// free-runs the clock forever → cosim deadlock / 23GB OOM. This is the
// data-driven hls::task cluster (unique_task_regions et al.); the identical
// `while(1)` design cosims PASS, so this is a pure Rust-port-shape gap, not a
// Vitis limitation.
//
// Skips args that ALSO carry `__vxx_axis` (AXIS streams go through
// disaggAxisStructKernelSig). Bridges body uses via GEP-through-field-0 → T*,
// so existing `llvm.fpga.fifo.{pop,push}` on `T*` keep working. The HLS
// backend accepts the `class.hls::stream<T>*` → T* bridge (same as the mode-3
// directio path in renameDirectioArgs).
bool renameTopStreamArgs(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Function *Marker = M.getFunction("__vxx_top_stream_param");
  if (!Marker) return false;

  // Args that also carry `__vxx_axis` are AXIS — exclude (handled elsewhere).
  SmallPtrSet<Argument *, 8> AxisArgs;
  if (Function *AxisM = M.getFunction("__vxx_axis")) {
    for (User *U : AxisM->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->arg_size() < 1) continue;
      if (Argument *A = resolveMarkerArg(CI->getArgOperand(0)))
        AxisArgs.insert(A);
    }
  }

  struct Cand {
    Function *F;
    unsigned ArgIdx;
    Type *ElemTy;
  };
  SmallVector<Cand, 8> Cands;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    MarkerCalls.push_back(CI);
    Function *F = CI->getParent()->getParent();
    if (!F->hasFnAttribute("fpga.top.func")) continue;
    auto *IdxC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    if (!IdxC) continue;
    unsigned ArgIdx = (unsigned)IdxC->getZExtValue();
    if (ArgIdx >= F->arg_size()) continue;
    Argument *A = F->getArg(ArgIdx);
    if (AxisArgs.count(A)) continue;
    auto *PT = dyn_cast<PointerType>(A->getType());
    if (!PT) continue;
    Type *ElemT = PT->getElementType();
    if (!ElemT->isIntegerTy() && !ElemT->isFloatTy() && !ElemT->isDoubleTy())
      continue;
    Cands.push_back({F, ArgIdx, ElemT});
  }
  // Erase markers regardless of whether retype succeeds.
  for (CallInst *CI : MarkerCalls) CI->eraseFromParent();
  if (Cands.empty()) return false;

  DenseMap<Function *, SmallVector<Cand, 4>> ByFn;
  for (auto &C : Cands) ByFn[C.F].push_back(C);

  bool Changed = false;
  for (auto &KV : ByFn) {
    Function *F = KV.first;
    DenseMap<unsigned, Cand *> ByIdx;
    for (auto &C : KV.second) ByIdx[C.ArgIdx] = &C;

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
      Type *ElemTy = It->second->ElemTy;
      std::string TName;
      if (auto *IT = dyn_cast<IntegerType>(ElemTy)) {
        switch (IT->getBitWidth()) {
          case 8:  TName = "char"; break;
          case 16: TName = "short"; break;
          case 32: TName = "int"; break;
          case 64: TName = "long"; break;
          default: TName = ("i" + std::to_string(IT->getBitWidth())); break;
        }
      } else if (ElemTy->isFloatTy()) {
        TName = "float";
      } else {
        TName = "double";
      }
      std::string SName = "class.hls::stream<" + TName + ">";
      StructType *ST = M.getTypeByName(SName);
      if (!ST)
        ST = StructType::create(Ctx, {ElemTy}, SName, false);
      StreamTys[i] = ST;
      NewParamTys.push_back(PointerType::get(ST, 0));
    }

    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParamTys, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       F->getName() + ".strm_tmp",
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
    SmallVector<Argument *, 4> StreamArgs;
    for (unsigned i = 0; i < NParams; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (StreamTys[i] == nullptr) {
        AttributeSet AS = F->getAttributes().getParamAttributes(i);
        for (Attribute A : AS) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Bridge: GEP through stream struct field 0 → T*.
      Value *Idxs[] = {
          ConstantInt::get(I64Ty, 0),
          ConstantInt::get(I32Ty, 0),
      };
      Value *Bridge = B.CreateInBoundsGEP(StreamTys[i], &*NewA, Idxs,
                                           NewA->getName() + ".inner");
      OldA->replaceAllUsesWith(Bridge);
      StreamArgs.push_back(&*NewA);
    }
    // Inject `stream_interface(stream*)` sideeffect per renamed arg.
    for (Argument *SA : StreamArgs) {
      OperandBundleDef SB("stream_interface", ArrayRef<Value *>(SA));
      auto *SCall = B.CreateCall(Sideeffect, ArrayRef<Value *>(), {SB});
      SCall->setOnlyAccessesInaccessibleMemory();
      SCall->setDoesNotThrow();
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: renamed top stream kernel arg(s)\n";
  return Changed;
}

bool injectStreamInterface(Module &M) {
  // Consume __vxx_top_stream_param → rename T* args to class.hls::stream<T>*
  // BEFORE dropping the marker (the drop below now only cleans up the leftover
  // declaration + any AXIS args we intentionally skipped).
  bool S = renameTopStreamArgs(M);
  bool A = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_top_stream_param") || S;
  // injectStreamApFifoSpec + IfRead.Stream rewrite caused csynth FAIL on
  // `using_ap_float_accumulator` (broke to 201-504 conflict). Root cause traces to
  // dead `bitcast <T>* %stream to {}*` body residuals that HLS interprets
  // as a 2nd interface mode hint, conflicting with our SpecInterface(ap_fifo).
  // Needs a `{}*` bitcast stripper pass before re-enabling.
  // (definitions injectStreamApFifoSpec / rewriteFifoIntrinsicsToIfReadWriteStream
  // removed — reference-only, never called.)
  // `__vxx_axis_packed`: wrapAxisReadWriteInHelpers (earlier) emits the
  // axis.pop/push (inlined onto the kernel stream port) that the HLS backend
  // needs to disaggregate the 7 side channels — do NOT lower them back to
  // load/store here (that collapses the ports to 12 data-only ports). The
  // old packed read path uses whole-struct fifo.pop (not axis.pop) so this
  // conversion was a no-op there anyway.
  // axis.pop/push lowering MOVED to AFTER disaggAxisStructKernelSig: the
  // disagg pass emits the grouped IfRead/IfWrite at each pop/push SITE
  // (C++ places them inside the loop, one per iteration); lowering the
  // intrinsics first removed those sites and the looped kernels synthesized
  // a one-read FSM (xsim 0/N stall, A_TREADY never re-raised).
  bool E = false;
  // After axis.pop/push are in place, strip the trailing [N x i8] pad from any
  // remaining padded-AXIS allocas (rustc Packet-literal temps) so the HLS
  // backend's AXISProcess solver doesn't SIGSEGV on visitAllocaInst.
  bool EP = convertPaddedAxisAllocasToNoPad(M);
  // Eliminate rustc's i96 struct round-trip in the axis body so the HLS
  // backend's CorrelatedValuePropagation/LazyValueInfo doesn't SIGSEGV on the
  // wide-int pack/unpack chain (the goal is a field-wise body). First collapse
  // the read-result i96 copy-bridge (InstCombine can't forward it through the
  // opaque axis.pop), then SROA/InstCombine/DCE the rest.
  bool ECB = collapseScalarCopyBridge(M);
  // InstCombine first (folds the i96 repack so the i64 side-channel pack is left
  // with pure trunc/lshr uses), THEN scalarize the i64 into field-wise loads,
  // THEN InstCombine/DCE again to remove the now-dead wide load + extracts.
  bool EI96 = simplifyAxisI96Body(M);
  bool ESW = scalarizeAxisWideLoads(M);
  bool EI96b = simplifyAxisI96Body(M);
  EI96 = EI96 || EI96b;
  bool G = rewriteStreamToAxisBitcastAsGEP(M);
  // Run sig disagg AFTER body rewrites so the cloned body already has the
  // GEP-form (not bitcast) and per-channel load/store (not 14-arg axis intrinsic).
  bool H = disaggAxisStructKernelSig(M);
  // NOW lower the (cloned-body) axis.pop/push into per-field load/store —
  // the IfRead/IfWrite site markers were emitted next to them during H.
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    E = rewriteAxisPopPushToLoadStore(M);
  bool I = disaggCompletePartitionKernelSig(M);
  // After H, the cloned disagg body still carries rustc's single-packet i96
  // round-trip + local class.hls::stream temps. Collapse them to field-wise
  // form so the HLS backend's cosim TB classifies the per-channel ports as
  // stream feeders (not Register) → fixes side_channel / ap_hs COSIM 212-359.
  // (Default path; simplifyAxisI96Body above is packed-only.)
  return A || E || EP || ECB || ESW || EI96 || G || H || I;
}

// Pack a depth-marked internal DATAFLOW stream whose alloca is a multi-field
// integer struct (e.g. axi_stream_to_master `{ i64, i8 }` = `struct data
// { ap_int<64>; ap_int<1>; }`) into a single iN FIFO (`%buf_r = alloca i65`).
// Two problems otherwise:
//  (1) the HLS backend field-splits the struct stream into buf_i.0/buf_i.1 and
//      DROPS the SpecStream depth on the children (HLS 200-805 -> xsim
//      deadlock);
//  (2) the Rust producer side is lowered as `fifo.push.i64(gep0)` + a raw
//      volatile store to gep1 (`last` never pushed: XFORM 203-731 "no data
//      producer"), while the consumer pops the whole struct - asymmetric.
// Packing makes both sides a single symmetric iN push/pop.
// Returns the new packed alloca, or nullptr if the shape doesn't match
// (callers then keep the original pointer untouched).
AllocaInst *packStructStreamAlloca(Module &M, AllocaInst *AI) {
  auto *ST = dyn_cast<StructType>(AI->getAllocatedType());
  if (!ST || ST->getNumElements() < 2) return nullptr;
  unsigned NF = ST->getNumElements();
  SmallVector<unsigned, 4> Width(NF), Off(NF);
  unsigned Total = 0;
  for (unsigned i = 0; i < NF; ++i) {
    auto *IT = dyn_cast<IntegerType>(ST->getElementType(i));
    if (!IT) return nullptr;
    Width[i] = IT->getBitWidth();
    Off[i] = Total;
    Total += Width[i];
  }
  LLVMContext &Ctx = M.getContext();
  auto *PackedTy = IntegerType::get(Ctx, Total);

  // Classify every user; bail (nullptr) on anything unexpected so the
  // current depth-only behaviour is preserved for shapes we don't know.
  struct FieldOp {
    unsigned Field;
    Value *Val;
    Instruction *I;
  };
  SmallVector<CallInst *, 4> StructPops;
  SmallVector<FieldOp, 8> FieldOps;
  SmallVector<Instruction *, 4> LifetimeCasts;
  SmallVector<Instruction *, 8> DeadGEPs;
  for (User *U : AI->users()) {
    if (auto *CB = dyn_cast<CallInst>(U)) {
      Function *CF = CB->getCalledFunction();
      if (CF && CF->getName().startswith("llvm.fpga.fifo.pop.") &&
          CB->getType() == ST) {
        StructPops.push_back(CB);
        continue;
      }
      return nullptr;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getNumIndices() != 2 || !GEP->hasAllConstantIndices())
        return nullptr;
      auto *Idx0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
      auto *Idx1 = dyn_cast<ConstantInt>(GEP->getOperand(2));
      if (!Idx0 || !Idx0->isZero() || !Idx1) return nullptr;
      unsigned K = (unsigned)Idx1->getZExtValue();
      if (K >= NF) return nullptr;
      for (User *GU : GEP->users()) {
        if (auto *SI = dyn_cast<StoreInst>(GU)) {
          if (SI->getPointerOperand() != GEP) return nullptr;
          FieldOps.push_back({K, SI->getValueOperand(), SI});
        } else if (auto *PC = dyn_cast<CallInst>(GU)) {
          Function *PF = PC->getCalledFunction();
          if (!PF || !PF->getName().startswith("llvm.fpga.fifo.push.") ||
              PC->arg_size() != 2 || PC->getArgOperand(1) != GEP)
            return nullptr;
          FieldOps.push_back({K, PC->getArgOperand(0), PC});
        } else {
          return nullptr;
        }
      }
      DeadGEPs.push_back(GEP);
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(U)) {
      for (User *BU : BC->users()) {
        auto *LC = dyn_cast<CallInst>(BU);
        if (!LC) return nullptr;
        Function *LF = LC->getCalledFunction();
        if (!LF || !LF->getName().startswith("llvm.lifetime."))
          return nullptr;
      }
      LifetimeCasts.push_back(BC);
      continue;
    }
    return nullptr;
  }
  if (StructPops.empty() || FieldOps.empty()) return nullptr;

  // Producer pattern: each basic block's field ops must cover all NF fields
  // exactly once (the rustc FCA per-field store lowering of `stream.write`).
  DenseMap<BasicBlock *, SmallVector<FieldOp, 4>> ByBB;
  for (auto &FO : FieldOps) ByBB[FO.I->getParent()].push_back(FO);
  for (auto &KV : ByBB) {
    if (KV.second.size() != NF) return nullptr;
    SmallVector<bool, 4> Seen(NF, false);
    for (auto &FO : KV.second) {
      if (Seen[FO.Field]) return nullptr;
      Seen[FO.Field] = true;
    }
  }

  IRBuilder<> AB(AI);
  AllocaInst *NewAI =
      AB.CreateAlloca(PackedTy, nullptr, AI->getName() + ".pk");
  NewAI->setAlignment(Align(16));
  Function *PopN = hlsrs::vxx::getOrInsertFifoPopAny(M, PackedTy);
  Function *PushN = hlsrs::vxx::getOrInsertFifoPushAny(M, PackedTy);

  // Consumer: single iN pop, then unpack into the original struct value.
  for (CallInst *PC : StructPops) {
    IRBuilder<> B(PC);
    Value *P = B.CreateCall(PopN, {NewAI});
    Value *SV = UndefValue::get(ST);
    for (unsigned i = 0; i < NF; ++i) {
      Value *F = Off[i] ? B.CreateLShr(P, Off[i]) : P;
      F = B.CreateTrunc(F, ST->getElementType(i));
      SV = B.CreateInsertValue(SV, F, {i});
    }
    PC->replaceAllUsesWith(SV);
    PC->eraseFromParent();
  }

  // Producer: pack all field values and emit one iN push at the program
  // point of the LAST field op in each block (preserves write ordering).
  for (auto &KV : ByBB) {
    SmallPtrSet<Instruction *, 8> OpSet;
    for (auto &FO : KV.second) OpSet.insert(FO.I);
    Instruction *Last = nullptr;
    for (Instruction &I : *KV.first)
      if (OpSet.count(&I)) Last = &I;
    IRBuilder<> B(Last);
    Value *P = ConstantInt::get(PackedTy, 0);
    for (auto &FO : KV.second) {
      Value *Z = B.CreateZExt(FO.Val, PackedTy);
      if (Off[FO.Field]) Z = B.CreateShl(Z, Off[FO.Field]);
      P = B.CreateOr(P, Z);
    }
    B.CreateCall(PushN, {P, NewAI});
    for (auto &FO : KV.second) FO.I->eraseFromParent();
  }
  for (Instruction *G : DeadGEPs)
    if (G->use_empty()) G->eraseFromParent();
  for (Instruction *BC : LifetimeCasts) {
    while (!BC->use_empty())
      cast<Instruction>(*BC->user_begin())->eraseFromParent();
    BC->eraseFromParent();
  }
  if (AI->use_empty()) AI->eraseFromParent();
  return NewAI;
}

// `barista_hls::ap_hs_valid(&x)` (-> `__vxx_ap_hs_valid(ptr) -> i32` marker)
// is C++ `hls::ap_hs<T>::valid()`: a NON-blocking "value present" probe on
// the handshake input. Emitted form (TLP directio/ap_hs):
//   %23 = call i1 @_ssdm_op_IfCanRead.ap_hs.p0i32(i32* nocapture %reset_myCounter)
// Rewrite each marker call into that intrinsic (zext i1 back to the
// marker's i32 return for the existing users).
bool injectApHsValid(Module &M) {
  Function *Marker = M.getFunction("__vxx_ap_hs_valid");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  const bool ApHsStream = hlsrs::vxx::markerUsed(M, "__vxx_aphs_stream");
  for (CallInst *CI : Calls) {
    if (CI->arg_size() < 1) continue;
    Value *Ptr = CI->getArgOperand(0)->stripPointerCasts();
    auto *PT = dyn_cast<PointerType>(Ptr->getType());
    if (!PT) continue;
    // `__vxx_aphs_stream`: an INPUT looped ap_hs was retyped to
    // `class.hls::stream<int>*` (renameApScalarArgsToSsdm) with its read lowered
    // to `llvm.fpga.fifo.pop`. Its valid probe must NOT stay an ap_hs handshake:
    // an `IfCanRead.ap_hs` on the stream port makes reflow synthesize it as a
    // scalar `ap_none` port (no per-beat advance) → the RTL never consumes the TV
    // beats → cosim deadlock (0/1). Model valid() as "always readable" (i1 true):
    // the C++ TB feeds exactly the beats the kernel consumes before its
    // data-dependent break, so the FIFO is never empty during a read (the host
    // C-model's cosim_direct_valid = !stream.empty() is likewise true throughout,
    // keeping C-model and RTL consistent).
    if (ApHsStream) {
      Value *Base = Ptr;
      while (auto *GEP = dyn_cast<GetElementPtrInst>(Base)) Base = GEP->getPointerOperand();
      Base = Base->stripPointerCasts();
      auto *BPT = dyn_cast<PointerType>(Base->getType());
      auto *ST = BPT ? dyn_cast<StructType>(BPT->getElementType()) : nullptr;
      if (ST && ST->hasName() && ST->getName().startswith("class.hls::stream")) {
        Value *True = ConstantInt::get(CI->getType(), 1);
        CI->replaceAllUsesWith(True);
        CI->eraseFromParent();
        Changed = true;
        continue;
      }
    }
    std::string Name =
        std::string("_ssdm_op_IfCanRead.ap_hs.") + hlsrs::vxx::mangleForIntrinsic(PT);
    FunctionType *FT = FunctionType::get(Type::getInt1Ty(M.getContext()),
                                         {PT}, /*isVarArg=*/false);
    FunctionCallee Fn = M.getOrInsertFunction(Name, FT);
    if (auto *FF = dyn_cast<Function>(Fn.getCallee()))
      FF->addFnAttr(Attribute::NoUnwind);
    IRBuilder<> B(CI);
    Value *V = B.CreateCall(Fn, {Ptr});
    Value *Z = B.CreateZExt(V, CI->getType());
    CI->replaceAllUsesWith(Z);
    CI->eraseFromParent();
    Changed = true;
  }
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

// `Stream::empty()` (`__vxx_stream_empty(ptr) -> i32`) → the non-blocking
// FIFO probe `llvm.fpga.fifo.not.empty` (IntrinsicsFPGA.td), inverted. The
// C++ `while (!s.empty())` loop's pre-reflow form is the same probe as its
// `_ssdm_op_IfCanRead.Stream` branch condition. Runs AFTER the stream args
// are retyped (injectStreamInterface) so the pointer chain leads to the
// canonical stream object.
bool injectStreamEmpty(Module &M) {
  Function *Marker = M.getFunction("__vxx_stream_empty");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    if (CI->arg_size() < 1) continue;
    // Strip only BITCASTS — stripPointerCasts also walks zero-GEPs, which
    // would jump past the stream-field pointer to the enclosing struct.
    Value *Ptr = CI->getArgOperand(0);
    for (;;) {
      if (auto *BC = dyn_cast<BitCastInst>(Ptr)) { Ptr = BC->getOperand(0); continue; }
      if (auto *BO = dyn_cast<BitCastOperator>(Ptr)) { Ptr = BO->getOperand(0); continue; }
      break;
    }
    auto *PT = dyn_cast<PointerType>(Ptr->getType());
    if (!PT) continue;
    // The fifo intrinsics take the stream's INNER element pointer (field 0
    // of class.hls::stream — same shape as the pop/push `.inner` GEPs).
    if (auto *ST = dyn_cast<StructType>(PT->getElementType()))
      if (ST->hasName() && ST->getName().startswith("class.hls::stream") &&
          ST->getNumElements() >= 1) {
        IRBuilder<> GB(CI);
        Ptr = GB.CreateInBoundsGEP(
            ST, Ptr,
            {ConstantInt::get(Type::getInt64Ty(M.getContext()), 0),
             ConstantInt::get(Type::getInt32Ty(M.getContext()), 0)},
            Ptr->getName() + ".inner");
        PT = cast<PointerType>(Ptr->getType());
      }
    std::string Name = std::string("llvm.fpga.fifo.not.empty.") +
                       hlsrs::vxx::mangleForIntrinsic(PT);
    FunctionType *FT = FunctionType::get(Type::getInt1Ty(M.getContext()),
                                         {PT}, /*isVarArg=*/false);
    FunctionCallee Fn = M.getOrInsertFunction(Name, FT);
    if (auto *FF = dyn_cast<Function>(Fn.getCallee()))
      FF->addFnAttr(Attribute::NoUnwind);
    IRBuilder<> B(CI);
    Value *NE = B.CreateCall(Fn, {Ptr});
    Value *Empty = B.CreateXor(NE, ConstantInt::getTrue(M.getContext()));
    Value *Z = B.CreateZExt(Empty, CI->getType());
    CI->replaceAllUsesWith(Z);
    CI->eraseFromParent();
    Changed = true;
  }
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

// Reassemble the token-chain `stream()` builder into the legacy
// `__vxx_stream_depth(port, depth, kind)` marker, which injectStreamDepth (and
// the FFT path) then consume unchanged. The builder emits one marker per
// clause threaded by an SSA token: `__vxx_streamcfg_begin()` ->
// `_var(tok, ptr)` -> `_depth(tok, D)` -> `_type(tok, K)` (any order). Follow
// the chain from each begin; only if a depth clause is present do we emit the
// legacy marker (a bare `variable`-only chain is documentation-only, like the
// old bare `stream!`). Runs before injectStreamDepth. The port pointer flows
// through untouched (its `.variable()` marker kept the alloca pinned, so it is
// still the stream's address, not SROA-degenerated packed data).
bool injectStreamChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_streamcfg_begin");
  if (!Begin) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction("__vxx_streamcfg_var");
  Function *DepthMk = M.getFunction("__vxx_streamcfg_depth");
  Function *TypeMk = M.getFunction("__vxx_streamcfg_type");
  FunctionCallee DepthFn = M.getOrInsertFunction(
      "__vxx_stream_depth",
      FunctionType::get(Type::getVoidTy(Ctx), {I8P, I32, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(DepthFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI) continue;
    ToErase.push_back(BeginCI);
    Value *Tok = BeginCI;
    Value *Port = nullptr;
    uint64_t Depth = 0, Kind = 0;
    bool HaveDepth = false;
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok) continue;
        Function *Callee = CI->getCalledFunction();
        if (Callee == VarMk || Callee == DepthMk || Callee == TypeMk) {
          Next = CI;
          break;
        }
      }
      if (!Next) break;
      Function *Callee = Next->getCalledFunction();
      if (Callee == VarMk && Next->arg_size() >= 2) {
        Port = Next->getArgOperand(1);
      } else if (Callee == DepthMk && Next->arg_size() >= 2) {
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1))) {
          Depth = C->getZExtValue();
          HaveDepth = true;
        }
      } else if (Callee == TypeMk && Next->arg_size() >= 2) {
        if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1)))
          Kind = C->getZExtValue();
      }
      ToErase.push_back(Next);
      Tok = Next;
    }
    if (Port && HaveDepth) {
      IRBuilder<> B(BeginCI);
      // Match the legacy marker's i8* first arg.
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI =
          B.CreateCall(DepthFn, {P, ConstantInt::get(I32, Depth),
                                 ConstantInt::get(I32, Kind)});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty()) Begin->eraseFromParent();
  if (VarMk && VarMk->use_empty()) VarMk->eraseFromParent();
  if (DepthMk && DepthMk->use_empty()) DepthMk->eraseFromParent();
  if (TypeMk && TypeMk->use_empty()) TypeMk->eraseFromParent();
  return Changed;
}

bool injectStreamDepth(Module &M) {
  // `barista_hls::stream_depth_fifo(&stream, depth)` -> `hls::stream<T, DEPTH>`
  // equivalent: `_ssdm_SpecStream(ptr, i32 0, i32 DEPTH, [1 x i8]* @"")` at the
  // marker site (e.g. buf=4096/count=64).
  // Without it DATAFLOW internal FIFOs default to depth 2 -> producer blocks on
  // a full data FIFO before publishing the count -> xsim dependence-cycle
  // DEADLOCK (Loop_0 empty count_i <- Loop_2 full buf_i).
  Function *Marker = M.getFunction("__vxx_stream_depth");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_SpecStream", SpecTy);
  if (auto *FF = dyn_cast<Function>(SpecFn.getCallee()))
    FF->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  bool Changed = false;
  SmallVector<CallInst *, 8> Dead;
  SmallVector<CallInst *, 8> MarkerCalls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) MarkerCalls.push_back(CI);
  for (CallInst *CI : MarkerCalls) {
    if (CI->arg_size() < 2) continue;
    Value *Ptr = CI->getArgOperand(0);
    auto *DepthC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!DepthC) { Dead.push_back(CI); continue; }
    uint64_t Depth = DepthC->getZExtValue();
    // Erase the marker (and its feeding cast) BEFORE packing so the alloca's
    // user list only contains the real fifo traffic.
    Value *Root = Ptr->stripPointerCasts();
    auto *RootAI = dyn_cast<AllocaInst>(Root);
    Instruction *PtrI = dyn_cast<Instruction>(Ptr);
    CI->eraseFromParent();
    if (PtrI && PtrI != RootAI && PtrI->use_empty()) PtrI->eraseFromParent();
    Value *SpecPtr = Ptr;
    Instruction *InsertPt = nullptr;
    if (RootAI) {
      AllocaInst *Tgt = RootAI;
      // Multi-field struct stream: pack to a single iN FIFO (i65 form);
      // otherwise the HLS backend field-splits and drops the depth (HLS 200-805).
      if (AllocaInst *Packed = packStructStreamAlloca(M, RootAI)) Tgt = Packed;
      SpecPtr = Tgt;
      InsertPt = Tgt->getNextNode();
    } else if (auto *ArgV = dyn_cast<Argument>(Root)) {
      // Kernel-ARG stream depth (FFT array-interface ports: the inferred
      // stream interface needs an explicit depth for cosim — "A depth
      // specification is required for interface port 'xn'"). Emit the
      // SpecStream at function entry on the raw arg (C++ `#pragma HLS
      // stream variable=xn` + interface depth pragma equivalent).
      // SCALAR pointer args are NOT streamed arrays: a SpecStream on an
      // ap_fifo scalar ref (stream_good's `interface().mode(ap_fifo)
      // .port(&*d_i).depth(4)`) demotes the FIFO handshake to a plain
      // ap_vld scalar (Stage A: d_i_dout/empty_n/read ports vanish). Only
      // array/aggregate args take the arg-SpecStream path.
      {
        Type *Pointee = ArgV->getType()->isPointerTy()
                            ? ArgV->getType()->getPointerElementType()
                            : nullptr;
        // (structs excluded too: the ap_fifo scalar path retypes the arg to
        // `class.hls::stream<T>*`, which must NOT take the arg-SpecStream.)
        if (!Pointee || !Pointee->isArrayTy()) {
          Changed = true;
          continue;
        }
      }
      SpecPtr = ArgV;
      Function *PF = ArgV->getParent();
      if (PF && !PF->isDeclaration())
        InsertPt = &*PF->getEntryBlock().getFirstInsertionPt();
    }
    if (!InsertPt) { Changed = true; continue; }
    IRBuilder<> B(InsertPt);
    // Pass a SCALAR-typed ptr (packed iN element alloca, e.g.
    // i65* for {i64,bool}); a wrapper-struct ptr makes the HLS backend treat
    // the directive as an aggregate pragma -> HLS 214-319. Bitcast away the
    // aggregate pointee.
    Value *V = B.CreateBitCast(SpecPtr, Type::getInt8PtrTy(Ctx));
    B.CreateCall(SpecFn, {V, ConstantInt::get(I32, 0),
                          ConstantInt::get(I32, Depth),
                          EmptyStr});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}

// Marker-independent retype for cpp_proxy cosim (`__vxx_axis_packed` marker). By the
// time this runs the AXIS markers are erased and the read/write have been
// lowered to `fifo.pop/push.iN` (struct packed to integer) on a
// `class.hls::stream<StructT>*`. Vitis synthesises rust_<top>'s arg from that
// iN fifo intrinsic as a stream<iN>, which mismatches the C++
// `hls::stream<Struct>` element passed by cosim_top (HLS 214-136). Retype the
// fifo intrinsics (and the struct-write memcpy form) to the struct element so
// both sides agree. Keyed purely on the intrinsic + its stream-pointer pointee
// type, so it does not depend on marker survival through inlining.
// Strip rustc's repr(C) `[0 x T]` zero-size alignment markers AND a trailing
// `[K x i8]` pad array from a struct, leaving only the real scalar fields —
// e.g. `%Packet = {[0 x i32], i32, [0 x i8], i8, ..., [2 x i8]}` -> a named
// `{i32, i8×6}` matching the C++ clang `struct Packet`. HLS's stream solver
// rejects the zero-size array fields (HLS 214-309). Returns ST unchanged if it
// has no array fields. The result is byte-compatible (same scalar layout), so
// callers size-safe-bitcast the stream pointer to it.
static StructType *cleanFifoStruct(Module &M, StructType *ST) {
  SmallVector<Type *, 8> Fields;
  bool HadZsa = false;
  for (unsigned i = 0; i < ST->getNumElements(); i++) {
    Type *FT = ST->getElementType(i);
    // Drop ONLY the zero-size `[0 x T]` alignment markers (HLS 214-309). KEEP
    // any real trailing `[K x i8]` pad so the struct stays the SAME shape as
    // the globally-stripped `%Packet` stream element (stripZeroSizeArrayStruct
    // also keeps it) AND the C++ proxy struct (which emits a matching `_pad`
    // field). All three being identical 8-field structs is what keeps the HLS
    // prototype check happy (214-136) while staying 12 bytes (no 214-211).
    if (auto *AT = dyn_cast<ArrayType>(FT))
      if (AT->getNumElements() == 0) { HadZsa = true; continue; }
    Fields.push_back(FT);
  }
  if (!HadZsa)
    return ST;
  std::string Name = (ST->hasName() ? ST->getName().str() : std::string("axis")) + ".nopad";
  if (StructType *Existing = M.getTypeByName(Name))
    return Existing;
  return StructType::create(M.getContext(), Fields, Name);
}

bool retypeAxisPackedFifo(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    return false;
  const DataLayout &DL = M.getDataLayout();
  bool Changed = false;
  SmallVector<CallInst *, 16> Pops, Pushes;
  SmallVector<MemCpyInst *, 16> MemCpys;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        if (auto *MC = dyn_cast<MemCpyInst>(&I)) {
          MemCpys.push_back(MC);
          continue;
        }
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee)
          continue;
        StringRef N = Callee->getName();
        if (N.startswith("llvm.fpga.fifo.pop.i"))
          Pops.push_back(CI);
        else if (N.startswith("llvm.fpga.fifo.push.i"))
          Pushes.push_back(CI);
      }
  }
  // Reads: `iN = fifo.pop.iN(streamPtr)` -> struct pop + iN reconstruction.
  for (CallInst *CI : Pops) {
    auto *ILT = dyn_cast<IntegerType>(CI->getType());
    if (!ILT)
      continue;
    StructType *ST = streamElemStructOf(CI->getArgOperand(0));
    if (!ST || DL.getTypeAllocSizeInBits(ST) != ILT->getBitWidth())
      continue;
    StructType *CST = cleanFifoStruct(M, ST);
    IRBuilder<> B(CI);
    Value *STPtr =
        B.CreateBitCast(CI->getArgOperand(0), PointerType::get(CST, 0));
    Function *Pop = getOrInsertFifoPopAny(M, CST);
    CallInst *NewPop = B.CreateCall(Pop, {STPtr});
    Function *F = CI->getParent()->getParent();
    IRBuilder<> EntryB(&F->getEntryBlock(),
                       F->getEntryBlock().getFirstInsertionPt());
    AllocaInst *Tmp = EntryB.CreateAlloca(CST, nullptr, "axis.packed.read");
    B.CreateStore(NewPop, Tmp);
    Value *AsInt = B.CreateBitCast(Tmp, PointerType::get(ILT, 0));
    Value *IntVal = B.CreateLoad(ILT, AsInt);
    CI->replaceAllUsesWith(IntVal);
    CI->eraseFromParent();
    Changed = true;
  }
  // Writes (memcpy form): `memcpy(dst=stream<StructT>*, src, N)` is the real
  // write (the paired `fifo.push.iN` targets a dead local). Convert to
  // `fifo.push.<StructT>(load(src), dst)`.
  for (MemCpyInst *MC : MemCpys) {
    auto *LenC = dyn_cast<ConstantInt>(MC->getLength());
    if (!LenC)
      continue;
    uint64_t Bits = LenC->getZExtValue() * 8;
    // (a) memcpy whose dest is the stream wrapper itself = the real write ->
    //     `fifo.push.<Struct>(load(src), dst)`.
    if (StructType *ST = streamElemStructOf(MC->getDest())) {
      if (DL.getTypeAllocSizeInBits(ST) == Bits) {
        StructType *CST = cleanFifoStruct(M, ST);
        IRBuilder<> B(MC);
        Type *STPtrTy = PointerType::get(CST, 0);
        Value *SrcST = B.CreateBitCast(MC->getSource(), STPtrTy);
        Value *DstST = B.CreateBitCast(MC->getDest(), STPtrTy);
        Value *StructVal = B.CreateLoad(CST, SrcST);
        Function *Push = getOrInsertFifoPushAny(M, CST);
        B.CreateCall(Push, {StructVal, DstST});
        MC->eraseFromParent();
        Changed = true;
        continue;
      }
    }
  }
  // Writes: remaining `fifo.push.iN`. If the stream pointer is a real
  // `class.hls::stream<StructT>*`, retype it; if it targets a dead local alloca
  // (the SROA-split staging temp whose data the memcpy above already moved),
  // erase it so HLS does not see a push to an unconnected local FIFO.
  for (CallInst *CI : Pushes) {
    if (CI->arg_size() < 2)
      continue;
    auto *ILT = dyn_cast<IntegerType>(CI->getArgOperand(0)->getType());
    if (!ILT)
      continue;
    Value *Ptr = CI->getArgOperand(1);
    StructType *ST = streamElemStructOf(Ptr);
    if (ST && DL.getTypeAllocSizeInBits(ST) == ILT->getBitWidth()) {
      StructType *CST = cleanFifoStruct(M, ST);
      IRBuilder<> B(CI);
      Function *F = CI->getParent()->getParent();
      IRBuilder<> EntryB(&F->getEntryBlock(),
                         F->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Tmp = EntryB.CreateAlloca(CST, nullptr, "axis.packed.write");
      Value *AsInt = B.CreateBitCast(Tmp, PointerType::get(ILT, 0));
      B.CreateStore(CI->getArgOperand(0), AsInt);
      Value *StructVal = B.CreateLoad(CST, Tmp);
      Value *STPtr = B.CreateBitCast(Ptr, PointerType::get(CST, 0));
      Function *Push = getOrInsertFifoPushAny(M, CST);
      B.CreateCall(Push, {StructVal, STPtr});
      CI->eraseFromParent();
      Changed = true;
    } else {
      Value *Base = Ptr;
      while (true) {
        if (auto *BC = dyn_cast<BitCastInst>(Base)) { Base = BC->getOperand(0); continue; }
        if (auto *G = dyn_cast<GetElementPtrInst>(Base)) { Base = G->getPointerOperand(); continue; }
        break;
      }
      if (isa<AllocaInst>(Base)) {
        // Erase ONLY a push into a genuinely DEAD staging temp (the AXIS
        // SROA-split local whose data the memcpy above already moved — it has
        // no reader). An INTERNAL dataflow stream (`Stream::new()` alloca,
        // e.g. axi_stream_to_master's buf/count) is ALSO a plain alloca, but
        // it HAS `fifo.pop` consumers; erasing its push kills the producer
        // half of the FIFO → the RTL stream never fills → the downstream
        // m_axi write emits 0 beats → cosim hangs 0/1. Keep any push whose
        // alloca is popped somewhere (walk through casts/GEPs both ways).
        bool HasPopUser = false;
        SmallVector<Value *, 8> Work{Base};
        SmallPtrSet<Value *, 16> Seen;
        while (!Work.empty() && !HasPopUser) {
          Value *V = Work.pop_back_val();
          if (!Seen.insert(V).second) continue;
          for (User *U : V->users()) {
            if (auto *UC = dyn_cast<CallInst>(U)) {
              Function *UF = UC->getCalledFunction();
              if (UF && UF->getName().startswith("llvm.fpga.fifo.pop")) {
                HasPopUser = true;
                break;
              }
            }
            if (isa<BitCastInst>(U) || isa<GetElementPtrInst>(U))
              Work.push_back(U);
          }
        }
        if (!HasPopUser) {
          CI->eraseFromParent();
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

// cpp_proxy cosim (`__vxx_axis_packed`): convert the per-channel 14-arg
// `llvm.fpga.axis.pop/push` into a single whole-struct `fifo.pop/push.<axis>`.
// The per-channel form reads/writes each AXIS side-channel separately, which
// the cosim_top AXIS port can't reconcile (XFORM 203-801 "data pack only on
// source/destination"). A whole-packet read/write matches the C++
// `stream.read()/.write()` and the `hls::stream<ap_axis>` cosim_top port.
// Each channel arg is `hlsaxis_get_<chan>_ptr(base)`; we recover the common
// `base` (the `hls::axis` struct pointer) for the stream and the local temp.
bool rewriteAxisPopToWholeStruct(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    return false;
  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N.startswith("llvm.fpga.axis.pop.") ||
            N.startswith("llvm.fpga.axis.push."))
          Calls.push_back(CI);
      }
  }
  auto axisBase = [](Value *ChanPtr) -> Value * {
    if (auto *C = dyn_cast<CallInst>(ChanPtr))
      if (Function *F = C->getCalledFunction())
        if (F->getName().startswith("hlsaxis_get_") && C->arg_size() >= 1)
          return C->getArgOperand(0);
    return nullptr;
  };
  bool Changed = false;
  for (CallInst *CI : Calls) {
    unsigned NA = CI->arg_size();
    if (NA == 0 || (NA & 1))
      continue;
    unsigned Half = NA / 2;
    bool IsPush =
        CI->getCalledFunction()->getName().startswith("llvm.fpga.axis.push.");
    // first half = stream-side channels, last half = local-temp channels.
    Value *StreamBase = axisBase(CI->getArgOperand(0));
    Value *LocalBase = axisBase(CI->getArgOperand(Half));
    if (!StreamBase || !LocalBase)
      continue;
    auto *SPT = dyn_cast<PointerType>(StreamBase->getType());
    if (!SPT)
      continue;
    StructType *AxisST = dyn_cast<StructType>(SPT->getElementType());
    if (!AxisST)
      continue;
    IRBuilder<> B(CI);
    Type *AxisPtrTy = PointerType::get(AxisST, 0);
    Value *LocalP = LocalBase->getType() == AxisPtrTy
                        ? LocalBase
                        : B.CreateBitCast(LocalBase, AxisPtrTy);
    if (!IsPush) {
      Function *Pop = getOrInsertFifoPopAny(M, AxisST);
      CallInst *V = B.CreateCall(Pop, {StreamBase});
      B.CreateStore(V, LocalP);
    } else {
      Value *V = B.CreateLoad(AxisST, LocalP);
      Function *Push = getOrInsertFifoPushAny(M, AxisST);
      B.CreateCall(Push, {V, StreamBase});
    }
    // Collect the per-channel getter calls so we can drop the now-dead ones.
    SmallVector<Value *, 16> Operands(CI->arg_begin(), CI->arg_end());
    CI->eraseFromParent();
    for (Value *Op : Operands)
      if (auto *G = dyn_cast<CallInst>(Op))
        if (Function *GF = G->getCalledFunction())
          if (GF->getName().startswith("hlsaxis_get_") && G->use_empty())
            G->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Build a fully-scalar copy of `ST`: every `[1 x i8]` array field AND every
// single-`i8` nested struct (rustc's `AxisDisabled` 0-width-channel placeholder
// `{ i8 }`) is replaced by a plain `i8`. All three are byte-identical (size 1,
// align 1, field index/offset unchanged), so the FIFO width and all field GEPs
// stay valid. Reflow's array-to-stream EntiretyAccessCheck SIGSEGVs when the
// stream element is a non-flat aggregate (arrays / nested structs), so a flat
// `{ i8.., i16, i8.. }` element keeps it happy. Returns null if nothing changes.
static StructType *noArrayFifoStruct(Module &M, StructType *ST) {
  SmallVector<Type *, 12> Fields;
  bool Changed = false;
  Type *I8 = Type::getInt8Ty(M.getContext());
  for (unsigned i = 0; i < ST->getNumElements(); i++) {
    Type *FT = ST->getElementType(i);
    if (auto *AT = dyn_cast<ArrayType>(FT))
      if (AT->getNumElements() == 1 && AT->getElementType()->isIntegerTy(8)) {
        Fields.push_back(I8);
        Changed = true;
        continue;
      }
    if (auto *NST = dyn_cast<StructType>(FT))
      if (NST->getNumElements() == 1 && NST->getElementType(0)->isIntegerTy(8)) {
        Fields.push_back(I8);
        Changed = true;
        continue;
      }
    Fields.push_back(FT);
  }
  if (!Changed)
    return nullptr;
  std::string Name =
      (ST->hasName() ? ST->getName().str() : std::string("fifo")) + ".noarr";
  if (StructType *Ex = M.getTypeByName(Name))
    return Ex;
  return StructType::create(M.getContext(), Fields, Name);
}

// cpp_proxy cosim (`__vxx_axis_packed`): a plain `#[repr(C)]` packet stream (e.g.
// `ap_axis_user!`'s `%Packet`) reaches the generic struct-fifo path as
// `fifo.pop/push.<%Packet>` WITHOUT being canonicalised to `hls::axis`. rustc's
// repr(C) layout interleaves `[1 x i8]` alignment-pad arrays, and reflow's
// `EntiretyAccessCheck` ("array-to-stream") SIGSEGVs (HLS 200-1715) on ANY array
// field inside a stream element. Re-pop/push such structs as a byte-identical
// array-free clone (`[1 x i8]` -> `i8`). Skip canonical `hls::axis<...>` streams
// (the ap_axis! path) — their C++ proxy type is the real AXIS type and must not
// be re-shaped. Marker-independent; runs after the struct fifo ops are final.
bool cleanStructFifoArrays(Module &M) {
  if (!hlsrs::vxx::markerUsed(M, "__vxx_axis_packed"))
    return false;
  auto skipName = [](StructType *ST) {
    return ST->hasName() && ST->getName().contains("hls::axis");
  };
  SmallVector<CallInst *, 16> Pops, Pushes;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef N = CI->getCalledFunction()->getName();
        if (N.startswith("llvm.fpga.fifo.pop."))
          Pops.push_back(CI);
        else if (N.startswith("llvm.fpga.fifo.push."))
          Pushes.push_back(CI);
      }
  }
  bool Changed = false;
  for (CallInst *CI : Pops) {
    if (CI->arg_size() != 1)
      continue;
    auto *ST = dyn_cast<StructType>(CI->getType());
    if (!ST || skipName(ST))
      continue;
    StructType *CST = noArrayFifoStruct(M, ST);
    if (!CST)
      continue;
    Function *F = CI->getParent()->getParent();
    IRBuilder<> B(CI);
    IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
    AllocaInst *Slot = EB.CreateAlloca(CST, nullptr, "fifo.noarr.read");
    Value *CPtr =
        B.CreateBitCast(CI->getArgOperand(0), PointerType::getUnqual(CST));
    Function *Pop = getOrInsertFifoPopAny(M, CST);
    Value *NewV = B.CreateCall(Pop, {CPtr});
    B.CreateStore(NewV, Slot);
    Value *Repl =
        B.CreateLoad(ST, B.CreateBitCast(Slot, PointerType::getUnqual(ST)));
    CI->replaceAllUsesWith(Repl);
    CI->eraseFromParent();
    Changed = true;
  }
  for (CallInst *CI : Pushes) {
    if (CI->arg_size() != 2)
      continue;
    Value *Val = CI->getArgOperand(0);
    auto *ST = dyn_cast<StructType>(Val->getType());
    if (!ST || skipName(ST))
      continue;
    StructType *CST = noArrayFifoStruct(M, ST);
    if (!CST)
      continue;
    Function *F = CI->getParent()->getParent();
    IRBuilder<> B(CI);
    IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
    AllocaInst *Tmp = EB.CreateAlloca(ST, nullptr, "fifo.noarr.write");
    B.CreateStore(Val, Tmp);
    Value *CVal =
        B.CreateLoad(CST, B.CreateBitCast(Tmp, PointerType::getUnqual(CST)));
    Value *CPtr =
        B.CreateBitCast(CI->getArgOperand(1), PointerType::getUnqual(CST));
    Function *Push = getOrInsertFifoPushAny(M, CST);
    B.CreateCall(Push, {CVal, CPtr});
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}


// Un-merge a select-of-stream-pointers feeding llvm.fpga.fifo.pop/push back
// into per-case branches. rustc's -O pipeline sinks the two identical
// branch-arm reads of `if sel == 0 { in0.read() } else { in1.read() }` into
// one call behind `select i1 %c, i32* %in0, i32* %in1` — but the HLS backend
// cannot trace a FIFO through a select (the historical "select-pointer"
// limitation), while the C++ pp form keeps one IfRead per case block
// (free_running_kernel_remerge_ii4to1's mux: V.i2.case.0/1). Rebuilding the
// diamond puts each fifo op on a concrete stream pointer, exactly the C++
// input shape.
bool splitSelectPtrFifoOps(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 8> Work;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) continue;
        StringRef N = CI->getCalledFunction()->getName();
        bool IsPop = N.startswith("llvm.fpga.fifo.pop");
        bool IsPush = N.startswith("llvm.fpga.fifo.push");
        if (!IsPop && !IsPush) continue;
        unsigned PtrIdx = IsPop ? 0 : 1;
        if (isa<SelectInst>(CI->getArgOperand(PtrIdx))) Work.push_back(CI);
      }
  }
  for (CallInst *CI : Work) {
    StringRef N = CI->getCalledFunction()->getName();
    bool IsPop = N.startswith("llvm.fpga.fifo.pop");
    unsigned PtrIdx = IsPop ? 0 : 1;
    auto *SI = cast<SelectInst>(CI->getArgOperand(PtrIdx));
    Instruction *ThenT = nullptr, *ElseT = nullptr;
    SplitBlockAndInsertIfThenElse(SI->getCondition(), CI, &ThenT, &ElseT);
    auto cloneTo = [&](Instruction *Term, Value *Ptr) -> CallInst * {
      auto *C = cast<CallInst>(CI->clone());
      C->setArgOperand(PtrIdx, Ptr);
      C->insertBefore(Term);
      return C;
    };
    CallInst *TC = cloneTo(ThenT, SI->getTrueValue());
    CallInst *EC = cloneTo(ElseT, SI->getFalseValue());
    if (IsPop) {
      IRBuilder<> B(CI);
      PHINode *Phi = B.CreatePHI(CI->getType(), 2);
      Phi->addIncoming(TC, TC->getParent());
      Phi->addIncoming(EC, EC->getParent());
      CI->replaceAllUsesWith(Phi);
    }
    CI->eraseFromParent();
    if (SI->use_empty()) SI->eraseFromParent();
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: split select-ptr fifo ops (" << Work.size() << ")\n";
  return Changed;
}

} } // namespace hlsrs::vxx
