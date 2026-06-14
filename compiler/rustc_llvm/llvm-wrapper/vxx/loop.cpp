//===----------------------------------------------------------------------===//
//
// loop.cpp — loop pragma passes: unroll / pipeline / flatten / trip-count, plus the array_stencil + GEP-shaping helpers.
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
#include "llvm/ADT/STLExtras.h"  // function_ref, is_contained (phase-2 chains)
#include "vxx_llvm.h"  // consolidated common LLVM headers

#include <string>

using namespace llvm;
using namespace hlsrs::vxx;

namespace hlsrs { namespace vxx {

// For every `__vxx_loop_unroll(i32 factor)` marker call, find the
// enclosing LLVM loop and attach
//   !llvm.loop !{ !self, !{!"llvm.loop.unroll.count", i64 factor, !"user", null} }
// to the latch terminator. This is the metadata shape the HLS backend expects
// for `#pragma HLS UNROLL factor=N`.
bool injectLoopUnroll(Module &M) {
  Function *Marker = M.getFunction("__vxx_loop_unroll");
  if (!Marker)
    return false;

  SmallVector<CallInst *, 8> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);

  // Group call-sites by enclosing function so we only build LoopInfo
  // once per function. (LoopInfoWrapperPass is a function pass and used
  // to be requested via getAnalysis<>; here we re-create it on demand.)
  DenseMap<Function *, SmallVector<CallInst *, 4>> ByFunc;
  for (CallInst *CI : Calls)
    ByFunc[CI->getParent()->getParent()].push_back(CI);

  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  for (auto &KV : ByFunc) {
    Function *F = KV.first;
    if (F->isDeclaration())
      continue;
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);

    for (CallInst *CI : KV.second) {
      auto *CFactor = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (!CFactor) {
        vxxDbg() << "vxx: non-constant __vxx_loop_unroll at " << *CI
               << "\n";
        continue;
      }
      uint64_t Factor = CFactor->getZExtValue();

      Loop *L = LI.getLoopFor(CI->getParent());
      if (!L) {
        vxxDbg() << "vxx: __vxx_loop_unroll not inside a loop in "
               << F->getName() << "\n";
        continue;
      }

      BasicBlock *Latch = L->getLoopLatch();
      if (!Latch) {
        vxxDbg() << "vxx: loop has no single latch in " << F->getName()
               << "\n";
        continue;
      }
      Instruction *Term = Latch->getTerminator();

      // Build !llvm.loop metadata in the standard 2-operand shape that
      // LoopUnroll's `unrollCountPragmaValue` expects:
      //   !N = distinct !{!N, !{!"llvm.loop.unroll.count", i32 Factor}}
      TempMDTuple TempSelf = MDNode::getTemporary(Ctx, None);
      Metadata *UnrollElems[] = {
          MDString::get(Ctx, "llvm.loop.unroll.count"),
          ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(Ctx), Factor)),
      };
      MDNode *UnrollMD = MDNode::get(Ctx, UnrollElems);
      // MERGE with any operands already on the loop (name / tripcount /
      // pipeline / flatten from sibling directives) — a fresh 2-op node here
      // silently discarded them (template_function lost its pipeline, lmem
      // its flatten).
      SmallVector<Metadata *, 4> LoopElems;
      LoopElems.push_back(TempSelf.get());
      if (MDNode *Existing = Term->getMetadata("llvm.loop"))
        for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
          LoopElems.push_back(Existing->getOperand(i));
      LoopElems.push_back(UnrollMD);
      MDNode *LoopMD = MDNode::getDistinct(Ctx, LoopElems);
      LoopMD->replaceOperandWith(0, LoopMD); // self-referencing

      Term->setMetadata("llvm.loop", LoopMD);
      vxxDbg() << "vxx: unroll factor " << Factor << " on loop in "
             << F->getName() << "\n";
      Changed = true;
    }
  }

  for (CallInst *CI : Calls)
    CI->eraseFromParent();
  return Changed;
}



// For `__vxx_ap_wait()` markers: lower each call to `_ssdm_op_Wait(i32 1)`,
// the lowering of `ap_wait()`. This
// inserts a scheduling barrier so an earlier side effect (e.g. an m_axi store)
// completes before a following one (e.g. a `Stream::write` gating a concurrent
// DATAFLOW reader) — without it the reader can observe stale memory.
bool injectApWait(Module &M) {
  Function *Marker = M.getFunction("__vxx_ap_wait");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *WaitTy =
      FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee WaitFn = M.getOrInsertFunction("_ssdm_op_Wait", WaitTy);
  SmallVector<CallInst *, 4> Dead;
  unsigned Emitted = 0;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    IRBuilder<> B(CI);
    B.CreateCall(WaitFn, {ConstantInt::get(I32, 1)});
    Dead.push_back(CI);
    ++Emitted;
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  if (Emitted)
    vxxDbg() << "vxx: lowered " << Emitted
           << " ap_wait marker(s) to _ssdm_op_Wait\n";
  return Emitted > 0;
}



// For `__vxx_loop_pipeline(II, off, rewind)` markers: attach
// `!llvm.loop.pipeline.enable = !{!"...", i64 II, i8 rewind, ...}` metadata
// to the loop's backedge branch. This matches the standard LLVM HLS loop
// metadata expected by OSS HLS clang's loop pipeline pragma.
bool injectLoopPipelineMetadata(Module &M) {
  bool Changed = false;
  Function *Marker = M.getFunction("__vxx_loop_pipeline");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  SmallVector<CallInst *, 4> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    Dead.push_back(CI);
    auto *IIC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    int II = IIC ? (int)IIC->getZExtValue() : 0;
    // arg1 (optional): off flag. `#pragma HLS pipeline off` lowers to the
    // same metadata with II = 0; a bare `pipeline()` (ii = 0) means "auto"
    // and must map to II = -1 (the infer form), NOT 0.
    bool Off = false;
    if (CI->arg_size() > 1)
      if (auto *OC = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
        Off = OC->getZExtValue() != 0;
    II = Off ? 0 : (II == 0 ? -1 : II);
    // arg2 (optional): rewind flag. `#pragma HLS pipeline rewind` lowers to
    // the first i8 operand = 1 (C++ a.pp.bc: !{..., i64 -1, i8 1, i8 -1, ...}).
    bool Rewind = false;
    if (CI->arg_size() > 2)
      if (auto *RC = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
        Rewind = RC->getZExtValue() != 0;
    Function *F = CI->getParent()->getParent();
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    BasicBlock *Header = nullptr;
    for (Loop *L : LI) {
      if (L->contains(CI->getParent())) {
        // Find the deepest containing loop.
        Loop *Cur = L;
        while (true) {
          bool Descended = false;
          for (Loop *Sub : Cur->getSubLoops()) {
            if (Sub->contains(CI->getParent())) {
              Cur = Sub;
              Descended = true;
              break;
            }
          }
          if (!Descended)
            break;
        }
        Header = Cur->getHeader();
        break;
      }
    }
    if (!Header)
      continue;
    // The backedge branch carries the !llvm.loop metadata.
    BasicBlock *Latch = nullptr;
    for (BasicBlock *Pred : predecessors(Header)) {
      if (DT.dominates(Header, Pred)) {
        Latch = Pred;
        break;
      }
    }
    if (!Latch)
      continue;
    Instruction *Term = Latch->getTerminator();
    if (!Term)
      continue;
    // The `llvm.loop.pipeline.enable` metadata uses 6
    // operands (vs the 3-op shape stock LLVM uses):
    //   !{!"llvm.loop.pipeline.enable",
    //     i64 II, i8 disable, i8 ?, i64 ?, !"user", !<source-ref>}
    // The trailing 4 operands are mandatory or the HLS backend's
    // "Lower intermediate type generated by HLSGen" pass crashes
    // (visited via the "dump pragma info via xml format" path).
    auto *I64 = Type::getInt64Ty(Ctx);
    auto *I8 = Type::getInt8Ty(Ctx);
    SmallVector<Metadata *, 4> Elts;
    Elts.push_back(nullptr); // self-ref placeholder
    // MERGE with existing loop operands (unroll.count etc.) — see unroll.
    if (MDNode *Existing = Term->getMetadata("llvm.loop"))
      for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
        Elts.push_back(Existing->getOperand(i));
    Elts.push_back(MDNode::get(
        Ctx, {MDString::get(Ctx, "llvm.loop.pipeline.enable"),
              ConstantAsMetadata::get(
                  ConstantInt::get(I64, II < 0 ? -1 : II)),
              ConstantAsMetadata::get(ConstantInt::get(I8, Rewind ? 1 : 0)),
              ConstantAsMetadata::get(ConstantInt::get(I8, -1)),
              ConstantAsMetadata::get(ConstantInt::get(I64, -1)),
              MDString::get(Ctx, "user")}));
    MDNode *LoopID = MDNode::get(Ctx, Elts);
    LoopID->replaceOperandWith(0, LoopID);
    Term->setMetadata("llvm.loop", LoopID);
    Changed = true;
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  return Changed;
}

// Token-chain form of the pipeline directive (the `pipeline()` builder,
// mirroring the trip-count chain): `__vxx_pipeline_begin()` ->
// `__vxx_pipeline_ii(tok, N)` / `_off(tok)` / `_rewind(tok)` (any subset).
// The begin marker's position anchors the loop; clause values reach their
// markers as immediate constants (the old single-call Drop form read the
// builder fields back from memory, which opt-level=1 does not const-fold —
// that silently dropped `rewind`). Emits the identical 6-operand
// `llvm.loop.pipeline.enable` metadata as injectLoopPipelineMetadata.
bool injectLoopPipelineChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_pipeline_begin");
  if (!Begin)
    return false;
  Function *IIMk = M.getFunction("__vxx_pipeline_ii");
  Function *OffMk = M.getFunction("__vxx_pipeline_off");
  Function *RewMk = M.getFunction("__vxx_pipeline_rewind");
  LLVMContext &Ctx = M.getContext();
  auto *I64 = Type::getInt64Ty(Ctx);
  auto *I8 = Type::getInt8Ty(Ctx);
  SmallVector<CallInst *, 8> BeginCalls;
  for (User *U : Begin->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      BeginCalls.push_back(CI);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (CallInst *BeginCI : BeginCalls) {
    ToErase.push_back(BeginCI);
    // Collect clauses along the token chain.
    int II = 0;
    bool Off = false, Rewind = false;
    Value *Tok = BeginCI;
    while (Tok) {
      CallInst *Next = nullptr;
      for (User *TU : Tok->users()) {
        auto *CI = dyn_cast<CallInst>(TU);
        if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok)
          continue;
        Function *Callee = CI->getCalledFunction();
        if ((IIMk && Callee == IIMk) || (OffMk && Callee == OffMk) ||
            (RewMk && Callee == RewMk)) {
          Next = CI;
          break;
        }
      }
      if (!Next)
        break;
      Function *Callee = Next->getCalledFunction();
      if (Callee == IIMk) {
        if (Next->arg_size() >= 2)
          if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1)))
            II = (int)C->getZExtValue();
      } else if (Callee == OffMk) {
        Off = true;
      } else {
        Rewind = true;
      }
      ToErase.push_back(Next);
      Tok = Next;
    }
    // `pipeline off` -> II = 0; bare `pipeline()` -> II = -1 (infer), same
    // as the direct-marker form.
    II = Off ? 0 : (II == 0 ? -1 : II);
    Function *F = BeginCI->getParent()->getParent();
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    Loop *L = LI.getLoopFor(BeginCI->getParent());
    if (!L)
      continue;
    BasicBlock *Header = L->getHeader();
    BasicBlock *Latch = nullptr;
    for (BasicBlock *Pred : predecessors(Header)) {
      if (DT.dominates(Header, Pred)) {
        Latch = Pred;
        break;
      }
    }
    if (!Latch)
      continue;
    Instruction *Term = Latch->getTerminator();
    if (!Term)
      continue;
    SmallVector<Metadata *, 4> Elts;
    Elts.push_back(nullptr); // self-ref placeholder
    if (MDNode *Existing = Term->getMetadata("llvm.loop"))
      for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
        Elts.push_back(Existing->getOperand(i));
    Elts.push_back(MDNode::get(
        Ctx, {MDString::get(Ctx, "llvm.loop.pipeline.enable"),
              ConstantAsMetadata::get(
                  ConstantInt::get(I64, II < 0 ? -1 : II)),
              ConstantAsMetadata::get(ConstantInt::get(I8, Rewind ? 1 : 0)),
              ConstantAsMetadata::get(ConstantInt::get(I8, -1)),
              ConstantAsMetadata::get(ConstantInt::get(I64, -1)),
              MDString::get(Ctx, "user")}));
    MDNode *LoopID = MDNode::get(Ctx, Elts);
    LoopID->replaceOperandWith(0, LoopID);
    Term->setMetadata("llvm.loop", LoopID);
    Changed = true;
  }
  // Erase consumer-before-producer (chain tail first).
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  for (Function *Mk : {IIMk, OffMk, RewMk})
    if (Mk && Mk->use_empty())
      Mk->eraseFromParent();
  return Changed;
}



// For `__vxx_dependence(ptr, class, type, direction, distance, dependent)`
// markers (`#pragma HLS DEPENDENCE variable= class= type= direction=
// distance= dependent=`): emit the `_ssdm_SpecDependence` call the HLS
// clang produces. Ground truth (ram_uram `variable=buffer inter WAR false`):
//   _ssdm_SpecDependence(ptr, i32 0, i32 0, i32 1, i64 0, i32 1, i1 true)
// — (class 0 = unspecified, type 0 = inter, direction 1 = WAR (RAW 0 /
// WAR 1 / WAW 2), distance 0, i32 1, and the final i1 is the *false-
// dependence* assertion: `dependent=false` → true).
bool injectDependenceSpec(llvm::Module &M) {
  Function *Marker = M.getFunction("__vxx_dependence");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn = M.getOrInsertFunction("_ssdm_SpecDependence", SpecTy);
  SmallVector<CallInst *, 4> Dead;
  bool Changed = false;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 6)
      continue;
    Dead.push_back(CI);
    auto Arg = [&](unsigned I) -> uint64_t {
      if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(I)))
        return C->getZExtValue();
      return 0;
    };
    Value *Ptr = CI->getArgOperand(0)->stripPointerCasts();
    IRBuilder<> B(CI);
    Value *Args[] = {
        Ptr,
        ConstantInt::get(I32, Arg(1)),            // class
        ConstantInt::get(I32, Arg(2)),            // type (0=inter 1=intra)
        ConstantInt::get(I32, Arg(3)),            // direction (0 RAW/1 WAR/2 WAW)
        ConstantInt::get(I64, Arg(4)),            // distance
        ConstantInt::get(I32, 1),
        ConstantInt::get(I1, Arg(5) == 0 ? 1 : 0) // dependent=false → i1 true
    };
    B.CreateCall(SpecFn, Args);
    Changed = true;
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

// For `__vxx_stable(ptr)` markers (`#pragma HLS stable variable=<v>`): emit
// `_ssdm_op_SpecStableContent(ptr, i1 true, [1 x i8] zeroinitializer)`, the
// HLS clang lowering (using_maxi_in_tasks a.pp.bc ground truth).
bool injectStableSpec(llvm::Module &M) {
  Function *Marker = M.getFunction("__vxx_stable");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I1 = Type::getInt1Ty(Ctx);
  Constant *EmptyArr =
      ConstantAggregateZero::get(ArrayType::get(Type::getInt8Ty(Ctx), 1));
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn =
      M.getOrInsertFunction("_ssdm_op_SpecStableContent", SpecTy);
  SmallVector<CallInst *, 4> Dead;
  bool Changed = false;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 1)
      continue;
    Dead.push_back(CI);
    Value *Ptr = CI->getArgOperand(0)->stripPointerCasts();
    IRBuilder<> B(CI);
    Value *Args[] = {Ptr, ConstantInt::get(I1, 1), EmptyArr};
    B.CreateCall(SpecFn, Args);
    Changed = true;
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

// ── Token-chain reassembly for phase-2 pragma builders ──────────────────
//
// The `dependence()` / `stable()` / `loop_flatten()` builders emit one marker
// per clause threaded by an SSA token (see marker.rs, phase-2 block). Each pass
// below follows the chain from its `__vxx_<fam>_begin` marker and reassembles
// the LEGACY marker call (`__vxx_dependence` / `__vxx_stable` /
// `__vxx_loop_flatten`) at the begin site, which the existing consumer above
// lowers unchanged. Runs BEFORE that consumer; both marker forms are accepted
// during migration. Uniform with the phase-1 cache/stream/tripcount chains.

// Walk the token chain rooted at Tok, invoking Visit(Callee, CI) for each clause
// CallInst whose first arg is the running token and whose callee is in Clauses.
// Appends every visited clause call to ToErase (tail order). Stops at the first
// token with no matching successor.
static void walkClauseChain(Value *Tok,
                            ArrayRef<Function *> Clauses,
                            SmallVectorImpl<CallInst *> &ToErase,
                            function_ref<void(Function *, CallInst *)> Visit) {
  while (Tok) {
    CallInst *Next = nullptr;
    for (User *TU : Tok->users()) {
      auto *CI = dyn_cast<CallInst>(TU);
      if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok)
        continue;
      Function *Callee = CI->getCalledFunction();
      if (llvm::is_contained(Clauses, Callee)) {
        Next = CI;
        break;
      }
    }
    if (!Next)
      break;
    Visit(Next->getCalledFunction(), Next);
    ToErase.push_back(Next);
    Tok = Next;
  }
}

// dependence() token chain → legacy `__vxx_dependence(ptr, class, type,
// direction, distance, dependent)`. Emitted only if a `variable=` clause is
// present. Defaults match the old Drop-builder (class 0, type 0=inter,
// direction 0, distance 0, dependent 1=true).
bool injectDependenceChain(llvm::Module &M) {
  Function *Begin = M.getFunction("__vxx_dep_begin");
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction("__vxx_dep_var");
  Function *TypeMk = M.getFunction("__vxx_dep_type");
  Function *DirMk = M.getFunction("__vxx_dep_direction");
  Function *DistMk = M.getFunction("__vxx_dep_distance");
  Function *DepMk = M.getFunction("__vxx_dep_dependent");
  SmallVector<Function *, 5> Clauses;
  for (Function *F : {VarMk, TypeMk, DirMk, DistMk, DepMk})
    if (F)
      Clauses.push_back(F);
  FunctionCallee DepFn = M.getOrInsertFunction(
      "__vxx_dependence",
      FunctionType::get(Type::getVoidTy(Ctx),
                        {I8P, I32, I32, I32, I32, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(DepFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    Value *Port = nullptr;
    uint64_t Ty = 0, Dir = 0, Dist = 0, Dep = 1;
    // Insert the reassembled marker at the LAST clause (where every clause
    // operand — notably the `variable=` pointer, computed at its clause site —
    // dominates), matching the old Drop-builder's end-of-statement position.
    CallInst *InsertPt = BeginCI;
    walkClauseChain(
        BeginCI, Clauses, ToErase, [&](Function *C, CallInst *CI) {
          InsertPt = CI;
          if (CI->arg_size() < 2)
            return;
          if (C == VarMk) {
            Port = CI->getArgOperand(1);
          } else if (auto *K = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
            uint64_t V = K->getZExtValue();
            if (C == TypeMk)
              Ty = V;
            else if (C == DirMk)
              Dir = V;
            else if (C == DistMk)
              Dist = V;
            else if (C == DepMk)
              Dep = V;
          }
        });
    if (Port) {
      IRBuilder<> B(InsertPt);
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI = B.CreateCall(
          DepFn, {P, ConstantInt::get(I32, 0), ConstantInt::get(I32, Ty),
                  ConstantInt::get(I32, Dir), ConstantInt::get(I32, Dist),
                  ConstantInt::get(I32, Dep)});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  for (Function *F : {VarMk, TypeMk, DirMk, DistMk, DepMk})
    if (F && F->use_empty())
      F->eraseFromParent();
  return Changed;
}

// stable() token chain → legacy `__vxx_stable(ptr)` (only if `variable=` set).
bool injectStableChain(llvm::Module &M) {
  Function *Begin = M.getFunction("__vxx_stable_begin");
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction("__vxx_stable_var");
  SmallVector<Function *, 1> Clauses;
  if (VarMk)
    Clauses.push_back(VarMk);
  FunctionCallee StableFn = M.getOrInsertFunction(
      "__vxx_stable",
      FunctionType::get(Type::getVoidTy(Ctx), {I8P}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(StableFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    Value *Port = nullptr;
    CallInst *InsertPt = BeginCI;
    walkClauseChain(BeginCI, Clauses, ToErase,
                    [&](Function *C, CallInst *CI) {
                      InsertPt = CI;
                      if (C == VarMk && CI->arg_size() >= 2)
                        Port = CI->getArgOperand(1);
                    });
    if (Port) {
      IRBuilder<> B(InsertPt);
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI = B.CreateCall(StableFn, {P});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  if (VarMk && VarMk->use_empty())
    VarMk->eraseFromParent();
  return Changed;
}

// loop_flatten() token chain → legacy `__vxx_loop_flatten(enable)`. A bare
// chain means enable=1; an `off` clause sets enable=0.
bool injectLoopFlattenChain(llvm::Module &M) {
  Function *Begin = M.getFunction("__vxx_flatten_begin");
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Function *OffMk = M.getFunction("__vxx_flatten_off");
  SmallVector<Function *, 1> Clauses;
  if (OffMk)
    Clauses.push_back(OffMk);
  FunctionCallee FlattenFn = M.getOrInsertFunction(
      "__vxx_loop_flatten",
      FunctionType::get(Type::getVoidTy(Ctx), {I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(FlattenFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    uint64_t Enable = 1;
    walkClauseChain(BeginCI, Clauses, ToErase,
                    [&](Function *C, CallInst *) {
                      if (C == OffMk)
                        Enable = 0;
                    });
    IRBuilder<> B(BeginCI);
    CallInst *NewCI = B.CreateCall(FlattenFn, {ConstantInt::get(I32, Enable)});
    NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
    Changed = true;
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  if (OffMk && OffMk->use_empty())
    OffMk->eraseFromParent();
  return Changed;
}

// ── Phase-2 token chains for the struct-layout / storage pragma builders ────
//
// `array_partition()` / `array_reshape()` / `bind_storage()` move to the same
// token-chain form as batch A. Unlike batch A (whose consumers all live in the
// Vitis lowering stage), these markers are consumed EARLY — the arg-decay
// trigger and `disaggCompletePartitionKernelSig` read `__vxx_array_partition`,
// and `redirectWholeStructBindStorage` reads `__vxx_bind_storage`, both in the
// shape stage. So their chains are reassembled at the TOP of vxxShapeIR (before
// any consumer, before SROA), which reproduces the old Drop-builder's emission
// byte-for-byte: the clause markers keep the port alloca alive until then, and
// the legacy marker is emitted at the last clause node (dominance-safe). Both
// marker forms are accepted during migration.

// Shared reassembly for the `array_partition` / `array_reshape` builders, which
// share the exact `(ptr, kind, factor, dim)` legacy-marker shape. Defaults
// match the builder (kind=2=complete, factor=0, dim=0). Emitted only when a
// `variable=` clause pinned a port.
static bool injectPartitionLikeChain(llvm::Module &M, StringRef BeginName,
                                     StringRef VarName, StringRef TypeName,
                                     StringRef FactorName, StringRef DimName,
                                     StringRef LegacyName) {
  Function *Begin = M.getFunction(BeginName);
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction(VarName);
  Function *TypeMk = M.getFunction(TypeName);
  Function *FactorMk = M.getFunction(FactorName);
  Function *DimMk = M.getFunction(DimName);
  SmallVector<Function *, 4> Clauses;
  for (Function *F : {VarMk, TypeMk, FactorMk, DimMk})
    if (F)
      Clauses.push_back(F);
  FunctionCallee Fn = M.getOrInsertFunction(
      LegacyName, FunctionType::get(Type::getVoidTy(Ctx),
                                    {I8P, I32, I32, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(Fn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    Value *Port = nullptr;
    uint64_t Kind = 2, Factor = 0, Dim = 0;
    CallInst *InsertPt = BeginCI;
    walkClauseChain(
        BeginCI, Clauses, ToErase, [&](Function *C, CallInst *CI) {
          InsertPt = CI;
          if (CI->arg_size() < 2)
            return;
          if (C == VarMk) {
            Port = CI->getArgOperand(1);
          } else if (auto *K = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
            uint64_t V = K->getZExtValue();
            if (C == TypeMk)
              Kind = V;
            else if (C == FactorMk)
              Factor = V;
            else if (C == DimMk)
              Dim = V;
          }
        });
    if (Port) {
      IRBuilder<> B(InsertPt);
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI =
          B.CreateCall(Fn, {P, ConstantInt::get(I32, Kind),
                            ConstantInt::get(I32, Factor),
                            ConstantInt::get(I32, Dim)});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  for (Function *F : {VarMk, TypeMk, FactorMk, DimMk})
    if (F && F->use_empty())
      F->eraseFromParent();
  return Changed;
}

bool injectArrayPartitionChain(llvm::Module &M) {
  return injectPartitionLikeChain(M, "__vxx_apart_begin", "__vxx_apart_var",
                                  "__vxx_apart_type", "__vxx_apart_factor",
                                  "__vxx_apart_dim", "__vxx_array_partition");
}

bool injectArrayReshapeChain(llvm::Module &M) {
  return injectPartitionLikeChain(M, "__vxx_reshape_begin", "__vxx_reshape_var",
                                  "__vxx_reshape_type", "__vxx_reshape_factor",
                                  "__vxx_reshape_dim", "__vxx_array_reshape");
}

// bind_storage() token chain → legacy `__vxx_bind_storage(ptr, code, latency)`.
// The clause chain threads `type=` and `impl=` separately (faithful UG1399
// clause order); this reassembly folds the (type, impl) pair into the same
// discriminant the old Drop-builder computed. Defaults: type=0/impl=0 → auto
// (code 0), latency=-1. Emitted only when a `variable=` clause pinned a port.
bool injectBindStorageChain(llvm::Module &M) {
  Function *Begin = M.getFunction("__vxx_bindstg_begin");
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction("__vxx_bindstg_var");
  Function *TypeMk = M.getFunction("__vxx_bindstg_type");
  Function *ImplMk = M.getFunction("__vxx_bindstg_impl");
  Function *LatMk = M.getFunction("__vxx_bindstg_latency");
  SmallVector<Function *, 4> Clauses;
  for (Function *F : {VarMk, TypeMk, ImplMk, LatMk})
    if (F)
      Clauses.push_back(F);
  FunctionCallee Fn = M.getOrInsertFunction(
      "__vxx_bind_storage",
      FunctionType::get(Type::getVoidTy(Ctx), {I8P, I32, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(Fn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  // (type, impl) → legacy discriminant. Mirrors BindStorage::drop in lib.rs.
  auto foldCode = [](uint64_t Ty, uint64_t Im) -> int64_t {
    switch (Ty * 100 + Im) {
    case 100: return 17; case 101: return 18; case 102: return 19; case 103: return 20; // ram_1p
    case 200: return 21; case 201: return 22; case 202: return 23; case 203: return 24; // ram_2p
    case 301: return 25; case 306: return 26; case 302: return 27; case 303: return 28; case 307: return 29; // ram_s2p
    case 401: return 30; case 403: return 31;                                            // ram_t2p
    case 500: return 32; case 501: return 33; case 502: return 34;                       // rom_1p
    case 605: return 7;  case 601: return 8;  case 602: return 9;  case 604: return 10; case 603: return 11; // fifo
    default: return 0; // unmapped pair -> auto
    }
  };
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    Value *Port = nullptr;
    uint64_t Ty = 0, Im = 0;
    int64_t Lat = -1;
    CallInst *InsertPt = BeginCI;
    walkClauseChain(
        BeginCI, Clauses, ToErase, [&](Function *C, CallInst *CI) {
          InsertPt = CI;
          if (CI->arg_size() < 2)
            return;
          if (C == VarMk) {
            Port = CI->getArgOperand(1);
          } else if (auto *K = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
            if (C == TypeMk)
              Ty = K->getZExtValue();
            else if (C == ImplMk)
              Im = K->getZExtValue();
            else if (C == LatMk)
              Lat = K->getSExtValue();
          }
        });
    if (Port) {
      IRBuilder<> B(InsertPt);
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI = B.CreateCall(
          Fn, {P, ConstantInt::get(I32, foldCode(Ty, Im)),
               ConstantInt::get(I32, Lat)});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  for (Function *F : {VarMk, TypeMk, ImplMk, LatMk})
    if (F && F->use_empty())
      F->eraseFromParent();
  return Changed;
}

// array_stencil() token chain → legacy `__vxx_array_stencil(ptr, 0)`. Single
// `variable=` clause. Reassembled at the top of vxxShapeIR so the many
// consumers (loop/axilite/maxi/vxx_prep, both stages) see it unchanged.
bool injectArrayStencilChain(llvm::Module &M) {
  Function *Begin = M.getFunction("__vxx_stencil_begin");
  if (!Begin)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8P = Type::getInt8PtrTy(Ctx);
  Function *VarMk = M.getFunction("__vxx_stencil_var");
  SmallVector<Function *, 1> Clauses;
  if (VarMk)
    Clauses.push_back(VarMk);
  FunctionCallee Fn = M.getOrInsertFunction(
      "__vxx_array_stencil",
      FunctionType::get(Type::getVoidTy(Ctx), {I8P, I32}, /*Var=*/false));
  if (auto *F = dyn_cast<Function>(Fn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (User *U : Begin->users()) {
    auto *BeginCI = dyn_cast<CallInst>(U);
    if (!BeginCI)
      continue;
    ToErase.push_back(BeginCI);
    Value *Port = nullptr;
    CallInst *InsertPt = BeginCI;
    walkClauseChain(BeginCI, Clauses, ToErase,
                    [&](Function *C, CallInst *CI) {
                      InsertPt = CI;
                      if (C == VarMk && CI->arg_size() >= 2)
                        Port = CI->getArgOperand(1);
                    });
    if (Port) {
      IRBuilder<> B(InsertPt);
      Value *P = Port->getType() == I8P ? Port : B.CreateBitCast(Port, I8P);
      CallInst *NewCI = B.CreateCall(Fn, {P, ConstantInt::get(I32, 0)});
      NewCI->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
      Changed = true;
    }
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  if (VarMk && VarMk->use_empty())
    VarMk->eraseFromParent();
  return Changed;
}

// NB: aggregate(), disaggregate() and alias() are kept as Drop-builders — no
// token-chain reassembly. Their consumers need the marker's port operand to
// resolve to the already-retyped AXIS / m_axi / ap_fifo arg, which a reassembled
// marker doesn't provide (disagg → port unsplit HLS 214-244; alias → VXXPrep
// crash; aggregate on an ap_fifo struct port → FIFO degrades to BRAM, Stage A DIFF).


// For `__vxx_loop_tripcount(min, max, avg)` markers (the explicit
// `#pragma HLS LOOP_TRIPCOUNT min= max= avg=`): attach
// `!{"llvm.loop.tripcount", i64 min, max, avg, "user"}` to the enclosing loop's
// latch, preserving any existing `llvm.loop` operands (unroll.count /
// pipeline.enable). An omitted avg (0) defaults to (min+max)/2, matching the
// HLS clang lowering of a `LOOP_TRIPCOUNT` pragma without an `avg=` clause
// (`_ssdm_op_SpecLoopTripCount(1, 64, 32)` for min=1 max=64). This is the
// ONLY source of trip counts — VXXPrep never auto-derives them (a constant
// bound is computed by HLS itself; a runtime bound is declared here by the
// user, exactly as in the C++ baseline).
bool injectLoopTripCount(Module &M) {
  Function *Marker = M.getFunction("__vxx_loop_tripcount");
  if (!Marker)
    return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  DenseMap<Function *, SmallVector<CallInst *, 4>> ByFunc;
  for (CallInst *CI : Calls)
    ByFunc[CI->getParent()->getParent()].push_back(CI);

  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  bool Changed = false;
  for (auto &KV : ByFunc) {
    Function *F = KV.first;
    if (F->isDeclaration())
      continue;
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    for (CallInst *CI : KV.second) {
      auto *CMin = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      auto *CMax = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (!CMin || !CMax) {
        vxxDbg() << "vxx: non-constant __vxx_loop_tripcount at " << *CI << "\n";
        continue;
      }
      Loop *L = LI.getLoopFor(CI->getParent());
      if (!L) {
        vxxDbg() << "vxx: __vxx_loop_tripcount not inside a loop in "
                 << F->getName() << "\n";
        continue;
      }
      BasicBlock *Latch = L->getLoopLatch();
      if (!Latch)
        continue;
      Instruction *Term = Latch->getTerminator();
      uint64_t Min = CMin->getZExtValue();
      uint64_t Max = CMax->getZExtValue();
      // arg2: avg. 0 (or a legacy 2-arg marker) = unset → (min+max)/2, the
      // same default HLS clang computes for a pragma without `avg=`.
      uint64_t Avg = 0;
      if (CI->getNumArgOperands() >= 3)
        if (auto *CAvg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
          Avg = CAvg->getZExtValue();
      if (Avg == 0)
        Avg = (Min + Max) / 2;
      ConstantAsMetadata *MinM =
          ConstantAsMetadata::get(ConstantInt::get(I64, Min));
      ConstantAsMetadata *MaxM =
          ConstantAsMetadata::get(ConstantInt::get(I64, Max));
      ConstantAsMetadata *AvgM =
          ConstantAsMetadata::get(ConstantInt::get(I64, Avg));
      MDNode *TripMD = MDNode::get(
          Ctx, {MDString::get(Ctx, "llvm.loop.tripcount"), MinM, MaxM, AvgM,
                MDString::get(Ctx, "user")});
      SmallVector<Metadata *, 4> Args;
      Args.push_back(nullptr); // self-ref placeholder
      if (MDNode *Existing = Term->getMetadata("llvm.loop"))
        for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
          Args.push_back(Existing->getOperand(i));
      Args.push_back(TripMD);
      MDNode *NewID = MDNode::getDistinct(Ctx, Args);
      NewID->replaceOperandWith(0, NewID);
      Term->setMetadata("llvm.loop", NewID);
      Changed = true;
    }
  }
  for (CallInst *CI : Calls)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

// Token-chain form of the trip-count directive (the `loop_tripcount()`
// builder). Each clause is a marker threading an SSA token:
// `__vxx_tripcount_begin()` -> `__vxx_tripcount_min(tok, N)` /
// `_max(tok, N)` / `_avg(tok, N)` (any subset, any order). The begin marker's
// position anchors the loop (same as the direct marker); follow the token
// def-use chain to collect min/max/avg, then attach the identical
// `!{"llvm.loop.tripcount", ...}` metadata. Values reach their markers as
// constants (inline builder methods), so they read as ConstantInt here.
bool injectLoopTripCountChain(Module &M) {
  Function *Begin = M.getFunction("__vxx_tripcount_begin");
  if (!Begin)
    return false;
  Function *MinMk = M.getFunction("__vxx_tripcount_min");
  Function *MaxMk = M.getFunction("__vxx_tripcount_max");
  Function *AvgMk = M.getFunction("__vxx_tripcount_avg");
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  SmallVector<CallInst *, 8> BeginCalls;
  for (User *U : Begin->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      BeginCalls.push_back(CI);
  DenseMap<Function *, SmallVector<CallInst *, 4>> ByFunc;
  for (CallInst *CI : BeginCalls)
    ByFunc[CI->getParent()->getParent()].push_back(CI);

  SmallVector<CallInst *, 16> ToErase;
  bool Changed = false;
  for (auto &KV : ByFunc) {
    Function *F = KV.first;
    if (F->isDeclaration())
      continue;
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    for (CallInst *BeginCI : KV.second) {
      ToErase.push_back(BeginCI);
      // Collect clauses along the token chain.
      uint64_t Min = 0, Max = 0, Avg = 0;
      Value *Tok = BeginCI;
      while (Tok) {
        CallInst *Next = nullptr;
        for (User *TU : Tok->users()) {
          auto *CI = dyn_cast<CallInst>(TU);
          if (!CI || CI->arg_size() < 1 || CI->getArgOperand(0) != Tok)
            continue;
          Function *Callee = CI->getCalledFunction();
          if (Callee == MinMk || Callee == MaxMk || Callee == AvgMk) {
            Next = CI;
            break;
          }
        }
        if (!Next)
          break;
        Function *Callee = Next->getCalledFunction();
        if (Next->arg_size() >= 2)
          if (auto *C = dyn_cast<ConstantInt>(Next->getArgOperand(1))) {
            if (Callee == MinMk)
              Min = C->getZExtValue();
            else if (Callee == MaxMk)
              Max = C->getZExtValue();
            else
              Avg = C->getZExtValue();
          }
        ToErase.push_back(Next);
        Tok = Next;
      }
      Loop *L = LI.getLoopFor(BeginCI->getParent());
      if (!L)
        continue;
      BasicBlock *Latch = L->getLoopLatch();
      if (!Latch)
        continue;
      Instruction *Term = Latch->getTerminator();
      if (Avg == 0)
        Avg = (Min + Max) / 2;
      MDNode *TripMD = MDNode::get(
          Ctx, {MDString::get(Ctx, "llvm.loop.tripcount"),
                ConstantAsMetadata::get(ConstantInt::get(I64, Min)),
                ConstantAsMetadata::get(ConstantInt::get(I64, Max)),
                ConstantAsMetadata::get(ConstantInt::get(I64, Avg)),
                MDString::get(Ctx, "user")});
      SmallVector<Metadata *, 4> Args;
      Args.push_back(nullptr); // self-ref placeholder
      if (MDNode *Existing = Term->getMetadata("llvm.loop"))
        for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
          Args.push_back(Existing->getOperand(i));
      Args.push_back(TripMD);
      MDNode *NewID = MDNode::getDistinct(Ctx, Args);
      NewID->replaceOperandWith(0, NewID);
      Term->setMetadata("llvm.loop", NewID);
      Changed = true;
    }
  }
  // Erase consumer-before-producer (chain tail first).
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
  if (Begin->use_empty())
    Begin->eraseFromParent();
  if (MinMk && MinMk->use_empty())
    MinMk->eraseFromParent();
  if (MaxMk && MaxMk->use_empty())
    MaxMk->eraseFromParent();
  if (AvgMk && AvgMk->use_empty())
    AvgMk->eraseFromParent();
  return Changed;
}

// For `__vxx_loop_name(ptr, len)` markers (a C++ loop label, harvested from a
// real Rust loop label `'SUM_LOOP: for ...` by the #[barista_hls::top] macro):
// attach !{"llvm.loop.name", "SUM_LOOP"} to the enclosing loop's latch,
// preserving existing operands. Runs BEFORE injectAutoLoopName, whose HasName
// check then skips the loop — so the user label wins over the synthetic
// VITIS_LOOP_N and the pre-reflow IR carries the same SpecLoopName string as
// the C++ baseline (clang keeps source labels).
static StringRef loopNameFromMarkerArg(Value *V) {
  V = V->stripPointerCasts();
  auto *GV = dyn_cast<GlobalVariable>(V);
  if (!GV)
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::GetElementPtr)
        GV = dyn_cast<GlobalVariable>(CE->getOperand(0)->stripPointerCasts());
  if (!GV || !GV->hasInitializer())
    return "";
  Constant *Init = GV->getInitializer();
  // rustc wraps string literals in a packed single-field struct:
  //   <{ [9 x i8] }> <{ [9 x i8] c"SUM_LOOP\00" }>
  if (auto *CS = dyn_cast<ConstantStruct>(Init))
    if (CS->getNumOperands() == 1)
      Init = CS->getOperand(0);
  if (auto *CDA = dyn_cast<ConstantDataArray>(Init)) {
    if (CDA->isCString())
      return CDA->getAsCString();
    if (CDA->isString())
      return CDA->getAsString();
  }
  return "";
}

bool injectLoopUserName(Module &M) {
  Function *Marker = M.getFunction("__vxx_loop_name");
  if (!Marker)
    return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  DenseMap<Function *, SmallVector<CallInst *, 4>> ByFunc;
  for (CallInst *CI : Calls)
    ByFunc[CI->getParent()->getParent()].push_back(CI);

  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  for (auto &KV : ByFunc) {
    Function *F = KV.first;
    if (F->isDeclaration())
      continue;
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    for (CallInst *CI : KV.second) {
      StringRef Name = loopNameFromMarkerArg(CI->getArgOperand(0));
      // trim a trailing NUL from a non-cstring byte literal
      if (!Name.empty() && Name.back() == '\0')
        Name = Name.drop_back();
      if (CI->arg_size() >= 2)
        if (auto *CLen = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
          if (CLen->getZExtValue() < Name.size())
            Name = Name.take_front(CLen->getZExtValue());
      if (Name.empty()) {
        vxxDbg() << "vxx: __vxx_loop_name without a static string in "
                 << F->getName() << "\n";
        continue;
      }
      Loop *L = LI.getLoopFor(CI->getParent());
      if (!L) {
        vxxDbg() << "vxx: __vxx_loop_name not inside a loop in "
                 << F->getName() << "\n";
        continue;
      }
      BasicBlock *Latch = L->getLoopLatch();
      if (!Latch)
        continue;
      Instruction *Term = Latch->getTerminator();
      MDNode *NameMD = MDNode::get(
          Ctx, {MDString::get(Ctx, "llvm.loop.name"), MDString::get(Ctx, Name)});
      SmallVector<Metadata *, 4> Args;
      Args.push_back(nullptr); // self-ref placeholder
      if (MDNode *Existing = Term->getMetadata("llvm.loop"))
        for (unsigned i = 1; i < Existing->getNumOperands(); ++i) {
          // drop any earlier name (user label is authoritative)
          if (MDNode *Op = dyn_cast_or_null<MDNode>(Existing->getOperand(i)))
            if (Op->getNumOperands() >= 1)
              if (MDString *S = dyn_cast_or_null<MDString>(Op->getOperand(0)))
                if (S->getString() == "llvm.loop.name")
                  continue;
          Args.push_back(Existing->getOperand(i));
        }
      Args.push_back(NameMD);
      MDNode *NewID = MDNode::getDistinct(Ctx, Args);
      NewID->replaceOperandWith(0, NewID);
      Term->setMetadata("llvm.loop", NewID);
      vxxDbg() << "vxx: loop name '" << Name << "' in " << F->getName()
               << "\n";
      Changed = true;
    }
  }
  for (CallInst *CI : Calls)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

// For `__vxx_performance(target_ti, target_tl)` markers (`#pragma HLS
// performance target_ti= [target_tl=]`): emit an `_ssdm_op_SpecPerformance`
// call at the marker (i.e. in the enclosing loop body), matching the C++
// pre-reflow shape
//   _ssdm_op_SpecPerformance(i32 1, i64 ti, i64 tl, i64 0, i64 0, i64 0, i64 0,
//                            i32 0, [1 x i8]* @"")
// Scheduling-only: leaves ports and functional behaviour unchanged.
bool injectPerformance(Module &M) {
  Function *Marker = M.getFunction("__vxx_performance");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecPerfFn =
      M.getOrInsertFunction("_ssdm_op_SpecPerformance", SpecTy);
  if (auto *F = dyn_cast<Function>(SpecPerfFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  SmallVector<CallInst *, 8> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    uint64_t Ti = 0, Tl = 0;
    if (CI->arg_size() >= 1)
      if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
        Ti = C->getZExtValue();
    if (CI->arg_size() >= 2)
      if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
        Tl = C->getZExtValue();
    IRBuilder<> Bld(CI);
    Bld.SetCurrentDebugLocation(DebugLoc());
    Value *Args[] = {
        ConstantInt::get(I32, 1),
        ConstantInt::get(I64, Ti),  ConstantInt::get(I64, Tl),
        ConstantInt::get(I64, 0),   ConstantInt::get(I64, 0),
        ConstantInt::get(I64, 0),   ConstantInt::get(I64, 0),
        ConstantInt::get(I32, 0),   EmptyStr};
    Bld.CreateCall(SpecPerfFn, Args);
    CI->eraseFromParent();
  }
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return !Calls.empty();
}

// For `__vxx_function_instantiate(v)` markers (`#pragma HLS FUNCTION_INSTANTIATE
// variable=v`): emit `llvm.sideeffect() [ "fpga.func.instantiate"(<v>) ]` at the
// marker — the pre-reflow shape OSS clang emits (with `"xlx.source"="user"`).
// Reflow then specialises the enclosing (INLINE OFF / #[inline(never)])
// function per constant value of `v` at its call sites. Area/faithfulness only:
// ports and functional behaviour are unchanged.
bool injectFunctionInstantiate(Module &M) {
  Function *Marker = M.getFunction("__vxx_function_instantiate");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    if (CI->arg_size() < 1) {
      CI->eraseFromParent();
      continue;
    }
    Value *V = CI->getArgOperand(0);
    Function *EnclosingFn = CI->getFunction();
    IRBuilder<> B(CI);
    OperandBundleDef OBD("fpga.func.instantiate", ArrayRef<Value *>{V});
    CallInst *SE = B.CreateCall(SEFn, {}, {OBD});
    SE->setOnlyAccessesInaccessibleMemory();
    SE->setDoesNotThrow();
    SE->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(Ctx, "xlx.source", "user"));
    // Reflow keys the FUNCTION_INSTANTIATE specialisation on the enclosing
    // function carrying `!fpga.function.pragma` (the C++ baseline puts the
    // `fpga.inline` record here); the LLVM noinline attr alone is not enough.
    // Attach it if absent (the top fn already has its own).
    if (!EnclosingFn->getMetadata("fpga.function.pragma")) {
      MDNode *FpgaInlinePragma = MDNode::get(
          Ctx, {MDNode::get(Ctx, {MDString::get(Ctx, "fpga.inline"),
                                  MDString::get(Ctx, "user"),
                                  (Metadata *)nullptr})});
      EnclosingFn->setMetadata("fpga.function.pragma", FpgaInlinePragma);
    }
    CI->eraseFromParent();
  }
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return !Calls.empty();
}

// Auto-inject `_ssdm_op_SpecPipeline + SpecLoopTripCount + SpecLoopName`
// at every loop body entry in top kernels.  These are emitted
// on loop bodies of top fns even without `#pragma HLS pipeline` —
// HLS's scheduler relies on them to estimate latency.  Without them
// kernels like ecc_flags / burst_rw trip the `PerformanceInfo::
// getEstimatedSchedLatency` segfault during csynth.
//
// Skip loops that already carry SpecPipeline.
// Emit _ssdm_op_SpecLoopTripCount(N,N,N) for each countable loop in the top
// function, derived from a constant loop bound.
//
// Why this exists: when the IR has NO trip count, the HLS backend's
// auto-loop-pipeline ("HLS 214-376 Pipelining loop<unknown>") pipelines a loop
// whose trip count it cannot resolve, and the post-schedule LatencyEstimator
// null-derefs → csynth SIGSEGV. Emitting SpecPipeline +
// SpecLoopTripCount(-1,-1,-1) (unknown) crashes identically.
// The fix is to give the loop a REAL constant trip count and
// let the backend auto-pipeline; we do NOT pre-emit SpecPipeline (the backend
// owns that, and pre-emitting with bad operands was itself a crash trigger). Loops
// whose constant trip count cannot be resolved are skipped (no marker), since only
// constant-bounded loops get infer-from-design counts.
bool injectAutoLoopName(Module &M) {
  LLVMContext &Ctx = M.getContext();
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionCallee SpecTripFn =
      M.getOrInsertFunction("_ssdm_op_SpecLoopTripCount", SpecTy);
  FunctionCallee SpecLoopNameFn =
      M.getOrInsertFunction("_ssdm_op_SpecLoopName", SpecTy);
  FunctionCallee SpecPipeFn =
      M.getOrInsertFunction("_ssdm_op_SpecPipeline", SpecTy);
  for (FunctionCallee FC : {SpecTripFn, SpecLoopNameFn, SpecPipeFn})
    if (auto *F = dyn_cast<Function>(FC.getCallee()))
      F->addFnAttr(Attribute::NoUnwind);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  (void)EmptyStr;

  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("fpga.top.func")) continue;
    // Skip trip-count emit for functions binding a uram_ecc memory (kind 29).
    // HLS also derives the constant loop bound, so our SpecLoopTripCount is a
    // duplicate (XFORM 203-561) — harmless for plain BRAM, but on a uram_ecc
    // URAM it makes the LatencyEstimator SIGSEGV (ecc_flags). Only uram_ecc
    // (kind 29) uses this, so no other example is affected. The countable bound
    // is still derived by HLS; we just don't add the redundant directive.
    // This pass runs BEFORE injectBindStorageSpecResource, so the bind is still
    // the `__vxx_bind_storage(ptr, kind, latency)` marker — check kind (arg 1).
    {
      bool HasUramEcc = false;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          if (CI->getCalledFunction()->getName() != "__vxx_bind_storage")
            continue;
          if (CI->arg_size() >= 2)
            if (auto *K = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
              if (K->getZExtValue() == 29) { HasUramEcc = true; break; }
        }
        if (HasUramEcc) break;
      }
      if (HasUramEcc) continue;
    }
    DominatorTree DT(F);
    LoopInfo LI;
    LI.analyze(DT);
    if (LI.empty()) continue;

    SmallVector<Loop *, 8> AllLoops;
    for (Loop *L : LI) {
      AllLoops.push_back(L);
      SmallVector<Loop *, 8> Stack(L->getSubLoops().begin(),
                                    L->getSubLoops().end());
      while (!Stack.empty()) {
        Loop *Sub = Stack.pop_back_val();
        AllLoops.push_back(Sub);
        Stack.append(Sub->getSubLoops().begin(), Sub->getSubLoops().end());
      }
    }
    unsigned LoopIdx = 0;
    for (Loop *L : AllLoops) {
      BasicBlock *Header = L->getHeader();
      if (!Header) continue;
      // Trip count ONLY from a reliable SCEV constant. Do NOT guess from the
      // size of an array the loop happens to index — that is WRONG in general
      // (e.g. array_partition_block_cyclic indexes a factor-16 partitioned array
      // but the loop runs 256: a guessed tripcount=16 made HLS clamp the loop
      // "Updating loop upper bound from 256 to 16" → corrupt design + crash).
      // When SCEV can't bound the loop we emit the NAME only (which is what
      // fixes the LatencyEstimator null-deref); HLS computes the real bound and
      // a variable-length m_axi burst is fine. If a
      // constant LOOP_TRIPCOUNT is wanted it must come from an explicit
      // barista_hls::loop_tripcount marker in the source.
      // Emit loop name as !llvm.loop METADATA on the latch — a
      // single `!llvm.loop` node carrying
      // !{"llvm.loop.name","vadd"} + !{"llvm.loop.tripcount", i64 N,N,N,"user"}.
      // The HLS backend lowers this into a SINGLE, correctly-placed
      // _ssdm_op_SpecLoopName/SpecLoopTripCount call AND — crucially — when it
      // unrolls the loop (UNROLL FACTOR=2 on vadd) it does NOT duplicate the
      // directive, because the latch metadata is the authoritative anchor.
      //
      // The previous form emitted the Spec* calls directly in the loop BODY.
      // When the backend unrolled the vadd loop, the body call was COPIED into
      // the unrolled epilogue → two SpecLoopName for the same conceptual loop →
      // the scheduler built inconsistent PerformanceInfo → LatencyEstimator
      // null-deref SIGSEGV (lmem_2rw / decimator / aliasing crash cluster).
      // Metadata on the latch is unroll-safe, so it round-trips identically.
      //
      // The NAME is the crash-critical directive (an unnamed loop → null
      // PerformanceInfo → LatencyEstimator SIGSEGV), so emit it for EVERY loop
      // that doesn't already have one. The TRIP COUNT is separate: skip emitting
      // it if the loop already carries one (explicit barista_hls::loop_tripcount
      // body call, or a prior llvm.loop.tripcount) to avoid XFORM 203-561
      // "multiple trip count directives".
      bool HasName = false;
      if (MDNode *LID = L->getLoopID())
        for (unsigned i = 1; i < LID->getNumOperands(); ++i)
          if (MDNode *Op = dyn_cast_or_null<MDNode>(LID->getOperand(i)))
            if (Op->getNumOperands() >= 1)
              if (MDString *S = dyn_cast_or_null<MDString>(Op->getOperand(0)))
                if (S->getString() == "llvm.loop.name") HasName = true;
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          if (CI->getCalledFunction()->getName() == "_ssdm_op_SpecLoopName")
            HasName = true;
        }
      }
      if (HasName) continue; // already handled (don't double-name)
      // setLoopID needs a single dedicated latch (LoopSimplify form, which
      // simplifyKernelLoops ensures for top kernels).
      if (!L->getLoopLatch()) continue;
      std::string Name = "VITIS_LOOP_" + std::to_string(LoopIdx++);
      SmallVector<Metadata *, 6> IDArgs;
      IDArgs.push_back(nullptr); // self-ref placeholder (operand 0)
      if (MDNode *LID = L->getLoopID())
        for (unsigned i = 1; i < LID->getNumOperands(); ++i)
          IDArgs.push_back(LID->getOperand(i)); // preserve unroll.count etc.
      IDArgs.push_back(MDNode::get(
          Ctx, {MDString::get(Ctx, "llvm.loop.name"), MDString::get(Ctx, Name)}));
      // NOTE: the trip count is intentionally NOT auto-derived here. A constant
      // loop bound is computed by HLS itself; a runtime bound must be declared
      // explicitly with `barista_hls::loop_tripcount(min, max)` (lowered by
      // injectLoopTripCount, which runs before this pass and leaves a
      // `llvm.loop.tripcount` operand that the loop-operand copy above
      // preserves). This pass only guarantees every top-fn loop has a NAME, the
      // crash-critical directive for the latency estimator.
      MDNode *NewID = MDNode::getDistinct(Ctx, IDArgs);
      NewID->replaceOperandWith(0, NewID);
      L->setLoopID(NewID);
      Changed = true;
    }
  }
  return Changed;
}



// __vxx_loop_flatten(on): turn into loop metadata "llvm.loop.flatten.enable"
// on the enclosing loop's latch. arg0 = 1 for `loop_flatten`, 0 for
// `loop_flatten off`. Merges with existing loop operands.
bool injectLoopFlatten(Module &M) {
  Function *F = M.getFunction("__vxx_loop_flatten");
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  for (CallInst *CI : Calls) {
    uint64_t On = 1;
    if (CI->arg_size() >= 1)
      if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
        On = C->getZExtValue();
    Function *Fn = CI->getFunction();
    DominatorTree DT(*Fn);
    LoopInfo LI;
    LI.analyze(DT);
    Loop *L = LI.getLoopFor(CI->getParent());
    if (!L) {
      CI->eraseFromParent();
      continue;
    }
    BasicBlock *Latch = L->getLoopLatch();
    if (!Latch) {
      CI->eraseFromParent();
      continue;
    }
    Instruction *Term = Latch->getTerminator();
    MDNode *FlattenMD = MDNode::get(
        Ctx, {MDString::get(Ctx, "llvm.loop.flatten.enable"),
              ConstantAsMetadata::get(
                  ConstantInt::get(Type::getInt1Ty(Ctx), On != 0))});
    SmallVector<Metadata *, 4> Elts;
    Elts.push_back(nullptr); // self-ref placeholder
    if (MDNode *Existing = Term->getMetadata("llvm.loop"))
      for (unsigned i = 1; i < Existing->getNumOperands(); ++i)
        Elts.push_back(Existing->getOperand(i));
    Elts.push_back(FlattenMD);
    MDNode *NewID = MDNode::getDistinct(Ctx, Elts);
    NewID->replaceOperandWith(0, NewID);
    Term->setMetadata("llvm.loop", NewID);
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}



// Stamp `llvm.loop.mustprogress` on every loop backedge in stencil-marked
// functions. Without mustprogress the HLS LatencyEstimator cannot prove the
// pipelined leaf loop terminates → the leaf is not kept as a separately-scheduled
// module → the top's `getEstimatedSchedLatency` for the (now-inlined-but-
// unscheduled) call hits a null PerformanceInfo → SIGSEGV after "Finished
// scheduling". Add it so the outlined `_Pipeline_` leaf functions' loop metadata
// carries `!{self, startloc, endloc, llvm.loop.mustprogress, name, tripcount}`.
bool stampStencilLoopMustProgress(Module &M) {
  LLVMContext &Ctx = M.getContext();
  // The `__vxx_array_stencil` marker is consumed earlier by injectArrayStencil;
  // detect stencil kernels via the persistent `fpga_array_stencil` op-bundle.
  SmallPtrSet<Function *, 4> StencilFns;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        for (unsigned BI = 0; BI < CI->getNumOperandBundles(); ++BI)
          if (CI->getOperandBundleAt(BI).getTagName() == "fpga_array_stencil")
            StencilFns.insert(&F);
      }
    }
  }
  if (StencilFns.empty())
    return false;

  // Helpers to build legacy (pre-LLVM-3.7) debug-descriptor MDNodes as PLAIN
  // metadata. We do NOT use DIBuilder / typed DI nodes: rustc strips all modern
  // debug info after VXXPrep for the fpga64 target, and the Xilinx LLVM-11
  // parser rejects `!dbg` function attachments. The legacy `i32 786xxx`-tagged
  // tuples are ordinary MDNodes that rustc does not recognize as debug (so it
  // leaves them alone) and that the HLS backend consumes as loop source-locations.
  // The 786443 lexical-block nodes are what the backend uses to KEEP an outlined
  // pipeline leaf as a separately-scheduled module instead of force-inlining it
  // into the top — which is the latency-estimator-null-deref crash. The target
  // loop-md shape is `!{self, locStart, locEnd, mustprogress, name, tripcount}`.
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  auto mdI32 = [&](int64_t v) -> Metadata * {
    return ConstantAsMetadata::get(ConstantInt::get(I32, v, /*signed=*/true));
  };
  auto mdI64 = [&](int64_t v) -> Metadata * {
    return ConstantAsMetadata::get(ConstantInt::get(I64, v));
  };
  auto mdI1 = [&](bool v) -> Metadata * {
    return ConstantAsMetadata::get(ConstantInt::get(I1, v));
  };

  bool Changed = false;
  for (Function *F : StencilFns) {
    // Per-function shared legacy descriptors (file + subprogram + type).
    // File:  !{i32 786473 (DW_TAG_file_type), "name", "dir", null}
    MDNode *FileMD = MDNode::get(
        Ctx, {mdI32(786473), MDString::get(Ctx, "filter2d_hw.rs"),
              MDString::get(Ctx, "."), nullptr});
    // Subroutine type: !{i32 786453 (DW_TAG_subroutine_type), ...}
    MDNode *TypeArr = MDNode::get(Ctx, {nullptr});
    MDNode *SubrTy = MDNode::get(
        Ctx, {mdI32(786453), mdI32(0), MDString::get(Ctx, ""), mdI32(0),
              mdI32(0), mdI64(0), mdI64(0), mdI64(0), mdI32(0), nullptr,
              TypeArr, mdI32(0), mdI32(0)});
    MDNode *EmptyVars = MDNode::get(Ctx, {});
    // Subprogram: !{i32 786478 (DW_TAG_subprogram), 0, file, name, displayname,
    //   linkagename(null), file, line, type, isLocal, isDef, virt, virtidx,
    //   containingType(null), flags, isOpt, function*, tmpl(null), decl(null),
    //   variables, scopeLine}. References the (top) function — exactly as C++.
    MDNode *SP = MDNode::get(
        Ctx,
        {mdI32(786478), mdI32(0), FileMD, MDString::get(Ctx, F->getName()),
         MDString::get(Ctx, F->getName()), nullptr, FileMD, mdI32(22), SubrTy,
         mdI1(false), mdI1(true), mdI32(0), mdI32(0), nullptr, mdI32(256),
         mdI1(false), ConstantAsMetadata::get(F), nullptr, nullptr, EmptyVars,
         mdI32(26)});

    unsigned LoopBase = 40;
    for (BasicBlock &BB : *F) {
      Instruction *Term = BB.getTerminator();
      if (!Term)
        continue;
      MDNode *LoopMD = Term->getMetadata(LLVMContext::MD_loop);
      if (!LoopMD || LoopMD->getNumOperands() == 0)
        continue;
      // Skip if already stamped (loc node present → mustprogress/loc done).
      bool Has = false;
      for (unsigned i = 1; i < LoopMD->getNumOperands(); ++i)
        if (auto *Op = dyn_cast<MDNode>(LoopMD->getOperand(i)))
          if (Op->getNumOperands() >= 1)
            if (auto *S = dyn_cast<MDString>(Op->getOperand(0)))
              if (S->getString() == "llvm.loop.mustprogress")
                Has = true;
      if (Has)
        continue;
      // Lexical-block source locations: !{i32 786443, scope(SP), line, col,
      // file, 0}. Two per loop (start + end), like C++.
      unsigned LS = LoopBase, LE = LoopBase + 5;
      LoopBase += 10;
      MDNode *LocStart = MDNode::get(
          Ctx, {mdI32(786443), SP, mdI32(LS), mdI32(1), FileMD, mdI32(0)});
      MDNode *LocEnd = MDNode::get(
          Ctx, {mdI32(786443), SP, mdI32(LE), mdI32(1), FileMD, mdI32(0)});
      MDNode *MustProgress =
          MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.mustprogress")});
      // Rebuild loop md to match C++ order:
      //   !{self, locStart, locEnd, mustprogress, <existing name/tripcount/...>}
      SmallVector<Metadata *, 8> Ops;
      Ops.push_back(nullptr); // self-ref placeholder
      Ops.push_back(LocStart);
      Ops.push_back(LocEnd);
      Ops.push_back(MustProgress);
      for (unsigned i = 1; i < LoopMD->getNumOperands(); ++i)
        Ops.push_back(LoopMD->getOperand(i));
      MDNode *NewLoopMD = MDNode::getDistinct(Ctx, Ops);
      NewLoopMD->replaceOperandWith(0, NewLoopMD);
      Term->setMetadata(LLVMContext::MD_loop, NewLoopMD);
      Changed = true;
    }
  }
  return Changed;
}



// For `__vxx_top_pipeline(II)` markers: stamp `fpga.static.pipeline=II`
// function attribute on the enclosing function. OSS HLS clang emits this
// for `#pragma HLS PIPELINE` at function scope.
bool injectTopPipelineAttribute(Module &M) {
  bool Changed = false;
  Function *Marker = M.getFunction("__vxx_top_pipeline");
  if (!Marker)
    return false;
  SmallVector<CallInst *, 4> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    // Marker may be called with no args (default II = -1) or with i32 II.
    int64_t II = -1;
    if (CI->arg_size() >= 1) {
      if (auto *IIC = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
        II = (int64_t)IIC->getZExtValue();
    }
    Function *F = CI->getParent()->getParent();
    if (!F->hasFnAttribute("fpga.static.pipeline")) {
      // Format mirrors OSS HLS: "<II>.<style=0|1|2>.<rewind=0|1>".
      F->addFnAttr("fpga.static.pipeline",
                   std::to_string(II) + ".0.0");
      Changed = true;
    }
    Dead.push_back(CI);
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  return Changed;
}



// Restore a perfect loop nest around the pipelined stencil loop so the HLS
// backend outlines the outer + stencil loops *together* (e.g.
// `Filter2DKernel_Pipeline_VITIS_LOOP_47_2_VITIS_LOOP_49_3`).
//
// rustc -O runs LICM, which hoists loop-invariant window-offset math
// (e.g. `y - FV/2`) out of the inner stencil loop into the *outer* loop's
// header. That leaves non-control instructions between the outer-loop header
// and the inner-loop header, so the nest is no longer "perfect". The backend
// then outlines only the inner (stencil) loop; the line-buffer state, which must
// persist across the outer (row) iteration, ends up spanning the un-outlined
// outer loop → a malformed CDFG → `PerformanceInfo::getEstimatedSchedLatency`
// null-deref SIGSEGV right after "Finished scheduling".
//
// We keep that math inside the innermost loop so the nest stays
// perfect: sink every side-effect-free instruction that lives
// in the outer loop (but not the stencil loop) and is used *only* inside the
// stencil loop back down into the stencil-loop header. The values are
// loop-invariant w.r.t. the stencil loop, so recomputing them per inner
// iteration is functionally identical (the backend re-hoists internally).
bool sinkIntoStencilInnerLoop(Function &F, BasicBlock *StencilBB) {
  DominatorTree DT(F);
  LoopInfo LI;
  LI.analyze(DT);

  Loop *Lx = LI.getLoopFor(StencilBB);
  if (!Lx)
    return false;
  Loop *Ly = Lx->getParentLoop();
  if (!Ly)
    return false;

  auto inStencilLoop = [&](Instruction *I) {
    return Lx->contains(I->getParent());
  };
  auto inOuterNotInner = [&](Instruction *I) {
    return Ly->contains(I->getParent()) && !Lx->contains(I->getParent());
  };
  // Only sink pure, cheap, speculatable ops — never memory/side-effecting.
  auto isSinkable = [&](Instruction *I) {
    if (isa<PHINode>(I) || I->isTerminator())
      return false;
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      return false;
    return isa<CastInst>(I) || isa<BinaryOperator>(I) ||
           isa<GetElementPtrInst>(I) || isa<CmpInst>(I) ||
           isa<SelectInst>(I);
  };

  bool Changed = false;
  bool Again = true;
  while (Again) {
    Again = false;
    // Snapshot candidates so we don't iterate a mutating list.
    SmallVector<Instruction *, 16> Cands;
    for (BasicBlock *BB : Ly->blocks()) {
      if (Lx->contains(BB))
        continue;
      for (Instruction &I : *BB)
        if (isSinkable(&I))
          Cands.push_back(&I);
    }
    for (Instruction *I : Cands) {
      if (I->use_empty())
        continue;
      // All users must already live inside the stencil loop.
      bool AllInside = true;
      for (User *U : I->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI || !inStencilLoop(UI)) {
          AllInside = false;
          break;
        }
      }
      if (!AllInside)
        continue;
      // Sink to the top of the stencil-loop header (after PHIs). Inserting at
      // the current first-insertion-pt keeps operands before users when a
      // dependency chain is sunk across iterations.
      Instruction *InsertPt = &*Lx->getHeader()->getFirstInsertionPt();
      if (I == InsertPt)
        continue;
      I->moveBefore(InsertPt);
      Changed = true;
      Again = true;
    }
  }
  (void)inOuterNotInner;
  return Changed;
}



// Split a combined 2-D array GEP `getelementptr [W x T], [W x T]* %p, i, j`
// (one GEP, two indices) into the clang two-step chain
//   %row  = getelementptr [W x T], [W x T]* %p, i        ; pick the i-th row
//   %elem = getelementptr [W x T], [W x T]* %row, 0, j    ; pick the j-th elem
// The m_axi WRITE burst-inference keys off this per-dimension
// chain (same reason injectArrayStencil rewrites the stencil READ): rustc emits
// the combined form for `dst[y][x]`, so the backend leaves the write as a
// per-element `i8P1A` transaction (no burst → II=1 unmet →
// getEstimatedSchedLatency SIGSEGV) while the two-step form bursts (`p1i8`
// WriteReq(W*H)). Gate: GEPs whose base
// is a kernel-arg pointer-to-array (the m_axi ports), in top kernels only.
bool splitCombinedArrayGeps(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = IntegerType::get(Ctx, 64);
  Constant *Zero64 = ConstantInt::get(I64, 0);
  SmallPtrSet<Function *, 4> Tops;
  if (Function *TM = M.getFunction("__vxx_top_kernel"))
    for (User *U : TM->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *F = CI->getFunction()) Tops.insert(F);
  // EXCLUDE stencil-marked args: those go through injectArrayStencil, which does
  // its own spill + 2-step canonicalisation to build line buffers + a burst read
  // feed. Splitting them here first interferes with that transform and reverts
  // the src read to per-element i8P1A (x-loop outlined alone with ~227 scalar
  // args → null PerformanceInfo → SIGSEGV). Only the plain m_axi write args
  // (e.g. dst[y][x], no stencil pragma) need this split for write-burst.
  SmallPtrSet<Argument *, 4> StencilArgs;
  if (Function *ASten = M.getFunction("__vxx_array_stencil"))
    for (User *U : ASten->users())
      if (auto *SCI = dyn_cast<CallInst>(U))
        if (Argument *A = resolveMarkerArg(SCI->getArgOperand(0)))
          StencilArgs.insert(A);
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!Tops.count(&F) && !F.hasFnAttribute("fpga.top.func")) continue;
    SmallVector<GetElementPtrInst *, 8> Work;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *G = dyn_cast<GetElementPtrInst>(&I))
          if (G->getNumIndices() == 2 &&
              isa<ArrayType>(G->getSourceElementType()) &&
              isa<Argument>(G->getPointerOperand()->stripPointerCasts()) &&
              !StencilArgs.count(
                  cast<Argument>(G->getPointerOperand()->stripPointerCasts())))
            Work.push_back(G);
    for (GetElementPtrInst *G : Work) {
      IRBuilder<> B(G);
      Value *Row = B.CreateInBoundsGEP(G->getSourceElementType(),
                                       G->getPointerOperand(),
                                       {G->getOperand(1)}, "row2");
      Value *Elem = B.CreateInBoundsGEP(G->getSourceElementType(), Row,
                                        {Zero64, G->getOperand(2)}, "elem2");
      G->replaceAllUsesWith(Elem);
      G->eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: split combined 2-D array GEPs into clang 2-step\n";
  return Changed;
}



bool injectArrayStencil(Module &M) {
  // `#pragma HLS array_stencil variable=src` lowers to
  //   call void @llvm.sideeffect() [ "fpga_array_stencil"(ptr %src, i1 false) ]
  // placed at the top of the window-sliding (output-pixel) loop body, and the
  // HLS backend turns the sliding-window accesses into line buffers.
  //
  // Two structural conditions must hold for the line-buffer recogniser to fire:
  //  (1) the `fpga_array_stencil` op-bundle references the stencil array, AND
  //  (2) the array is an *addressable variable*: the array param is spilled
  //      into a `%src.addr` alloca and every windowed access reloads
  //      from it (`%p = load %src.addr; gep %p, ...`). The line-buffer pass
  //      replaces the storage behind that alloca. rustc -O passes the array as
  //      a pure SSA argument with no backing alloca, so the recogniser has no
  //      "variable" to replace and silently declines (LineBuffer=0) → the
  //      pipelined window loop is left as 225 random reads → LatencyEstimator
  //      SIGSEGV after scheduling.
  //
  // So we replicate that shape: emit the bundle at the marker site
  // (which rustc keeps in the output-pixel loop body), spill the stencil arg
  // to an alloca at function entry, and reroute every *data* use (not the
  // interface op-bundles) through a reload from that alloca. The reloaded value
  // always equals the stored arg, so it is functionally identical (mem2reg
  // would fold it).
  Function *Marker = M.getFunction("__vxx_array_stencil");
  if (!Marker)
    return false;

  LLVMContext &Ctx = M.getContext();
  Type *I1 = IntegerType::get(Ctx, 1);
  Function *SEFn = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);

  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  SmallPtrSet<Argument *, 4> Spilled;
  // (function, stencil-loop block) — used after marker cleanup to restore the
  // perfect loop nest so the backend outlines the outer+stencil loops together.
  SmallVector<std::pair<Function *, BasicBlock *>, 4> StencilSites;

  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->arg_size() < 1)
      continue;
    Dead.push_back(CI);
    StencilSites.push_back({CI->getFunction(), CI->getParent()});

    // Trace the marker arg back to the underlying function Argument through
    // any bitcasts that Rust's i8* marker lowering inserts.
    Value *V = CI->getArgOperand(0);
    while (true) {
      if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
      if (auto *BCC = dyn_cast<BitCastOperator>(V)) { V = BCC->getOperand(0); continue; }
      break;
    }
    auto *Arg = dyn_cast<Argument>(V);
    Value *StencilPtr = Arg ? (Value *)Arg : V;

    // (1) Emit the fpga_array_stencil op-bundle at the marker site.
    IRBuilder<> B(CI);
    Value *BundleArgs[2] = {StencilPtr, ConstantInt::get(I1, 0)};
    OperandBundleDef OBD("fpga_array_stencil",
                         SmallVector<Value *, 2>(std::begin(BundleArgs),
                                                 std::end(BundleArgs)));
    CallInst *SE = B.CreateCall(SEFn, {}, {OBD});
    SE->setOnlyAccessesInaccessibleMemory();
    SE->setDoesNotThrow();
    Changed = true;

    // (2) Spill the stencil arg to an alloca and reroute data uses.
    if (Arg && !Spilled.count(Arg)) {
      Spilled.insert(Arg);
      Function *F = Arg->getParent();
      BasicBlock &Entry = F->getEntryBlock();
      IRBuilder<> EB(&Entry, Entry.getFirstInsertionPt());
      AllocaInst *Addr =
          EB.CreateAlloca(Arg->getType(), nullptr, Arg->getName() + ".addr");
      StoreInst *SpillSt = EB.CreateStore(Arg, Addr);

      // Snapshot the data uses (skip interface op-bundles and our spill store).
      SmallVector<Use *, 16> DataUses;
      for (Use &Us : Arg->uses()) {
        User *UU = Us.getUser();
        if (auto *C = dyn_cast<CallInst>(UU))
          if (C->getIntrinsicID() == Intrinsic::sideeffect)
            continue;
        if (UU == SpillSt)
          continue;
        DataUses.push_back(&Us);
      }
      SmallVector<LoadInst *, 16> Reloads;
      for (Use *Us : DataUses) {
        auto *UI = dyn_cast<Instruction>(Us->getUser());
        if (!UI)
          continue;
        IRBuilder<> RB(UI);
        LoadInst *Reload =
            RB.CreateLoad(Arg->getType(), Addr, Arg->getName() + ".reload");
        Us->set(Reload);
        Reloads.push_back(Reload);
      }

      // Canonicalise the windowed 2D access into the two-step GEP
      // chain the line-buffer recogniser expects. rustc emits a single
      // combined GEP `gep [W x T], %p, row, col`; the two-step form is
      // `%r = gep [W x T], %p, row` (select the row) then
      // `gep [W x T], %r, 0, col` (index within the row). Semantically equal,
      // but the recogniser keys off the per-dimension chain.
      Type *I64 = IntegerType::get(Ctx, 64);
      Constant *Zero64 = ConstantInt::get(I64, 0);
      for (LoadInst *Reload : Reloads) {
        SmallVector<GetElementPtrInst *, 4> Geps;
        for (User *RU : Reload->users())
          if (auto *G = dyn_cast<GetElementPtrInst>(RU))
            if (G->getPointerOperand() == Reload && G->getNumIndices() == 2 &&
                isa<ArrayType>(G->getSourceElementType()))
              Geps.push_back(G);
        for (GetElementPtrInst *G : Geps) {
          IRBuilder<> GB(G);
          // The window offsets (x+col-FH/2, y+row-FV/2) are negative at image
          // boundaries. rustc proves them non-negative inside the bounds-check
          // `else` branch and emits `zext i32 off to i64` for the GEP index.
          // But the line-buffer recogniser does its own range analysis
          // on the raw index and sees zext(negative) ≈ 2^32 → it computes an
          // absurd buffer size (999*2^32) and bails:
          //   "HLS 214-442 ... buffering size 4290672328704 is larger than 65536".
          // A `sext` of the same signed offset gives a
          // small signed range the recogniser accepts. Since the index is
          // provably in [0,dim) here, sext≡zext numerically — rewrite zext→sext
          // on the index operands to unblock the transform.
          auto signExtIndex = [&](Value *Idx) -> Value * {
            if (auto *ZE = dyn_cast<ZExtInst>(Idx))
              return GB.CreateSExt(ZE->getOperand(0), ZE->getType(),
                                   "stencil.sidx");
            return Idx;
          };
          Value *Idx0 = signExtIndex(G->getOperand(1));
          Value *Idx1 = signExtIndex(G->getOperand(2));
          Value *Row = GB.CreateInBoundsGEP(G->getSourceElementType(), Reload,
                                            {Idx0}, "stencil.row");
          Value *Elem = GB.CreateInBoundsGEP(G->getSourceElementType(), Row,
                                             {Zero64, Idx1}, "stencil.elem");
          G->replaceAllUsesWith(Elem);
          G->eraseFromParent();
        }
      }
    }
  }

  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();

  // NOTE: sinkIntoStencilInnerLoop (restoring the perfect y/x nest that rustc
  // -O LICM broke) was tried and is INEFFECTIVE — the backend still outlines the
  // x-loop alone (threading ~600 line-buffer registers as scalar args) and
  // still SIGSEGVs in LatencyEstimator. Left unused pending a different angle.
  (void)StencilSites;

  return Changed;
}






} } // namespace hlsrs::vxx
