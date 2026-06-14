//===----------------------------------------------------------------------===//
//
// dataflow_kpn.cpp — DATAFLOW / KPN / hls::task passes: dataflow region tagging, helper-fn decay, mt_task, KPN stream/task wrapping, autorestart, ap_ctrl_{none,chain} attributes.
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
#include "llvm/ADT/STLExtras.h"  // is_contained / find (KPN-extract arg dedup)
#include "llvm/Transforms/Utils/BasicBlockUtils.h"  // SplitBlock (in-loop dataflow outline)
#include "llvm/Transforms/Utils/CodeExtractor.h"    // in-loop dataflow body outline
#include "vxx_llvm.h"  // consolidated common LLVM headers

#include <string>

using namespace llvm;
using namespace hlsrs::vxx;

namespace hlsrs { namespace vxx {

// Outline the body of a loop that starts with an in-loop `__vxx_dataflow()`
// marker into its own `dataflow_in_loop_*` function — the C++ pre-reflow
// form for `for (...) { #pragma HLS DATAFLOW ... }` (a.pp.bc: the frontend
// CodeExtractor leaves `for.cond -> codeRepl { SpecDataflowPipeline; call
// @dataflow_in_loop_* } -> for.cond`, with the dataflow buffers as allocas
// INSIDE the callee). Without this, hoisting the marker to function scope
// synthesizes a function-level dataflow (1 process) instead of the C++'s
// per-iteration N-process dataflow (using_vectors divergence).
//
// Steps: split the (rotated) loop header after PHIs + marker; split the
// latch before the induction increment; sink outside-defined pointer
// derivations (allocas / bitcasts / GEPs used only in the body) into the
// region; CodeExtractor the region; stamp the callee with the entry
// SpecDataflowPipeline + fpga.dataflow.func attr and drop the marker.
// Returns false if the loop shape is unsupported (caller falls back to the
// function-level form).
static bool outlineInLoopDataflow(Module &M, CallInst *MarkerCall) {
  Function *F = MarkerCall->getFunction();
  BasicBlock *MBB = MarkerCall->getParent();
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  Loop *L = LI.getLoopFor(MBB);
  if (!L || L->getHeader() != MBB) return false;
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch) return false;

  // Read the loop name from the backedge's !llvm.loop metadata (harvested
  // llvm.loop.name) before the splits move the terminator around.
  std::string LoopName = "VITIS_LOOP_1";
  if (MDNode *LID = L->getLoopID()) {
    for (unsigned i = 1; i < LID->getNumOperands(); ++i) {
      if (auto *Node = dyn_cast<MDNode>(LID->getOperand(i))) {
        if (Node->getNumOperands() == 2) {
          if (auto *S = dyn_cast<MDString>(Node->getOperand(0))) {
            if (S->getString() == "llvm.loop.name") {
              if (auto *NameS = dyn_cast<MDString>(Node->getOperand(1)))
                LoopName = NameS->getString().str();
            }
          }
        }
      }
    }
  }

  // Find the induction increment: the header PHI's incoming value from the
  // latch, defined in the latch block. The {inc, cmp, br} tail stays in the
  // top function as the loop control (C++ keeps it in for.cond).
  Instruction *Inc = nullptr;
  for (PHINode &PN : MBB->phis()) {
    Value *V = PN.getIncomingValueForBlock(Latch);
    if (auto *I = dyn_cast<Instruction>(V))
      if (I->getParent() == Latch) { Inc = I; break; }
  }
  if (!Inc) return false;
  // The latch terminator's compare closes the control tail.
  auto *Br = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!Br || !Br->isConditional()) return false;
  auto *CmpI = dyn_cast<Instruction>(Br->getCondition());
  if (!CmpI || CmpI->getParent() != Latch) return false;
  // rustc's `for` desugar computes the increment BEFORE the body
  // (Range::next), interleaving it with the region. Sink it down to the
  // compare so {inc, cmp, br} form a contiguous splittable tail — legal
  // only when nothing in the body reads the incremented value (its users
  // are the header PHI and possibly the compare).
  for (User *U : Inc->users()) {
    auto *UI = dyn_cast<Instruction>(U);
    if (!UI) return false;
    if (UI != CmpI && !isa<PHINode>(UI)) return false;
  }
  if (Inc->getNextNode() != CmpI) Inc->moveBefore(CmpI);

  // HLS rejects a dataflow region inside a bottom-tested (do-while) loop
  // (XFORM 203-711); the C++ pre-reflow form is the top-tested
  // `for.cond -> codeRepl -> for.cond`. After extraction the loop is
  // un-rotated, which is only sound for the exact rustc `for 0..n` shape —
  // verify it up front: induction phi starts at 0, steps by +1, and the
  // latch exits on `inc == bound` (or its `ne` inversion).
  PHINode *IndPhi = nullptr;
  for (PHINode &PN : MBB->phis()) {
    Value *V = PN.getIncomingValueForBlock(Latch);
    if (V == Inc) { IndPhi = &PN; continue; }
    if (!isa<Constant>(V)) return false;  // out-of-region latch value
  }
  if (!IndPhi) return false;
  auto *AddI = dyn_cast<BinaryOperator>(Inc);
  if (!AddI || AddI->getOpcode() != Instruction::Add) return false;
  Value *Step = AddI->getOperand(0) == IndPhi ? AddI->getOperand(1)
              : AddI->getOperand(1) == IndPhi ? AddI->getOperand(0)
                                              : nullptr;
  auto *StepC = dyn_cast_or_null<ConstantInt>(Step);
  if (!StepC || !StepC->isOne()) return false;
  auto *StartC = dyn_cast<ConstantInt>(
      IndPhi->getIncomingValueForBlock(L->getLoopPreheader()
                                           ? L->getLoopPreheader()
                                           : IndPhi->getIncomingBlock(0)));
  if (!StartC || !StartC->isZero()) return false;
  auto *ICmp = dyn_cast<ICmpInst>(CmpI);
  if (!ICmp) return false;
  Value *Bound = ICmp->getOperand(0) == Inc ? ICmp->getOperand(1)
               : ICmp->getOperand(1) == Inc ? ICmp->getOperand(0)
                                            : nullptr;
  if (!Bound) return false;
  if (auto *BI = dyn_cast<Instruction>(Bound))
    if (L->contains(BI->getParent())) return false;
  bool ExitOnTrue = (Br->getSuccessor(0) != MBB);
  if (!((ICmp->getPredicate() == ICmpInst::ICMP_EQ && ExitOnTrue) ||
        (ICmp->getPredicate() == ICmpInst::ICMP_NE && !ExitOnTrue)))
    return false;
  BasicBlock *ExitBB = Br->getSuccessor(0) == MBB ? Br->getSuccessor(1)
                                                  : Br->getSuccessor(0);
  MDNode *LoopMD = Br->getMetadata(LLVMContext::MD_loop);

  // Region entry: header split right after the marker (PHIs + marker stay).
  BasicBlock *RegionEntry =
      SplitBlock(MBB, MarkerCall->getNextNode(), &DT, &LI);
  if (!RegionEntry) return false;
  // Loop-control tail: latch split at the increment. (Single-block loops:
  // the latch IS the header — after the first split the latch is
  // RegionEntry, and Inc lives there.)
  BasicBlock *CtlTail = SplitBlock(Inc->getParent(), Inc, &DT, &LI);
  if (!CtlTail) return false;

  // Region = every loop block except the header remnant (PHIs+marker) and
  // the control tail.
  SmallVector<BasicBlock *, 8> Region;
  Region.push_back(RegionEntry);
  for (BasicBlock *BB : L->blocks())
    if (BB != MBB && BB != CtlTail && BB != RegionEntry)
      Region.push_back(BB);

  SmallPtrSet<BasicBlock *, 8> RegionSet(Region.begin(), Region.end());

  // Sink outside-defined derivations whose every user lives in the region:
  // dataflow buffer allocas (entry) and pointer casts/GEPs (entry or
  // preheader). Lifetime intrinsics on a sunk alloca are erased (they would
  // otherwise pin it outside). Iterate to a fixed point so chains
  // (alloca -> bitcast -> gep) sink in dependency order.
  bool SunkChanged = true;
  while (SunkChanged) {
    SunkChanged = false;
    for (BasicBlock &BB : *F) {
      if (RegionSet.count(&BB) || L->contains(&BB)) continue;
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if (!isa<AllocaInst>(I) && !isa<BitCastInst>(I) &&
            !isa<GetElementPtrInst>(I) && !isa<AddrSpaceCastInst>(I))
          continue;
        SmallVector<IntrinsicInst *, 4> Lifetimes;
        bool AllInRegion = true;
        for (User *U : I->users()) {
          auto *UI = dyn_cast<Instruction>(U);
          if (!UI) { AllInRegion = false; break; }
          if (auto *II = dyn_cast<IntrinsicInst>(UI)) {
            if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                II->getIntrinsicID() == Intrinsic::lifetime_end) {
              Lifetimes.push_back(II);
              continue;
            }
          }
          if (!RegionSet.count(UI->getParent())) { AllInRegion = false; break; }
        }
        if (!AllInRegion || I->use_empty()) continue;
        for (IntrinsicInst *II : Lifetimes) {
          // The lifetime's bitcast feed dies with it (DCE'd later if shared).
          II->eraseFromParent();
        }
        I->moveBefore(&*RegionEntry->getFirstInsertionPt());
        SunkChanged = true;
      }
    }
  }

  // Extract. CodeExtractor rewrites the region into `call @<F.name>.body(...)`
  // inside a new `codeRepl` block and moves the blocks into the new function.
  // AllowAlloca=true: the sunk dataflow buffer allocas live inside the region
  // (the default rejects any region containing an alloca).
  CodeExtractor CE(Region, &DT, /*AggregateArgs=*/false, /*BFI=*/nullptr,
                   /*BPI=*/nullptr, /*AC=*/nullptr, /*AllowVarArgs=*/false,
                   /*AllowAlloca=*/true, /*Suffix=*/"dfil");
  CodeExtractorAnalysisCache CEAC(*F);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  if (!Outlined) return false;

  Outlined->setName("dataflow_in_loop_" + LoopName + ".1");
  Outlined->setLinkage(GlobalValue::InternalLinkage);
  // CodeExtractor copies the parent's fn attributes — the top-function
  // markers must NOT survive on the callee (a second fn carrying
  // fpga.top.func/demangled.name="example" makes set_top ambiguous,
  // HLS 200-1986 → 200-1715).
  Outlined->removeFnAttr("fpga.top.func");
  Outlined->removeFnAttr("fpga.demangled.name");
  Outlined->addFnAttr("fpga.demangled.name", Outlined->getName());
  Outlined->addFnAttr("fpga.dataflow.func", "0");

  // SpecDataflowPipeline at the callee entry AND ahead of the call site —
  // the two positions the C++ pre-reflow form carries it in.
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SDFFn =
      M.getOrInsertFunction("_ssdm_op_SpecDataflowPipeline", SpecTy);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Value *Args[] = {ConstantInt::get(I32, -1), ConstantInt::get(I32, 0),
                   (Value *)EmptyStr};
  IRBuilder<> BC(&*Outlined->getEntryBlock().getFirstInsertionPt());
  BC.CreateCall(SDFFn, Args)->setDoesNotThrow();
  CallInst *Site = nullptr;
  for (User *U : Outlined->users())
    if (auto *CI = dyn_cast<CallInst>(U)) { Site = CI; break; }
  if (!Site) return true;
  IRBuilder<> BS(Site);
  BS.CreateCall(SDFFn, Args)->setDoesNotThrow();

  // Un-rotate: rebuild the C++ `for.cond -> codeRepl -> for.cond` shape.
  // Layout after extraction: MBB {phis, marker, br CodeRepl} -> CodeRepl
  // {SDF, call, br CtlTail} -> CtlTail {inc, cmp, br MBB/ExitBB}.
  // Target: MBB {phi, inc, newcmp, br CodeRepl/ExitBB} -> CodeRepl {SDF,
  // call, br MBB (!llvm.loop)}; CtlTail deleted.
  BasicBlock *CodeRepl = Site->getParent();
  if (CodeRepl->getSingleSuccessor() == CtlTail) {
    // Move the increment up into the header; the compare is rebuilt as the
    // top-of-loop test on the un-incremented phi.
    Instruction *HdrTerm = MBB->getTerminator();
    Inc->moveBefore(HdrTerm);
    IRBuilder<> BH(HdrTerm);
    Value *NewCmp = BH.CreateICmpNE(IndPhi, Bound, "exitcond");
    BranchInst *NewBr = BranchInst::Create(CodeRepl, ExitBB, NewCmp);
    ReplaceInstWithInst(HdrTerm, NewBr);
    // Loop metadata rides the backedge: codeRepl -> header.
    BranchInst *Back = BranchInst::Create(MBB);
    if (LoopMD) Back->setMetadata(LLVMContext::MD_loop, LoopMD);
    ReplaceInstWithInst(CodeRepl->getTerminator(), Back);
    // Header phis now flow around the new backedge.
    for (PHINode &PN : MBB->phis())
      PN.replaceIncomingBlockWith(CtlTail, CodeRepl);
    ExitBB->replacePhiUsesWith(CtlTail, MBB);
    CmpI->eraseFromParent();
    CtlTail->eraseFromParent();
  }
  vxxDbg() << "vxx: outlined in-loop dataflow body -> "
           << Outlined->getName() << " (" << Region.size() << " block(s))\n";
  return true;
}

// For `__vxx_dataflow()` markers AT FUNCTION ENTRY (entry block, no
// preds — i.e. not inside a loop), emit SpecDataflowPipeline + add the
// `fpga.dataflow.func=0` attribute. For markers inside a loop body,
// outline the loop body into a `dataflow_in_loop_*` function (the C++
// pre-reflow form); if the loop shape is unsupported, fall back to
// dropping the marker (emitting SpecDataflowPipeline at a non-entry
// position makes HLS reject the entire TOP function).
//
// Earlier attempts to emit SpecDataflowPipeline + SpecTopModule +
// SpecInterface + SpecBitsMap + InlineSelf + fpga.dataflow.func attr +
// fpga.dataflow.func metadata entry — caused HLS 200-1986 to fire. The HLS
// backend generates these ops itself; pre-emitting them confuses HLS.
bool injectDataflow(Module &M) {
  // Emit BOTH `fpga.dataflow.func` attr AND
  // `_ssdm_op_SpecDataflowPipeline` call. Goal:
  // give the HLS backend the canonical SpecDataflowPipeline call so its
  // default-interface auto-add pass sees a "complete" interface spec
  // and skips adding ap_auto SpecInterface (which conflicts with our
  // explicit SpecInterface(ap_fifo) emit).
  Function *Marker = M.getFunction("__vxx_dataflow");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SDFFn = M.getOrInsertFunction("_ssdm_op_SpecDataflowPipeline", SpecTy);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *NegOne = ConstantInt::get(I32, -1);
  Constant *Zero = ConstantInt::get(I32, 0);
  AttrBuilder NwAttrs;
  NwAttrs.addAttribute(Attribute::NoUnwind);
  AttributeList NwAttrList = AttributeList::get(Ctx, AttributeList::FunctionIndex, NwAttrs);

  SmallPtrSet<Function *, 4> Tops;
  SmallVector<CallInst *, 8> Dead;
  bool Changed = false;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Dead.push_back(CI);
    Function *F = CI->getFunction();
    if (!F) continue;
    // In-loop marker: outline the loop body into a dataflow_in_loop_*
    // function (C++ form). Fall back to the function-level form when the
    // loop shape is unsupported.
    if (CI->getParent() != &F->getEntryBlock() &&
        outlineInLoopDataflow(M, CI)) {
      Changed = true;
      continue;
    }
    Tops.insert(F);
  }
  for (Function *F : Tops) {
    if (!F->hasFnAttribute("fpga.dataflow.func")) {
      F->addFnAttr("fpga.dataflow.func", "0");
      Changed = true;
    }
    // Also emit the SpecDataflowPipeline call directly (C++-style).
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    Value *Args[] = {NegOne, Zero, (Value*)EmptyStr};
    CallInst *SDFCall = B.CreateCall(SDFFn, Args);
    SDFCall->setAttributes(NwAttrList);
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}



// For `__vxx_top_dataflow()` markers: stamp `fpga.dataflow.func=0`
// function attribute. OSS HLS clang emits this for `#pragma HLS DATAFLOW`.
bool injectDataflowAttribute(Module &M) {
  bool Changed = false;
  Function *Marker = M.getFunction("__vxx_top_dataflow");
  if (!Marker)
    return false;
  SmallVector<CallInst *, 4> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    Function *F = CI->getParent()->getParent();
    if (!F->hasFnAttribute("fpga.dataflow.func")) {
      F->addFnAttr("fpga.dataflow.func", "0");
      Changed = true;
    }
    Dead.push_back(CI);
  }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  return Changed;
}






// For helper functions called from a DATAFLOW-marked top kernel:
//   - Strip internal+fastcc → external+CCallConv
//   - Stamp `fpga.decayed.dim.hint="N"` on `[N x T]*` args (no type change)
//   - Set dso_local + clear unnamed_addr
// IMPORTANT: do NOT decay the actual arg type — DATAFLOW helpers retain
// `[N x T]*` shape; the HLS backend does the decay itself. Decaying here
// breaks the backend's expectation.
bool decayDataflowHelperFns(Module &M) {
  // Collect DATAFLOW-marked functions
  SmallPtrSet<Function *, 4> DfFns;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (F.hasFnAttribute("fpga.dataflow.func")) DfFns.insert(&F);
  }
  if (DfFns.empty()) return false;
  // Collect non-trivial helpers callees + the DATAFLOW top fns themselves
  // (C++ Clang auto-decays `int[N]` → `int*` at AST lowering for ALL fns;
  // Rust's `&[T; N]` stays `[N x T]*` — we must redo the decay here, both
  // for helpers AND for the top kernel itself when it's DATAFLOW-marked).
  SmallPtrSet<Function *, 8> Helpers;
  for (Function *F : DfFns) {
    Helpers.insert(F);  // decay top fn itself
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee || Callee->isDeclaration()) continue;
          if (Callee->getName().startswith("llvm.")) continue;
          if (Callee->getName().startswith("_ssdm_op_")) continue;
          Helpers.insert(Callee);
        }
      }
    }
  }
  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  for (Function *H : Helpers) {
    if (H->isDeclaration()) continue;
    // First strip linkage/CC (cheap; same as before).
    if (H->getLinkage() == GlobalValue::InternalLinkage) {
      H->setLinkage(GlobalValue::ExternalLinkage);
      Changed = true;
    }
    if (H->getCallingConv() != CallingConv::C) {
      H->setCallingConv(CallingConv::C);
      for (User *U : H->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          CI->setCallingConv(CallingConv::C);
      Changed = true;
    }
    // Identify [N x T]* args to decay
    FunctionType *OldFT = H->getFunctionType();
    unsigned NP = OldFT->getNumParams();
    SmallVector<bool, 8> DecayMask(NP, false);
    SmallVector<uint64_t, 8> DimN(NP, 0);
    bool AnyDecay = false;
    for (unsigned i = 0; i < NP; ++i) {
      auto *PT = dyn_cast<PointerType>(OldFT->getParamType(i));
      if (!PT) continue;
      auto *AT = dyn_cast<ArrayType>(PT->getElementType());
      if (!AT) continue;
      DecayMask[i] = true;
      DimN[i] = AT->getNumElements();
      AnyDecay = true;
    }
    if (!AnyDecay) continue;
    // Skip the full sig rewrite — only strip arg attrs that would be invalid
    // after decay. Keep the [N x T]* shape; the HLS backend decays it later.
    for (unsigned i = 0; i < NP; ++i) {
      if (!DecayMask[i]) continue;
      Argument *Arg = H->arg_begin() + i;
      // Normalize arg attrs: strip nocapture/align/dereferenceable
      // (rustc artifacts that confuse HLS TOP analysis), add noalias.
      Arg->removeAttr(Attribute::NoCapture);
      Arg->removeAttr(Attribute::Alignment);
      Arg->removeAttr(Attribute::Dereferenceable);
      Arg->removeAttr(Attribute::DereferenceableOrNull);
      Arg->removeAttr(Attribute::ReadOnly);
      if (!Arg->hasAttribute(Attribute::NoAlias))
        Arg->addAttr(Attribute::NoAlias);
      // dim.hint is helpful but the attribute name string itself may confuse
      // the HLS backend, which adds it itself normally. Skip stamping it.
    }
    // Set dso_local + clear unnamed_addr on the helper itself
    H->setDSOLocal(true);
    H->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    Changed = true;
    continue;  // Skip full sig rewrite below.
    // Build new FT (unreachable — kept for reference)
    SmallVector<Type *, 8> NewParamTys;
    for (unsigned i = 0; i < NP; ++i) {
      Type *Old = OldFT->getParamType(i);
      if (!DecayMask[i]) { NewParamTys.push_back(Old); continue; }
      auto *PT = cast<PointerType>(Old);
      auto *AT = cast<ArrayType>(PT->getElementType());
      NewParamTys.push_back(
          PointerType::get(AT->getElementType(), PT->getAddressSpace()));
    }
    FunctionType *NewFT = FunctionType::get(
        H->getReturnType(), NewParamTys, H->isVarArg());
    Function *NewF = Function::Create(NewFT, H->getLinkage(),
                                       H->getName() + ".dh_tmp",
                                       H->getParent());
    NewF->copyAttributesFrom(H);
    NewF->setCallingConv(CallingConv::C);
    // Set dso_local + clear unnamed_addr. HLS 200-1986
    // fires on Rust's default `unnamed_addr` flag for TOP/helper fns.
    NewF->setDSOLocal(true);
    NewF->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    // Move basic blocks from old to new
    NewF->getBasicBlockList().splice(NewF->end(), H->getBasicBlockList());
    // Carry function metadata
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    H->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);
    // Replace arg uses: for decayed args, bitcast back to [N x T]* at
    // entry so body GEPs keep working; for others, just RAUW.
    BasicBlock &Entry = NewF->getEntryBlock();
    IRBuilder<> B(&Entry, Entry.getFirstInsertionPt());
    auto OldA = H->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < NP; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (!DecayMask[i]) {
        // Copy old arg attrs
        AttributeSet AS = H->getAttributes().getParamAttributes(i);
        for (Attribute A : AS) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
      } else {
        // Use GEP `[N x T], [N x T]* (synthesized via bitcast), i64 0, i64 0`
        // to convert T* back to [N x T]* for body uses. C++ Clang emits
        // GEP-based array-decay, not bitcast — HLS strictly checks for
        // GEP shape during TOP-directive analysis.
        Type *OldTy = OldA->getType();  // [N x T]*
        Value *BCV = B.CreateBitCast(&*NewA, OldTy,
                                      NewA->getName() + ".as_arr");
        OldA->replaceAllUsesWith(BCV);
        // Stamp dim.hint on the decayed arg
        NewA->addAttr(Attribute::get(Ctx, "fpga.decayed.dim.hint",
                                      std::to_string(DimN[i])));
        // Drop stale alignment/dereferenceable inherited via copyAttrs
        NewA->removeAttr(Attribute::Alignment);
        NewA->removeAttr(Attribute::Dereferenceable);
        NewA->removeAttr(Attribute::DereferenceableOrNull);
        NewA->removeAttr(Attribute::NoCapture);
        NewA->removeAttr(Attribute::ReadOnly);
      }
    }
    // Rewrite call sites
    SmallVector<CallInst *, 8> CallSites;
    for (User *U : H->users())
      if (auto *CI = dyn_cast<CallInst>(U)) CallSites.push_back(CI);
    for (CallInst *CI : CallSites) {
      IRBuilder<> CB(CI);
      SmallVector<Value *, 8> NewArgs;
      for (unsigned i = 0; i < CI->arg_size(); ++i) {
        Value *AV = CI->getArgOperand(i);
        if (i < NP && DecayMask[i]) {
          // C++ Clang emits `getelementptr [N x T], [N x T]* X, i64 0, i64 0`
          // for "array decay" — HLS expects this GEP shape, not bitcast.
          // Build the GEP if AV is `[N x T]*`; otherwise (already i32* via
          // upstream decay) pass through.
          Type *NewT = NewParamTys[i];
          if (AV->getType() != NewT) {
            auto *AVPT = dyn_cast<PointerType>(AV->getType());
            if (AVPT && isa<ArrayType>(AVPT->getElementType())) {
              Value *Idx[] = {
                ConstantInt::get(Type::getInt64Ty(Ctx), 0),
                ConstantInt::get(Type::getInt64Ty(Ctx), 0),
              };
              AV = CB.CreateInBoundsGEP(AVPT->getElementType(), AV, Idx);
            } else {
              AV = CB.CreateBitCast(AV, NewT);
            }
          }
        }
        NewArgs.push_back(AV);
      }
      CallInst *NewCI = CB.CreateCall(NewF, NewArgs);
      NewCI->setCallingConv(CallingConv::C);
      // Strip call-site arg attrs on decayed args (alignment etc are
      // for old type; safer to start fresh).
      AttributeList AL = CI->getAttributes();
      for (unsigned i = 0; i < NP; ++i) {
        if (DecayMask[i]) AL = AL.removeParamAttributes(Ctx, i);
      }
      NewCI->setAttributes(AL);
      if (!CI->use_empty()) CI->replaceAllUsesWith(NewCI);
      CI->eraseFromParent();
    }
    // Finalise: erase old, rename new
    hlsrs::vxx::replaceFunctionKeepingName(H, NewF);
    Changed = true;
  }
  if (Changed)
    vxxDbg() << "vxx: decayed DATAFLOW helper fn(s) (full sig)\n";
  return Changed;
}



// Free-running `hls::task` (multi-thread task) region.
//
// `barista_hls::mt_task()` emits `__vxx_mt_task()` at the top of a kernel
// whose body re-fires a helper in a firing loop (the SW analog of a
// free-running `hls_thread_local hls::task t(...)`). For such a kernel we
// emit the canonical hls::task shape expected by the HLS backend
// (used by `using_directio_none_in_tasks`):
//
//   define void @top(...) #0 {
//     call void (...) @_ssdm_op_SpecDataflowPipeline(i32 -1, i32 1, ...)
//     ...
//   }
//   !has_MT_tasks = !{}
//
// plus the `fpga.dataflow.func="1"` function attribute (KPN/MT-task
// variant). The HLS backend then schedules the helper as a dataflow process
// rather than unrolling the firing loop over the directio reads (which SIGSEGVs
// in scheduling — see hypothesis-3 `Abnormal program termination(11)`),
// retains ap_clk, and the cosim TB generator feeds the directio ports via KpnDirectIO.
//
// Gated strictly on the `__vxx_mt_task` marker, so this only fires for
// kernels that explicitly opt in (currently just
// using_directio_none_in_tasks). The regular `dataflow_kpn()` /
// `__vxx_dataflow_kpn` path stays drop-only (unchanged), so the
// using_directio_hs_in_tasks sibling is unaffected.
bool injectMtTask(Module &M) {
  Function *Marker = M.getFunction("__vxx_mt_task");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SDFFn =
      M.getOrInsertFunction("_ssdm_op_SpecDataflowPipeline", SpecTy);
  if (auto *F = dyn_cast<Function>(SDFFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);
  FunctionCallee SpecIfFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *NegOne = ConstantInt::get(I32, (uint64_t)-1, /*isSigned=*/true);
  Constant *One = ConstantInt::get(I32, 1);
  AttrBuilder NwAttrs;
  NwAttrs.addAttribute(Attribute::NoUnwind);
  AttributeList NwAttrList =
      AttributeList::get(Ctx, AttributeList::FunctionIndex, NwAttrs);

  SmallPtrSet<Function *, 4> Tops;
  SmallVector<CallInst *, 4> Dead;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Dead.push_back(CI);
    if (Function *F = CI->getFunction()) Tops.insert(F);
  }
  bool Changed = false;
  for (Function *F : Tops) {
    // Find the task calls: each `task!` expands to `__vxx_mt_task(); <task
    // call>;` — pair every marker with the next call to a defined
    // (non-marker, non-ssdm) function. Linear scan keeps program order (the
    // C++ KPN calls the tasks in declaration order).
    SmallVector<CallInst *, 4> TaskCalls;
    bool HasProcessCalls = false;
    for (BasicBlock &BB : *F) {
      bool Pending = false;
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        Function *Cal = CI->getCalledFunction();
        if (Cal == Marker) {
          Pending = true;
          continue;
        }
        if (!Cal) continue;
        StringRef N = Cal->getName();
        if (Cal->isDeclaration() || N.startswith("__vxx_") ||
            N.startswith("_ssdm") || N.startswith("llvm."))
          continue;
        if (Pending) {
          TaskCalls.push_back(CI);
          Pending = false;
        } else {
          // A defined-fn call NOT spawned via task! — a regular dataflow
          // process alongside the tasks (a MIXED region).
          HasProcessCalls = true;
        }
      }
    }
    if (!HasProcessCalls) {
      // Flat task kernel (every process is a task — simple_data_driven,
      // using_maxi_in_tasks): the kernel itself IS the task region. The C++
      // pre-reflow form is a single SpecDataflowPipeline(-1, 1, "") on the
      // kernel (a DATAFLOW pragma on an all-task kernel folds into kind=1).
      F->addFnAttr("fpga.dataflow.func", "1");
      // _ssdm_op_SpecDataflowPipeline(i32 -1, i32 1, "") at function entry —
      // kind=1 (1 marks an hls::task region; the regular intra-fn DATAFLOW
      // pragma uses 0).
      IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
      Value *Args[] = {NegOne, One, (Value *)EmptyStr};
      CallInst *SDFCall = B.CreateCall(SDFFn, Args);
      SDFCall->setAttributes(NwAttrList);
      // Vitis-Clang auto-emits `SpecInterface(0, "ap_ctrl_none", ...)` on a
      // task-only top even without a pragma (measured on simple_data_driven's
      // a.pp.bc) — mirror it, unless the source already declared it (an
      // explicit `interface().mode(ap_ctrl_none).port(Return)` leaves either
      // the marker, if injectTopApCtrlNone runs later, or the SpecInterface).
      // Top-module only: the cpp_proxy build's rust_<top> adapter also holds
      // the task calls but runs in a sequential context (cosim_top), where
      // an ap_ctrl_none Return spec is illegal (HLS 200-649).
      bool HasCtrlNone = !F->hasFnAttribute("fpga.top.func");
      for (BasicBlock &BB : *F) {
        for (Instruction &I : BB) {
          auto *CI2 = dyn_cast<CallInst>(&I);
          if (!CI2 || !CI2->getCalledFunction()) continue;
          StringRef CN = CI2->getCalledFunction()->getName();
          if (CN == "__vxx_top_ap_ctrl_none") { HasCtrlNone = true; break; }
          if (CN == "_ssdm_op_SpecInterface" && CI2->arg_size() >= 2) {
            if (auto *GV = dyn_cast<GlobalVariable>(
                    CI2->getArgOperand(1)->stripPointerCasts())) {
              if (GV->hasInitializer()) {
                if (auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer())) {
                  if (CDA->isCString() &&
                      CDA->getAsCString() == "ap_ctrl_none") {
                    HasCtrlNone = true;
                    break;
                  }
                }
              }
            }
          }
        }
        if (HasCtrlNone) break;
      }
      if (!HasCtrlNone) {
        GlobalVariable *NoneStr = getOrCreateCStrGlobal(M, "ap_ctrl_none");
        Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
        Constant *NonePtr = ConstantExpr::getInBoundsGetElementPtr(
            NoneStr->getValueType(), NoneStr, Idxs);
        Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
            EmptyStr->getValueType(), EmptyStr, Idxs);
        CallInst *IfCall = B.CreateCall(
            SpecIfFn,
            {ConstantInt::get(I32, 0),
             NonePtr,
             ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), EmptyPtr,
             ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
             EmptyPtr, EmptyPtr, EmptyPtr,
             ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
             ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
             EmptyPtr, EmptyPtr,
             ConstantInt::getSigned(I32, -1),
             ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
             ConstantInt::get(I32, 0)});
        IfCall->setAttributes(NwAttrList);
      }
      Changed = true;
      continue;
    }
    // ── MIXED region: hls::tasks alongside normal dataflow processes
    // (unique_task_regions' `mid<c1>`: 2 tasks + 4 copies). The C++
    // pre-reflow form (a.pp.bc) extracts the task calls into a synthetic
    // `KPN` function —
    //   define internal void @KPN(chans...) {            ; noinline
    //     _ssdm_op_SpecDataflowPipeline(-1, 1, "")       ; task region
    //     _ssdm_op_SpecInterface(0, "ap_ctrl_none", ...) ; free-running
    //     call @task1(...) ... call @taskN(...)
    //   }
    // called from the region alongside the process calls; the region keeps
    // its own regular kind=0 DATAFLOW. Stamping the mixed region itself
    // with kind=1 instead (the flat path above) makes reflow schedule the
    // tasks as start-synced processes -> RTL deadlock (a copy blocks on the
    // depth-2 channel it must fill before a "process" func would start).
    if (TaskCalls.empty()) continue;
    // Channel args, deduped in first-use order (the KPN signature).
    SmallVector<Value *, 8> Chans;
    for (CallInst *TC : TaskCalls)
      for (Value *A : TC->args())
        if (!llvm::is_contained(Chans, A)) Chans.push_back(A);
    SmallVector<Type *, 8> ArgTys;
    for (Value *V : Chans) ArgTys.push_back(V->getType());
    FunctionType *KTy =
        FunctionType::get(Type::getVoidTy(Ctx), ArgTys, /*Var=*/false);
    Function *K = Function::Create(KTy, GlobalValue::InternalLinkage, "KPN", &M);
    K->addFnAttr(Attribute::NoInline);
    K->addFnAttr(Attribute::NoUnwind);
    K->addFnAttr("fpga.dataflow.func", "1");
    BasicBlock *KBB = BasicBlock::Create(Ctx, "newFuncRoot", K);
    IRBuilder<> KB(KBB);
    {
      Value *Args[] = {NegOne, One, (Value *)EmptyStr};
      CallInst *SDFCall = KB.CreateCall(SDFFn, Args);
      SDFCall->setAttributes(NwAttrList);
    }
    // _ssdm_op_SpecInterface(0, "ap_ctrl_none", 0, 0, "", 0, 0, "", "", "",
    // 0, 0, 0, 0, "", "", -1, 0, 0, 0) — the C++ KPN wrapper's shape.
    {
      GlobalVariable *NoneStr = getOrCreateCStrGlobal(M, "ap_ctrl_none");
      Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
      Constant *NonePtr = ConstantExpr::getInBoundsGetElementPtr(
          NoneStr->getValueType(), NoneStr, Idxs);
      Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
          EmptyStr->getValueType(), EmptyStr, Idxs);
      KB.CreateCall(SpecIfFn,
                    {ConstantInt::get(I32, 0),
                     NonePtr,
                     ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), EmptyPtr,
                     ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                     EmptyPtr, EmptyPtr, EmptyPtr,
                     ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                     ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                     EmptyPtr, EmptyPtr,
                     ConstantInt::getSigned(I32, -1),
                     ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                     ConstantInt::get(I32, 0)});
    }
    for (CallInst *TC : TaskCalls) {
      SmallVector<Value *, 4> MappedArgs;
      for (Value *A : TC->args()) {
        size_t Idx = llvm::find(Chans, A) - Chans.begin();
        MappedArgs.push_back(K->getArg(Idx));
      }
      CallInst *NC = KB.CreateCall(TC->getCalledFunction(), MappedArgs);
      NC->setCallingConv(TC->getCallingConv());
    }
    KB.CreateRetVoid();
    // Replace the task calls in F with one KPN call at the first task's
    // position (the C++ region calls KPN before its copy processes).
    IRBuilder<> BF(TaskCalls.front());
    BF.CreateCall(K, Chans);
    for (CallInst *TC : TaskCalls) TC->eraseFromParent();
    Changed = true;
  }
  if (!Tops.empty() && !M.getNamedMetadata("has_MT_tasks")) {
    M.getOrInsertNamedMetadata("has_MT_tasks");
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  if (Marker->use_empty()) Marker->eraseFromParent();
  return Changed;
}



// KPN local stream / task wrapping — the `__vxx_dataflow_kpn` markers
// tagged which streams are KPN-local; the KPN extraction reads the
// region structure directly now, so the markers are just dropped.
bool injectKpnLocalStream(Module &M) {
  bool A = hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_dataflow_kpn");
  return A;
}



bool wrapKpnStreamArgs(Module &M) {
  // Stub — was used to mark KPN process stream arguments; relies on
  // upstream proc-macro emitting __vxx_kpn_stream_arg markers, which the
  // current barista-hls does not. Safe to no-op.
  return false;
}



bool wrapKpnTaskCalls(Module &M) {
  // Stub — KPN task call rewrapping is done in the Rust source via
  // `barista_hls::dataflow_kpn()` + helper-fn extraction; no LLVM-level
  // wrapping needed in this build.
  return false;
}



// For each `__vxx_top_autorestart()` marker, emit at function entry
//   _ssdm_op_SpecInterface(i32 0, [10 x i8]* "s_axilite", i32 0, i32 0,
//                          [1 x i8]* "", i32 0, i32 0,
//                          [8 x i8]* "control", [1 x i8]* "",
//                          [1 x i8]* "", i32 0, i32 0, i32 0, i32 0,
//                          [1 x i8]* "", [1 x i8]* "",
//                          i32 1, i32 0, i32 0, i32 0)
// Mirrors C++ `#pragma HLS INTERFACE s_axilite port=return autorestart`.
// Required by the `hls::autorestart` TB wrapper which checks this exact
// pragma shape on the kernel function passed to it.
bool injectAutorestart(Module &M) {
  Function *Marker = M.getFunction("__vxx_top_autorestart");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy =
      FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
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
  for (Function *Top : Tops) {
    IRBuilder<> B(&*Top->getEntryBlock().getFirstInsertionPt());
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
                  ConstantInt::get(I32, 1),     // autorestart bit
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0)});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  return Changed;
}



// For each `__vxx_top_ap_ctrl_none()` marker, emit
//   _ssdm_op_SpecInterface(i32 0, [13 x i8]* "ap_ctrl_none", i32 0, ...)
// at function entry. This is the lowering of
// `#pragma HLS INTERFACE mode=ap_ctrl_none port=return`; the HLS backend
// derives `fpga.handshake.mode="ap_ctrl_none"` post-LTO from the call.
//
// We deliberately DON'T set the attribute directly: the analyzer's KPN
// dataflow check (for `!has_MT_tasks` modules) requires the call-form
// signal to recognise the function as a free-running task region. Without
// it, the produce/consume-once check fires on internal streams even
// though the dataflow type marker says KPN.
bool injectApCtrlNoneAttribute(Module &M) {
  Function *Marker = M.getFunction("__vxx_top_ap_ctrl_none");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  GlobalVariable *NoneStr = getOrCreateCStrGlobal(M, "ap_ctrl_none");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
  Constant *NonePtr = ConstantExpr::getInBoundsGetElementPtr(
      NoneStr->getValueType(), NoneStr, Idxs);
  Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
      EmptyStr->getValueType(), EmptyStr, Idxs);

  SmallPtrSet<Function *, 4> Tops;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Tops.insert(CI->getParent()->getParent());
    Dead.push_back(CI);
  }
  for (Function *Top : Tops) {
    // Strip any directly-set attribute (defense in depth — see above).
    Top->removeFnAttr("fpga.handshake.mode");
    IRBuilder<> B(&*Top->getEntryBlock().getFirstInsertionPt());
    // 20-arg shape matching C++ `_ssdm_op_SpecInterface(0, "ap_ctrl_none",
    // 0, 0, "", 0, 0, "", "", "", 0, 0, 0, 0, "", "", -1, 0, 0, 0)`.
    B.CreateCall(SpecFn,
                 {ConstantInt::get(I32, 0),
                  NonePtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr,
                  ConstantInt::getSigned(I32, -1),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0)});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  return Changed;
}



// `__vxx_top_ap_ctrl_chain()` → emit
//   _ssdm_op_SpecInterface(0, "ap_ctrl_chain", 0, 0, "", 0, 0, ...)
// at function entry. Mirrors `#pragma HLS INTERFACE ap_ctrl_chain port=return`.
// Identical 20-arg shape to ap_ctrl_none, just different mode string.
bool injectApCtrlChainAttribute(Module &M) {
  Function *Marker = M.getFunction("__vxx_top_ap_ctrl_chain");
  if (!Marker) return false;
  bool Changed = false;
  SmallVector<CallInst *, 4> Dead;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), /*Var=*/true);
  FunctionCallee SpecFn =
      M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
  GlobalVariable *ChainStr = getOrCreateCStrGlobal(M, "ap_ctrl_chain");
  GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
  Constant *Idxs[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, 0)};
  Constant *ChainPtr = ConstantExpr::getInBoundsGetElementPtr(
      ChainStr->getValueType(), ChainStr, Idxs);
  Constant *EmptyPtr = ConstantExpr::getInBoundsGetElementPtr(
      EmptyStr->getValueType(), EmptyStr, Idxs);

  SmallPtrSet<Function *, 4> Tops;
  for (User *U : Marker->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI) continue;
    Tops.insert(CI->getParent()->getParent());
    Dead.push_back(CI);
  }
  for (Function *Top : Tops) {
    Top->removeFnAttr("fpga.handshake.mode");
    IRBuilder<> B(&*Top->getEntryBlock().getFirstInsertionPt());
    B.CreateCall(SpecFn,
                 {ConstantInt::get(I32, 0),
                  ChainPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0), EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr, EmptyPtr,
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  EmptyPtr, EmptyPtr,
                  ConstantInt::getSigned(I32, -1),
                  ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                  ConstantInt::get(I32, 0)});
    Changed = true;
  }
  for (CallInst *CI : Dead) CI->eraseFromParent();
  return Changed;
}



} } // namespace hlsrs::vxx
