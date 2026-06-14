// vxx_prep.cpp — the IR-normalization stages of the Vitis-HLS prep.
// vxxShapeIR turns rustc-emitted Rust IR into synthesizable fpga64 LLVM IR
// (loop canonicalization, BitInt narrowing, static globals, struct layout),
// with no Xilinx intrinsics yet; vxxEarlyShape does the pre-opt rustc -O0
// cleanup. The Vitis pragma/interface lowering follows in VXXPrep.cpp
// (vxxLowerVitis). All compiled statically into rustc.

#include "VXXShared.h"
#include "VXXPrep.h"
#include "vxx_passes.h"


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
#include "llvm/Analysis/ConstantFolding.h"
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
#include "llvm/Support/CommandLine.h"
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

namespace hlsrs { namespace vxx {

static std::string demangleRustGlobal(StringRef Name) {
  if (!Name.startswith("_ZN") || !Name.endswith("E"))
    return "";
  StringRef Body = Name.drop_front(3).drop_back(1);
  SmallVector<std::string, 6> Segments;
  while (!Body.empty()) {
    size_t I = 0;
    while (I < Body.size() && std::isdigit(static_cast<unsigned char>(Body[I])))
      ++I;
    if (I == 0)
      break;
    unsigned Len = 0;
    if (Body.substr(0, I).getAsInteger(10, Len) || Len == 0)
      break;
    Body = Body.drop_front(I);
    if (Body.size() < Len)
      break;
    StringRef Seg = Body.substr(0, Len);
    Body = Body.drop_front(Len);
    // Drop the trailing 17-char hash segment ("h" + 16 hex).
    if (Len == 17 && Seg[0] == 'h')
      continue;
    Segments.push_back(Seg.str());
  }
  if (Segments.size() < 2)
    return "";
  // First segment is the crate, last is the variable name. Skip
  // intermediate impl/path segments to get a name close to the C-style
  // `<fn>.<var>` shape. Lower-case the variable
  // segment because Rust convention uses SCREAMING_SNAKE for statics
  // while C uses snake_case.
  std::string Var = Segments.back();
  for (auto &c : Var)
    c = std::tolower(static_cast<unsigned char>(c));
  return Segments.front() + "." + Var;
}

static std::string demangleRustFunction(StringRef Name) {
  if (!Name.startswith("_ZN") || !Name.endswith("E"))
    return "";
  StringRef Body = Name.drop_front(3).drop_back(1);
  SmallVector<std::string, 6> Segments;
  while (!Body.empty()) {
    size_t I = 0;
    while (I < Body.size() && std::isdigit(static_cast<unsigned char>(Body[I])))
      ++I;
    if (I == 0)
      break;
    unsigned Len = 0;
    if (Body.substr(0, I).getAsInteger(10, Len) || Len == 0)
      break;
    Body = Body.drop_front(I);
    if (Body.size() < Len)
      break;
    StringRef Seg = Body.substr(0, Len);
    Body = Body.drop_front(Len);
    if (Len == 17 && Seg[0] == 'h')
      continue;
    Segments.push_back(Seg.str());
  }
  if (Segments.size() < 2)
    return "";
  // Drop the crate (first segment); join the rest. e.g.
  // `_ZN<crate>15func_with_static17h..E` → `func_with_static`.
  std::string Out;
  for (size_t i = 1; i < Segments.size(); ++i) {
    if (!Out.empty())
      Out += "::";
    Out += Segments[i];
  }
  return Out;
}

static bool renameSingleUserStaticGlobals(Module &M) {
  bool Changed = false;
  // Collect candidate globals (internal, has-initializer, name like
  // `<crate>.<var>` or `<crate>.<var>.N`).
  SmallVector<GlobalVariable *, 8> Cands;
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasInitializer() || G.hasExternalLinkage())
      continue;
    StringRef Name = G.getName();
    // Skip private (no `<crate>.<var>` shape) and named constants.
    if (Name.startswith(".str") || Name.empty() || Name[0] == '_')
      continue;
    // Require at least one '.' so we know we have `<crate>.<var>`.
    if (Name.find('.') == StringRef::npos)
      continue;
    Cands.push_back(&G);
  }
  for (GlobalVariable *G : Cands) {
    // Find unique using function.
    SmallPtrSet<Function *, 4> Users;
    for (User *U : G->users()) {
      if (auto *I = dyn_cast<Instruction>(U))
        Users.insert(I->getParent()->getParent());
    }
    if (Users.size() != 1)
      continue;
    Function *F = *Users.begin();
    // Skip the top kernel itself — its globals stay `<crate>.<var>`.
    if (F->hasFnAttribute("fpga.top.func"))
      continue;
    // Demangle function name.
    std::string FnName = demangleRustFunction(F->getName());
    if (FnName.empty())
      continue;
    // Extract var name: from `<crate>.<var>[.N]`, drop everything before
    // the first '.', and any trailing `.N` numeric suffix.
    StringRef Old = G->getName();
    auto DotPos = Old.find('.');
    StringRef Tail = Old.substr(DotPos + 1);
    auto LastDot = Tail.find_last_of('.');
    if (LastDot != StringRef::npos) {
      // Trailing `.N` suffix? Check if everything after the last dot is digits.
      StringRef Suffix = Tail.substr(LastDot + 1);
      bool AllDigits = !Suffix.empty();
      for (char c : Suffix)
        if (!std::isdigit(static_cast<unsigned char>(c))) {
          AllDigits = false;
          break;
        }
      if (AllDigits)
        Tail = Tail.substr(0, LastDot);
    }
    // Rename: `<fn-demangle>.<var>`. Tag old name with `.tmp` first so
    // setName below doesn't fight an in-use name.
    std::string NewName = FnName + "." + Tail.str();
    G->setName(G->getName() + ".tmp");
    G->setName(NewName);
    Changed = true;
  }
  return Changed;
}

// Evaluate C++-style ROM-init calls at compile time, the way clang -O2 does
// for the C++ baseline. Pattern (coefficient_filter / rom_lookup_table_math):
//
//   static mut TABLE: [i16; N] = [0; N];          // internal global, zeroinit
//   #[inline(never)] fn table_init(t: &mut [i16; N]) { ... cos()/round() ... }
//   ... table_init(&mut TABLE); ...
//
// clang fully unrolls + constant-folds the libm calls so its a.pp.bc carries
// `constant [N x i16] [values...]` and NO init function. rustc -O leaves a
// runtime libm loop (full-unroll threshold + freestanding TLI disables libm
// folding), which would synthesize a cos() datapath — a different machine.
// This pass closes the gap with LLVM's static-constructor Evaluator: run the
// init call abstractly (libm folded through a host-linux TLI), and on success
// commit the computed elements as the global's CONSTANT initializer, drop the
// call, and drop the (now dead) init function — the exact C++ pre-reflow form.
static bool evaluateRomInitCalls(Module &M) {
  // Host TLI: the module targets fpga64 (no libc), which would veto libm
  // folding; the evaluation itself runs on the build host.
  TargetLibraryInfoImpl TLII(Triple("x86_64-unknown-linux-gnu"));
  TargetLibraryInfo TLI(TLII);
  bool Changed = false;

  SmallVector<CallInst *, 4> Cands;
  for (Function &F : M) {
    // rustc often promotes `init(&mut STATIC)` to a ZERO-arg function that
    // stores to the global directly, so accept 0- or 1-pointer-arg shapes.
    if (F.isDeclaration() || F.arg_size() > 1)
      continue;
    if (F.arg_size() == 1 && !F.getArg(0)->getType()->isPointerTy())
      continue;
    for (User *U : F.users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == &F)
          Cands.push_back(CI);
  }

  if (::getenv("VXX_DEBUG"))
    for (CallInst *CI : Cands)
      vxxDbg() << "vxx: romInit cand: " << CI->getCalledFunction()->getName()
               << " args=" << CI->arg_size() << "\n";

  for (CallInst *CI : Cands) {
    Function *F = CI->getCalledFunction();
    GlobalVariable *GV = nullptr;
    if (CI->arg_size() == 1) {
      GV = dyn_cast<GlobalVariable>(CI->getArgOperand(0)->stripPointerCasts());
    } else {
      // Zero-arg form: the ONE global the body stores into.
      for (Instruction &I : instructions(*F)) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *Root = SI->getPointerOperand()->stripPointerCasts();
          while (auto *GEP = dyn_cast<GEPOperator>(Root))
            Root = GEP->getPointerOperand()->stripPointerCasts();
          auto *G = dyn_cast<GlobalVariable>(Root);
          if (!G) { GV = nullptr; break; }
          if (GV && GV != G) { GV = nullptr; break; }
          GV = G;
        }
      }
    }
    if (!GV || !GV->hasLocalLinkage() || GV->isConstant() ||
        !GV->hasInitializer()) {
      vxxDbg() << "vxx: romInit bail(GV) " << F->getName() << "\n";
      continue;
    }
    auto *AT = dyn_cast<ArrayType>(GV->getValueType());
    if (!AT || !AT->getElementType()->isIntegerTy()) {
      vxxDbg() << "vxx: romInit bail(arrayty) " << GV->getName() << "\n";
      continue;
    }
    // The global must not be stored anywhere outside the init function.
    bool OtherStore = false;
    {
      SmallVector<Value *, 8> Work{GV};
      SmallPtrSet<Value *, 16> Seen;
      while (!Work.empty() && !OtherStore) {
        Value *V = Work.pop_back_val();
        if (!Seen.insert(V).second)
          continue;
        for (User *VU : V->users()) {
          if (isa<BitCastOperator>(VU) || isa<GEPOperator>(VU)) {
            Work.push_back(VU);
            continue;
          }
          if (auto *SI = dyn_cast<StoreInst>(VU))
            if (SI->getFunction() != F)
              OtherStore = true;
        }
      }
    }
    if (OtherStore) {
      vxxDbg() << "vxx: romInit bail(otherstore) " << GV->getName() << "\n";
      continue;
    }

    // Concrete mini-interpreter. LLVM's static-ctor Evaluator refuses any
    // loop ("looping function, cannot evaluate in reasonable time"), and the
    // init pattern IS a 0..N loop — so execute it ourselves with the
    // ConstantFold* APIs. Scope is deliberately tiny: SSA values only, memory
    // limited to element slots of GV, calls limited to what ConstantFoldCall
    // can fold (libm via the host TLI + intrinsics).
    unsigned N = AT->getNumElements();
    SmallVector<Constant *, 256> Elems(
        N, Constant::getNullValue(AT->getElementType()));
    if (auto *CA = dyn_cast<ConstantArray>(GV->getInitializer()))
      for (unsigned i = 0; i < N; ++i)
        Elems[i] = CA->getOperand(i);
    if (auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer()))
      for (unsigned i = 0; i < N; ++i)
        Elems[i] = CDA->getElementAsConstant(i);

    const DataLayout &DL = M.getDataLayout();
    DenseMap<Value *, Constant *> Vals;
    if (F->arg_size() == 1)
      Vals[F->getArg(0)] =
          ConstantExpr::getPointerCast(GV, F->getArg(0)->getType());
    auto getVal = [&](Value *V) -> Constant * {
      if (auto *C = dyn_cast<Constant>(V))
        return C;
      auto It = Vals.find(V);
      return It == Vals.end() ? nullptr : It->second;
    };
    // Resolve a folded pointer constant to an element slot of GV.
    auto elemIndex = [&](Constant *Ptr, uint64_t &Idx) -> bool {
      auto *CE = dyn_cast<ConstantExpr>(Ptr);
      if (!CE || CE->getOpcode() != Instruction::GetElementPtr ||
          CE->getOperand(0)->stripPointerCasts() != GV ||
          CE->getNumOperands() != 3)
        return false;
      auto *I0 = dyn_cast<ConstantInt>(CE->getOperand(1));
      auto *I1 = dyn_cast<ConstantInt>(CE->getOperand(2));
      if (!I0 || !I0->isZero() || !I1 || I1->getZExtValue() >= N)
        return false;
      Idx = I1->getZExtValue();
      return true;
    };

    BasicBlock *BB = &F->getEntryBlock(), *Prev = nullptr;
    uint64_t Budget = 1ull << 22;
    bool Bad = false, Done = false;
    Instruction *BadI = nullptr;
    while (!Bad && !Done) {
      // Phis read their incoming values in parallel.
      SmallVector<std::pair<PHINode *, Constant *>, 4> PhiVals;
      Instruction *Cur = &BB->front();
      while (auto *PN = dyn_cast<PHINode>(Cur)) {
        Constant *C = Prev ? getVal(PN->getIncomingValueForBlock(Prev))
                           : nullptr;
        if (!C) {
          Bad = true;
          break;
        }
        PhiVals.push_back({PN, C});
        Cur = Cur->getNextNode();
      }
      if (Bad)
        break;
      for (auto &PV : PhiVals)
        Vals[PV.first] = PV.second;

      for (Instruction *IP = Cur; IP && !Bad && !Done;
           IP = IP->getNextNode()) {
        Instruction &I = *IP;
        if (--Budget == 0) {
          Bad = true;
          BadI = &I;
          break;
        }
        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          BasicBlock *Next = nullptr;
          if (BI->isUnconditional()) {
            Next = BI->getSuccessor(0);
          } else {
            auto *Cond =
                dyn_cast_or_null<ConstantInt>(getVal(BI->getCondition()));
            if (!Cond) {
              Bad = true;
              BadI = &I;
              break;
            }
            Next = BI->getSuccessor(Cond->isOne() ? 0 : 1);
          }
          Prev = BB;
          BB = Next;
          break;
        }
        if (isa<ReturnInst>(&I)) {
          Done = true;
          break;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Constant *Ptr = getVal(SI->getPointerOperand());
          Constant *Val = getVal(SI->getValueOperand());
          uint64_t Idx;
          if (!Ptr || !Val || !elemIndex(Ptr, Idx) ||
              Val->getType() != AT->getElementType()) {
            Bad = true;
            BadI = &I;
            break;
          }
          Elems[Idx] = Val;
          continue;
        }
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Constant *Ptr = getVal(LI->getPointerOperand());
          uint64_t Idx;
          if (!Ptr || !elemIndex(Ptr, Idx) ||
              LI->getType() != AT->getElementType()) {
            Bad = true;
            BadI = &I;
            break;
          }
          Vals[&I] = Elems[Idx];
          continue;
        }
        if (auto *CB = dyn_cast<CallInst>(&I)) {
          if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
            Intrinsic::ID IID = II->getIntrinsicID();
            if (IID == Intrinsic::lifetime_start ||
                IID == Intrinsic::lifetime_end ||
                IID == Intrinsic::dbg_declare || IID == Intrinsic::dbg_value ||
                IID == Intrinsic::assume || IID == Intrinsic::sideeffect)
              continue;
          }
          Function *Callee = CB->getCalledFunction();
          SmallVector<Constant *, 4> Ops;
          for (Value *Op : CB->args()) {
            Constant *C = getVal(Op);
            if (!C) {
              Bad = true;
              BadI = &I;
              break;
            }
            Ops.push_back(C);
          }
          if (Bad)
            break;
          Constant *R =
              Callee ? ConstantFoldCall(CB, Callee, Ops, &TLI) : nullptr;
          if (!R) {
            Bad = true;
            BadI = &I;
            break;
          }
          Vals[&I] = R;
          continue;
        }
        if (isa<SelectInst>(&I) || I.isBinaryOp() || I.isCast() ||
            isa<CmpInst>(&I) || isa<GetElementPtrInst>(&I) ||
            isa<ExtractValueInst>(&I) || isa<InsertValueInst>(&I) ||
            isa<FreezeInst>(&I)) {
          SmallVector<Constant *, 4> Ops;
          for (Value *Op : I.operands()) {
            Constant *C = getVal(Op);
            if (!C) {
              Bad = true;
              BadI = &I;
              break;
            }
            Ops.push_back(C);
          }
          if (Bad)
            break;
          Constant *R = nullptr;
          if (isa<FreezeInst>(&I))
            R = isa<UndefValue>(Ops[0])
                    ? Constant::getNullValue(I.getType())
                    : Ops[0];
          else if (isa<SelectInst>(&I))
            R = ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
          else if (auto *CI2 = dyn_cast<CmpInst>(&I))
            R = ConstantFoldCompareInstOperands(CI2->getPredicate(), Ops[0],
                                                Ops[1], DL, &TLI);
          else
            R = ConstantFoldInstOperands(&I, Ops, DL, &TLI);
          if (!R || !isa<Constant>(R)) {
            Bad = true;
            BadI = &I;
            break;
          }
          // A select of two non-identical aggregates can stay symbolic; for
          // scalars ConstantExpr::getSelect folds on a ConstantInt condition.
          if (isa<ConstantExpr>(R) && !R->getType()->isPointerTy()) {
            R = ConstantFoldConstant(cast<Constant>(R), DL, &TLI);
            if (isa<ConstantExpr>(R)) {
              Bad = true;
              BadI = &I;
              break;
            }
          }
          Vals[&I] = cast<Constant>(R);
          continue;
        }
        Bad = true;
        BadI = &I;
      }
    }
    if (Bad || !Done) {
      vxxDbg() << "vxx: romInit bail(interp) " << F->getName()
               << " done=" << Done;
      if (BadI)
        vxxDbg() << " at: " << *BadI;
      vxxDbg() << "\n";
      continue;
    }

    GV->setInitializer(ConstantArray::get(AT, Elems));
    GV->setConstant(true);
    vxxDbg() << "vxx: evaluateRomInitCalls folded " << F->getName()
             << " into constant initializer of " << GV->getName() << "\n";
    CI->eraseFromParent();
    if (F->use_empty())
      F->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static bool rewriteStaticIntGlobals(Module &M) {
  bool Changed = false;
  SmallVector<GlobalVariable *, 8> Targets;
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasInitializer() || G.hasExternalLinkage())
      continue;
    auto *AnonStruct = dyn_cast<StructType>(G.getValueType());
    if (!AnonStruct || !AnonStruct->isPacked() ||
        AnonStruct->getNumElements() != 1)
      continue;
    auto *ArrTy = dyn_cast<ArrayType>(AnonStruct->getElementType(0));
    if (!ArrTy)
      continue;
    auto *ElemTy = dyn_cast<IntegerType>(ArrTy->getElementType());
    if (!ElemTy || ElemTy->getBitWidth() != 8)
      continue;
    Targets.push_back(&G);
  }
  for (GlobalVariable *Old : Targets) {
    auto *AnonStruct = cast<StructType>(Old->getValueType());
    auto *ArrTy = cast<ArrayType>(AnonStruct->getElementType(0));
    LLVMContext &Ctx = M.getContext();

    // Look at bitcast uses to figure out the intended user-facing type.
    // Rust emits `bitcast <{[N x i8]}>* @G to TgtTy*` for every load/
    // store/GEP, so any one of those gives us TgtTy. Skip 0-sized
    // bitcasts (e.g., `to {}*` introduced by xlx_bind_storage markers)
    // — those are placeholder casts for sideeffect bundles, not real
    // accesses. Also walk GEP-then-bitcast chains so multi-field
    // struct accesses (B, C fields at non-zero offsets) contribute.
    const DataLayout &DLPick = M.getDataLayout();
    Type *TgtTy = nullptr;
    auto pickTypeFromBitcast = [&](ConstantExpr *CE) -> Type * {
      auto *DstPty = dyn_cast<PointerType>(CE->getType());
      if (!DstPty)
        return nullptr;
      Type *DE = DstPty->getElementType();
      if (DLPick.getTypeAllocSize(DE) == 0)
        return nullptr;
      return DE;
    };
    // Scan ALL bitcast users and prefer an AGGREGATE target type whose alloc
    // size covers the whole global. A `static mut STATE: [i16; 7]` used both
    // element-wise (`STATE[0] = x` → `bitcast to i16*`) and indexed
    // (`bitcast to [7 x i16]*`) must resolve to the ARRAY type — taking the
    // first bitcast found (user order is arbitrary) can pick the scalar and
    // shrink the global to a single i16, whose array-typed accesses then fail
    // SYNCHK 200-61 "array with unknown size" (fir/decimator's FIR delay line).
    uint64_t WholeBytes = DLPick.getTypeAllocSize(ArrTy);
    Type *ExactAggTy = nullptr;
    auto considerType = [&](Type *T) {
      if (!T)
        return;
      if (!TgtTy)
        TgtTy = T;
      if (!ExactAggTy && T->isAggregateType() &&
          DLPick.getTypeAllocSize(T) == WholeBytes)
        ExactAggTy = T;
    };
    for (User *U : Old->users()) {
      auto *CE = dyn_cast<ConstantExpr>(U);
      if (!CE)
        continue;
      if (CE->getOpcode() == Instruction::BitCast)
        considerType(pickTypeFromBitcast(CE));
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        for (User *CU : CE->users()) {
          if (auto *BC = dyn_cast<ConstantExpr>(CU))
            if (BC->getOpcode() == Instruction::BitCast)
              considerType(pickTypeFromBitcast(BC));
        }
      }
    }
    if (ExactAggTy)
      TgtTy = ExactAggTy;
    if (!TgtTy)
      continue;

    // Parse the byte initializer back into a value of TgtTy. Handle
    // two shapes: scalar `iN` (the static-mut single-value case) and
    // `[K x T]` const array (the lookup-table case).
    Constant *NewInit = nullptr;
    Type *NewTy = nullptr;
    // Zero-initialized statics (`static mut STATE: [i16; 7] = [0; 7]`) carry a
    // ConstantAggregateZero, not a ConstantDataArray — materialize the zero
    // bytes so the array/struct parse below works for them too.
    std::string ZeroBytes;
    auto getRawBytes = [&]() -> StringRef {
      if (isa<ConstantAggregateZero>(Old->getInitializer())) {
        ZeroBytes.assign((size_t)WholeBytes, '\0');
        return StringRef(ZeroBytes);
      }
      auto *CS = dyn_cast<ConstantStruct>(Old->getInitializer());
      if (!CS || CS->getNumOperands() != 1)
        return StringRef();
      if (isa<ConstantAggregateZero>(CS->getOperand(0))) {
        ZeroBytes.assign((size_t)WholeBytes, '\0');
        return StringRef(ZeroBytes);
      }
      auto *CDA = dyn_cast<ConstantDataArray>(CS->getOperand(0));
      if (!CDA)
        return StringRef();
      return CDA->getRawDataValues();
    };
    // Recursive helper: parse `Bytes` according to `T`'s layout, returning
    // a typed `Constant*` (nested ConstantStruct / ConstantArray / ConstantInt).
    // Supports IntegerType, ArrayType, and StructType (handles
    // [N x %struct.S] for `static [Ts; N]` mirroring C++'s typed-struct
    // global lowering — closes ~250 sem on static_array_of_struct,
    // static_struct_with_array_RAM_Versal, and similar).
    const DataLayout &DL = M.getDataLayout();
    std::function<Constant *(Type *, StringRef)> parseConst =
        [&](Type *T, StringRef Bytes) -> Constant * {
      if (auto *IT = dyn_cast<IntegerType>(T)) {
        unsigned ByteCount = (IT->getBitWidth() + 7) / 8;
        if (Bytes.size() < ByteCount)
          return nullptr;
        APInt V(IT->getBitWidth(), 0);
        for (unsigned b = 0; b < ByteCount; ++b)
          V |= APInt(IT->getBitWidth(), (uint8_t)Bytes[b]) << (b * 8);
        return ConstantInt::get(IT, V);
      }
      if (auto *AT = dyn_cast<ArrayType>(T)) {
        uint64_t ElemSize = DL.getTypeAllocSize(AT->getElementType());
        SmallVector<Constant *, 16> Elems;
        Elems.reserve(AT->getNumElements());
        for (uint64_t i = 0; i < AT->getNumElements(); ++i) {
          if ((i + 1) * ElemSize > Bytes.size())
            return nullptr;
          Constant *E = parseConst(AT->getElementType(),
                                   Bytes.substr(i * ElemSize, ElemSize));
          if (!E)
            return nullptr;
          Elems.push_back(E);
        }
        return ConstantArray::get(AT, Elems);
      }
      if (auto *ST = dyn_cast<StructType>(T)) {
        // Compute field offsets manually rather than via
        // `DL.getStructLayout(ST)` — that caches the StructLayout per
        // StructType*, and after `setBody` mutates ST in-place the
        // cached layout is stale (returns offsets for the OLD field
        // count, causing field values to shift). For naturally-aligned
        // `#[repr(C)]` structs (the rustc shape we strip from), fields
        // pack sequentially with natural alignment — compute offsets
        // by summing field alloc sizes with alignment fix-up.
        SmallVector<Constant *, 8> Fields;
        Fields.reserve(ST->getNumElements());
        uint64_t Offset = 0;
        for (unsigned i = 0; i < ST->getNumElements(); ++i) {
          Type *FT = ST->getElementType(i);
          uint64_t Size = DL.getTypeAllocSize(FT);
          if (!ST->isPacked()) {
            // Round up to field's preferred alignment.
            uint64_t Align = DL.getABITypeAlignment(FT);
            if (Align > 0)
              Offset = (Offset + Align - 1) & ~(Align - 1);
          }
          if (Offset + Size > Bytes.size())
            return nullptr;
          Constant *F = parseConst(FT, Bytes.substr(Offset, Size));
          if (!F)
            return nullptr;
          Fields.push_back(F);
          Offset += Size;
        }
        return ConstantStruct::get(ST, Fields);
      }
      return nullptr; // unsupported (PointerType, FloatTy, etc.)
    };

    if (auto *NewIntTy = dyn_cast<IntegerType>(TgtTy)) {
      uint64_t Bits = NewIntTy->getBitWidth();
      if (Bits == 0 || Bits > 128)
        continue;
      // For static-mut scalars Rust currently only emits zeroinit; we
      // keep the legacy 0-init behaviour to match the existing
      // `pointer_basic.acc` etc. outputs.
      NewTy = NewIntTy;
      NewInit = ConstantInt::get(NewIntTy, 0);
    } else {
      // Try to parse the byte initializer as TgtTy (array/struct/nested).
      // If the global's byte size is a multiple of TgtTy's size > 1, treat
      // the global as `[N x TgtTy]` — Rust's `static [T; N]` lowers as
      // `<{ [N*sizeof(T) x i8] }>` with bitcasts to bare `%T*` (treating
      // the pointer as a C-style array via GEP), so we only see TgtTy=T
      // through bitcasts but the actual storage is N elements wide.
      StringRef Raw = getRawBytes();
      uint64_t TgtSize = DL.getTypeAllocSize(TgtTy);
      if (TgtSize == 0 || Raw.size() < TgtSize)
        continue;
      Type *EffectiveTy = TgtTy;
      if (Raw.size() > TgtSize && (Raw.size() % TgtSize) == 0) {
        uint64_t N = Raw.size() / TgtSize;
        // Heuristic: if `__vxx_bind_storage` markers target DIFFERENT byte
        // offsets within Old (one per sub-element), wrap the storage as a
        // named struct `{TgtTy, TgtTy, ..., TgtTy}` (N fields) instead of
        // `[N x TgtTy]`. Vitis HLS allocates one memory per struct field
        // (matching C++'s `struct { int A[10]; int B[10]; int C[10]; }`
        // shape) so per-field BIND_STORAGE pragmas can pick BRAM / LUTRAM /
        // URAM independently. Plain `[N x T]` is treated as one 2D memory
        // and only the first BIND_STORAGE survives — without this the Rust
        // port silently falls back to Auto RAM where C++ gets URAM=1.
        //
        // Only apply when:
        //   - bind_storage targets all `N` distinct offsets {0, S, 2*S, ...}
        //   - body GEPs into Old all use a constant element index (we can
        //     map them to struct-field GEPs); we approximate by only
        //     reshaping when all `Old->users()` GEPs have a constant 3rd
        //     operand. Loop-indexed array accesses (`@arr[i]`) keep the
        //     array shape.
        bool ReshapeToStruct = false;
        {
          // Collect distinct byte offsets into Old that have a
          // `xlx_bind_storage` sideeffect anchor. After
          // injectPragmaSideeffects ran earlier, the markers are
          // `llvm.sideeffect [ "xlx_bind_storage"(ptr, ...) ]` and the
          // anchor pointer is a ConstantExpr GEP rooted at Old (per-field)
          // or Old itself (whole struct).
          SmallSet<uint64_t, 8> HitOffsets;
          auto recordPtr = [&](Value *Ptr) {
            while (auto *BCI = dyn_cast<BitCastInst>(Ptr))
              Ptr = BCI->getOperand(0);
            while (auto *CEX = dyn_cast<ConstantExpr>(Ptr)) {
              if (CEX->getOpcode() == Instruction::GetElementPtr) {
                if (CEX->getOperand(0) == Old &&
                    CEX->getNumOperands() >= 4) {
                  if (auto *I0 = dyn_cast<ConstantInt>(CEX->getOperand(1)))
                    if (auto *I1 = dyn_cast<ConstantInt>(CEX->getOperand(2)))
                      if (auto *Iby = dyn_cast<ConstantInt>(CEX->getOperand(3)))
                        if (I0->isZero() && I1->isZero()) {
                          HitOffsets.insert(Iby->getZExtValue());
                          return;
                        }
                }
                return;
              }
              if (CEX->getOpcode() == Instruction::BitCast) {
                Ptr = CEX->getOperand(0);
                continue;
              }
              return;
            }
            if (Ptr == Old) HitOffsets.insert(0);
          };
          // Walk all users of Old (CallInst bundles + ConstantExpr GEPs
          // whose users are CallInsts with xlx_bind_storage bundle).
          for (User *U : Old->users()) {
            if (auto *Call = dyn_cast<CallInst>(U)) {
              if (Call->getIntrinsicID() != Intrinsic::sideeffect) continue;
              for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
                auto B = Call->getOperandBundleAt(i);
                if (B.getTagName() != "xlx_bind_storage") continue;
                if (B.Inputs.empty()) continue;
                recordPtr(B.Inputs[0]);
              }
            } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
              if (CE->getOpcode() != Instruction::GetElementPtr) continue;
              // Direct: the constant GEP itself is the sideeffect bundle
              // operand (Rust IR shape — bundle operand is `i8*
              // getelementptr (<{[N x i8]}>, @G, 0, 0, off)` without an
              // intervening bitcast). Also handle indirect: GEP → bitcast
              // → sideeffect (when the destination type is non-i8).
              SmallVector<Value *, 4> Anchors;
              Anchors.push_back(CE);
              for (User *CU : CE->users()) {
                if (auto *BC = dyn_cast<ConstantExpr>(CU))
                  if (BC->getOpcode() == Instruction::BitCast)
                    Anchors.push_back(BC);
              }
              for (Value *Anchor : Anchors) {
                for (User *AU : Anchor->users()) {
                  auto *Call = dyn_cast<CallInst>(AU);
                  if (!Call ||
                      Call->getIntrinsicID() != Intrinsic::sideeffect)
                    continue;
                  for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
                    auto B = Call->getOperandBundleAt(i);
                    if (B.getTagName() != "xlx_bind_storage") continue;
                    if (B.Inputs.empty()) continue;
                    recordPtr(B.Inputs[0]);
                  }
                }
              }
            }
          }
          // Check every per-element offset is hit and ALL Old GEP users
          // address constant elements (no loop-indexed access).
          if (HitOffsets.size() == N) {
            bool AllConstIdx = true;
            for (uint64_t i = 0; i < N; ++i)
              if (!HitOffsets.count(i * TgtSize)) {
                AllConstIdx = false;
                break;
              }
            if (AllConstIdx) {
              // Verify body GEPs into Old also use constant element index.
              for (User *U : Old->users()) {
                auto *CE2 = dyn_cast<ConstantExpr>(U);
                if (!CE2 || CE2->getOpcode() != Instruction::GetElementPtr)
                  continue;
                if (CE2->getNumOperands() < 4) continue;
                auto *Iby = dyn_cast<ConstantInt>(CE2->getOperand(3));
                if (!Iby) { AllConstIdx = false; break; }
                if (Iby->getZExtValue() % TgtSize != 0) {
                  AllConstIdx = false;
                  break;
                }
              }
              ReshapeToStruct = AllConstIdx;
            }
          }
        }
        if (ReshapeToStruct) {
          // Build a named struct `%struct.<oldname> = { TgtTy, ..., TgtTy }`
          // (N copies). Name based on the demangled global so it reads as a
          // distinct named type in the disassembled IR.
          SmallVector<Type *, 8> Fields(N, TgtTy);
          // Build a stable name from the Old name so different statics get
          // distinct named structs (avoids cross-global struct conflicts).
          std::string SName = "struct.vxx.";
          SName += Old->getName().str();
          // LLVM disallows '.' inside identifier safely but allows it;
          // however to avoid clashes use the global's hash-suffixed name
          // as-is.
          StructType *ST = StructType::create(Ctx, Fields, SName);
          EffectiveTy = ST;
        } else {
          EffectiveTy = ArrayType::get(TgtTy, N);
        }
      }
      Constant *Parsed = parseConst(EffectiveTy, Raw);
      if (!Parsed)
        continue;
      NewTy = EffectiveTy;
      NewInit = Parsed;
    }
    uint64_t Bits = NewTy->isIntegerTy() ? cast<IntegerType>(NewTy)->getBitWidth() : 0;
    // Demangle the Rust name to a C-style `<crate>.<var>` string when
    // possible — this matches the `<fn>.<var>` shape Vitis HLS Clang
    // emits for function-local statics. Fall back to the original name
    // if demangling doesn't apply.
    std::string Demangled = demangleRustGlobal(Old->getName());
    std::string OrigName =
        Demangled.empty() ? Old->getName().str() : Demangled;
    // Free up the chosen name (the old global keeps a `.tmp` suffix until
    // erased).
    Old->setName(Old->getName() + ".tmp");
    auto *NewG =
        new GlobalVariable(M, NewTy, /*isConstant=*/Old->isConstant(),
                           Old->getLinkage(), NewInit, OrigName, Old,
                           Old->getThreadLocalMode());
    NewG->setUnnamedAddr(Old->getUnnamedAddr());
    // Vitis HLS Clang sets a 512-byte alignment on internal static-mut
    // globals (its memory-pool default). Mirror that so loads/stores in
    // the body match byte-for-byte. Const arrays use natural elem
    // alignment.
    if (NewTy->isIntegerTy())
      NewG->setAlignment(Align(512));
    else if (auto *Arr = dyn_cast<ArrayType>(NewTy)) {
      Type *Inner = Arr->getElementType();
      while (auto *Inner2 = dyn_cast<ArrayType>(Inner))
        Inner = Inner2->getElementType();
      // Mutable static arrays get 512-byte alignment for "memory pool"
      // globals; constant lookup tables fall back to natural element alignment.
      if (!Old->isConstant())
        NewG->setAlignment(Align(512));
      else if (Inner->isIntegerTy())
        NewG->setAlignment(Align(llvm::PowerOf2Ceil(
            (Inner->getIntegerBitWidth() + 7) / 8)));
      else
        // Non-integer inner (struct/float) — let DataLayout pick a sane
        // ABI alignment. Mirrors what clang would emit for a const C
        // array of the same element type.
        NewG->setAlignment(Align(DL.getABITypeAlignment(Inner)));
    } else if (NewTy->isStructTy()) {
      // Reshaped to a named struct for per-field BIND_STORAGE — use the
      // same 512-byte "memory pool" alignment as the array-shape case.
      NewG->setAlignment(Align(512));
    }
    (void)Bits;

    // Fix up uses: the typical Rust pattern is
    //   bitcast (<{ [N x i8] }>* @G to TgtTy*)
    //   load/store iN  (or GEP for struct/array cases)
    // Replace the bitcast operand directly so the consumer now points
    // at NewG, then drop the dead Old global. Also handles the
    // GEP-then-bitcast pattern Rust emits for multi-field struct fields:
    //   bitcast (i8* getelementptr (<{ [N x i8] }>, ptr @G, 0, 0, off)
    //           to TgtTy*)
    // Each constant offset `off` maps to an index into NewG (when NewTy
    // is `[K x ElemTy]` with sizeof(ElemTy) dividing off). Without this
    // case the dead constant GEP keeps Old alive and Vitis HLS Clang's
    // "Lower intermediate type generated by HLSGen" pass crashes on the
    // mixed-typed leftover IR.
    auto rewriteBitCastTo = [&](ConstantExpr *BC, Constant *NewBase,
                                Type *DstElem) {
      auto *DstPty = cast<PointerType>(BC->getType());
      // Case: bitcast target equals NewBase's value type exactly.
      Type *NewBaseElem =
          cast<PointerType>(NewBase->getType())->getElementType();
      if (DstElem == NewBaseElem) {
        BC->replaceAllUsesWith(NewBase);
        return;
      }
      // Otherwise emit a new bitcast from NewBase.
      Constant *NewBC = ConstantExpr::getBitCast(NewBase, DstPty);
      BC->replaceAllUsesWith(NewBC);
    };
    SmallVector<User *, 8> Users(Old->user_begin(), Old->user_end());
    for (User *U : Users) {
      auto *CE = dyn_cast<ConstantExpr>(U);
      if (!CE)
        continue;
      // Direct bitcast `<{[N x i8]}>* @G to TgtTy*`.
      if (CE->getOpcode() == Instruction::BitCast) {
        rewriteBitCastTo(CE, NewG,
                         cast<PointerType>(CE->getType())->getElementType());
        continue;
      }
      // GEP into the byte array — extract the constant byte offset and
      // walk users for bitcasts that turn the i8* into a typed pointer.
      if (CE->getOpcode() != Instruction::GetElementPtr)
        continue;
      // Expect indices [0, 0, OFF] for `<{ [N x i8] }>` shape.
      if (CE->getNumOperands() < 4)
        continue;
      auto *I0 = dyn_cast<ConstantInt>(CE->getOperand(1));
      auto *I1 = dyn_cast<ConstantInt>(CE->getOperand(2));
      auto *Iby = dyn_cast<ConstantInt>(CE->getOperand(3));
      if (!I0 || !I1 || !Iby || !I0->isZero() || !I1->isZero())
        continue;
      uint64_t ByteOff = Iby->getZExtValue();
      // Compute index path into NewG. Support both:
      //   NewTy = [K x ElemTy]  → element index = ByteOff / sizeof(ElemTy)
      //   NewTy = {ElemTy, ElemTy, ..., ElemTy} (named struct, same field
      //          type repeated)  → field index = ByteOff / sizeof(ElemTy)
      uint64_t ElemSize = 0;
      uint64_t MaxIdx = 0;
      if (auto *NewArr = dyn_cast<ArrayType>(NewTy)) {
        ElemSize = DL.getTypeAllocSize(NewArr->getElementType());
        MaxIdx = NewArr->getNumElements();
      } else if (auto *NewST = dyn_cast<StructType>(NewTy)) {
        // Only meaningful when all fields have the same element type
        // (this is the only shape `rewriteStaticIntGlobals` ever forms).
        if (NewST->getNumElements() == 0) continue;
        ElemSize = DL.getTypeAllocSize(NewST->getElementType(0));
        MaxIdx = NewST->getNumElements();
      } else {
        continue;
      }
      if (ElemSize == 0 || ByteOff % ElemSize != 0)
        continue;
      uint64_t Idx = ByteOff / ElemSize;
      if (Idx >= MaxIdx)
        continue;
      // Build constant GEP into NewG: `gep NewTy, NewG, 0, Idx`.
      Type *Int32Ty = Type::getInt32Ty(Ctx);
      Constant *Indices[] = {ConstantInt::get(Int32Ty, 0),
                              ConstantInt::get(Int32Ty, Idx)};
      Constant *NewGep = ConstantExpr::getInBoundsGetElementPtr(
          NewTy, NewG, Indices);
      // Walk users of the old GEP — they are bitcasts to typed ptr.
      SmallVector<User *, 4> CEUsers(CE->user_begin(), CE->user_end());
      for (User *CU : CEUsers) {
        auto *BC = dyn_cast<ConstantExpr>(CU);
        if (!BC || BC->getOpcode() != Instruction::BitCast)
          continue;
        rewriteBitCastTo(BC, NewGep,
                         cast<PointerType>(BC->getType())->getElementType());
      }
      // Additionally: rewrite the GEP itself (still typed `i8*`) so any
      // remaining users — notably `xlx_bind_storage` / `stable` /
      // `fpga.dependence` sideeffect call bundle operands that didn't go
      // through a typed bitcast — point at the new typed GEP cast back to
      // i8*. Without this the sideeffect keeps the dead `<{[N x i8]}>` GEP
      // alive and the HLS backend can't associate the BIND_STORAGE pragma
      // with the typed sub-array it actually targets, silently dropping
      // the pragma (Auto RAM is selected instead of the user's BRAM/LUTRAM/
      // URAM choice).
      Type *I8PtrTy = Type::getInt8PtrTy(Ctx);
      Constant *NewGepAsI8 = ConstantExpr::getBitCast(NewGep, I8PtrTy);
      CE->replaceAllUsesWith(NewGepAsI8);
    }

    // Final sweep: any direct uses of Old that we haven't already rewritten
    // (e.g., `xlx_bind_storage` sideeffect call whose first bundle operand
    // is the bare `<{[N x i8]}>* @G` global without an intervening bitcast)
    // get migrated to a bitcast of NewG to the original aggregate type, so
    // Vitis HLS still sees the pragma anchored on the typed storage. Without
    // this the pragma stays pointed at the dead `.tmp` global and the HLS
    // backend drops it.
    // Special case for the struct-reshape path: a whole-struct
    // `xlx_bind_storage` bundle operand (Rust `bind_storage(&TS.field0,...)`
    // lowers to the whole-struct pointer because rustc represents the
    // static as a packed byte aggregate, so `&TS.a` and `&TS` produce the
    // same SSA value) would collide with the per-field bind_storage calls
    // for fields 1..N-1 (Vitis HLS error SYN 201-306: conflicting memory
    // assignment). Redirect the bundle operand to a typed field-0 GEP so
    // the per-field BRAM/LUTRAM/URAM assignments anchor on distinct
    // sub-arrays.
    if (NewTy->isStructTy() && !Old->use_empty()) {
      auto *StTy = cast<StructType>(NewTy);
      Type *Int32TyL = Type::getInt32Ty(Ctx);
      // Collect the whole-struct xlx_bind_storage calls first.
      SmallVector<CallInst *, 4> BindCalls;
      for (User *U : Old->users()) {
        if (auto *Call = dyn_cast<CallInst>(U)) {
          if (Call->getIntrinsicID() != Intrinsic::sideeffect) continue;
          for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
            auto B = Call->getOperandBundleAt(i);
            if (B.getTagName() == "xlx_bind_storage" &&
                !B.Inputs.empty() && B.Inputs[0] == Old) {
              BindCalls.push_back(Call);
              break;
            }
          }
        }
      }
      for (CallInst *Call : BindCalls) {
        // Insert a non-constant `getelementptr` instruction immediately
        // before the bind_storage call (using a runtime-style index
        // operand to defeat ConstantExpr folding that would collapse
        // `bitcast (gep (struct, NewG, 0, 0) to <{[N x i8]}>*)` back to
        // `bitcast (NewG to <{[N x i8]}>*)` and re-create the whole-struct
        // conflict). The GEP instruction stays in IR and the HLS backend
        // anchors the BIND_STORAGE pragma on field 0 specifically.
        IRBuilder<> BIR(Call);
        Value *FieldIdx[] = {ConstantInt::get(Int32TyL, 0),
                              ConstantInt::get(Int32TyL, 0)};
        Value *Field0GepI = BIR.CreateInBoundsGEP(StTy, NewG, FieldIdx,
                                                   "vxx_bind_field0");
        Value *Field0AsOldTyV = BIR.CreateBitCast(Field0GepI,
                                                   Old->getType());
        SmallVector<OperandBundleDef, 2> NewBundles;
        for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
          auto B = Call->getOperandBundleAt(i);
          SmallVector<Value *, 4> Inputs(B.Inputs.begin(), B.Inputs.end());
          if (B.getTagName() == "xlx_bind_storage" && !Inputs.empty() &&
              Inputs[0] == Old)
            Inputs[0] = Field0AsOldTyV;
          NewBundles.emplace_back(B.getTagName().str(), Inputs);
        }
        SmallVector<Value *, 4> Args(Call->arg_begin(), Call->arg_end());
        CallInst *NewCall = BIR.CreateCall(Call->getCalledFunction(), Args,
                                            NewBundles);
        NewCall->copyMetadata(*Call);
        Call->eraseFromParent();
      }
    }
    if (!Old->use_empty()) {
      Constant *NewAsOldTy =
          ConstantExpr::getBitCast(NewG, Old->getType());
      Old->replaceAllUsesWith(NewAsOldTy);
    }

    Old->removeDeadConstantUsers();
    if (Old->use_empty())
      Old->eraseFromParent();

    // Bump the alignment on every load/store of NewG to match the
    // global's 512-byte preferred alignment, so each access is `align 512`
    // rather than the default `align 4`.
    Align WantAlign = NewG->getAlign().valueOrOne();
    for (User *U : NewG->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U))
        LI->setAlignment(WantAlign);
      else if (auto *SI = dyn_cast<StoreInst>(U))
        SI->setAlignment(WantAlign);
    }

    Changed = true;
  }
  return Changed;
}

static bool redirectWholeStructBindStorage(Module &M) {
  bool Changed = false;
  unsigned Total = 0;
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  // Collect candidates first; mutating M.globals() during iteration is unsafe.
  SmallVector<GlobalVariable *, 4> Candidates;
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasInitializer() || G.hasExternalLinkage())
      continue;
    auto *StTy = dyn_cast<StructType>(G.getValueType());
    if (!StTy || StTy->getNumElements() < 2)
      continue;
    // Only meaningful when all fields have the same type — otherwise the
    // per-field BIND_STORAGE bundle types wouldn't agree (it's the
    // homogeneous case that triggers `__vxx_bind_storage` per field).
    Type *F0Ty = StTy->getElementType(0);
    bool Homogeneous = true;
    for (unsigned i = 1; i < StTy->getNumElements(); ++i)
      if (StTy->getElementType(i) != F0Ty) { Homogeneous = false; break; }
    if (!Homogeneous)
      continue;
    Candidates.push_back(&G);
  }
  for (GlobalVariable *G : Candidates) {
    auto *StTy = cast<StructType>(G->getValueType());
    unsigned NumFields = StTy->getNumElements();
    Type *F0Ty = StTy->getElementType(0);
    // Scan bind_storage anchors: collect (CallInst, FieldIdx, IsWhole).
    struct BindHit {
      CallInst *Call;
      uint64_t FieldIdx;
      bool IsWhole;
    };
    SmallVector<BindHit, 4> Hits;
    SmallSet<uint64_t, 4> HitFields;
    bool SawWhole = false;
    auto inspectCall = [&](CallInst *Call, bool IsWhole, uint64_t FieldIdx) {
      if (Call->getIntrinsicID() != Intrinsic::sideeffect) return;
      for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
        auto B = Call->getOperandBundleAt(i);
        if (B.getTagName() != "xlx_bind_storage") continue;
        if (B.Inputs.empty()) continue;
        Hits.push_back({Call, FieldIdx, IsWhole});
        if (IsWhole) SawWhole = true;
        else HitFields.insert(FieldIdx);
        break;
      }
    };
    for (User *U : G->users()) {
      if (auto *Call = dyn_cast<CallInst>(U)) {
        inspectCall(Call, /*IsWhole=*/true, 0);
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          for (User *BCU : CE->users())
            if (auto *Call = dyn_cast<CallInst>(BCU))
              inspectCall(Call, /*IsWhole=*/true, 0);
        } else if (CE->getOpcode() == Instruction::GetElementPtr) {
          if (CE->getNumOperands() < 3) continue;
          auto *I0 = dyn_cast<ConstantInt>(CE->getOperand(1));
          auto *I1 = dyn_cast<ConstantInt>(CE->getOperand(2));
          if (!I0 || !I1 || !I0->isZero()) continue;
          uint64_t FieldIdx = I1->getZExtValue();
          for (User *GU : CE->users()) {
            if (auto *Call = dyn_cast<CallInst>(GU)) {
              inspectCall(Call, /*IsWhole=*/false, FieldIdx);
            } else if (auto *BC = dyn_cast<ConstantExpr>(GU)) {
              if (BC->getOpcode() != Instruction::BitCast) continue;
              for (User *BCU : BC->users())
                if (auto *Call = dyn_cast<CallInst>(BCU))
                  inspectCall(Call, /*IsWhole=*/false, FieldIdx);
            }
          }
        }
      }
    }
    // Need a whole-struct call (field 0 collision source) AND at least one
    // distinct per-field anchor (else there's no collision to resolve).
    if (!SawWhole || HitFields.empty())
      continue;
    // If field 0 is already explicitly anchored via a GEP, the C++-style
    // separation already exists — skip. (This shouldn't happen in
    // practice because rustc collapses `&G.field0` to `&G`.)
    if (HitFields.count(0))
      continue;
    // Construct N new globals, one per field. The initializer for field i
    // is the i-th element of G's StructInitializer.
    Constant *Init = G->getInitializer();
    auto *SInit = dyn_cast<ConstantStruct>(Init);
    Constant *Zeros = dyn_cast<ConstantAggregateZero>(Init);
    if (!SInit && !Zeros)
      continue;
    SmallVector<GlobalVariable *, 4> Fields;
    Fields.reserve(NumFields);
    std::string BaseName = G->getName().str();
    for (unsigned i = 0; i < NumFields; ++i) {
      Constant *FInit = SInit ? SInit->getOperand(i)
                              : Constant::getNullValue(F0Ty);
      std::string FName = BaseName + ".__field" + std::to_string(i);
      auto *NewG = new GlobalVariable(M, F0Ty, /*isConstant=*/G->isConstant(),
                                       G->getLinkage(), FInit, FName, G,
                                       G->getThreadLocalMode());
      NewG->setUnnamedAddr(G->getUnnamedAddr());
      // Mirror the 512-byte "memory pool" alignment used elsewhere
      // (matches what rewriteStaticIntGlobals uses for mutable statics).
      if (!G->isConstant())
        NewG->setAlignment(Align(512));
      Fields.push_back(NewG);
    }
    // Rewrite users:
    //   - Whole-struct ConstantExpr bitcast `(StructTy* G to T*)` →
    //     `bitcast (FieldTy* @G.field0 to T*)`.
    //   - Direct use of G (e.g., bundle operand `@G` after unwind) →
    //     `bitcast (@G.field0 to StructTy*)`.
    //   - Per-field GEP `getelementptr (StructTy, @G, 0, i)` →
    //     `@G.field<i>` (just the global itself, when result type matches).
    auto rewriteFieldGep = [&](ConstantExpr *GEP) {
      uint64_t I = cast<ConstantInt>(GEP->getOperand(2))->getZExtValue();
      if (I >= NumFields) return;
      Constant *Repl = Fields[I];
      // If GEP type matches field type (`[K x ElemTy]*`), use directly.
      auto *DstPty = cast<PointerType>(GEP->getType());
      if (DstPty->getElementType() != F0Ty) {
        // Cast to the GEP's result type for ConstantExpr-shape compatibility.
        Repl = ConstantExpr::getBitCast(Fields[I], DstPty);
      }
      GEP->replaceAllUsesWith(Repl);
    };
    auto rewriteWholeBitcast = [&](ConstantExpr *BC) {
      auto *DstPty = cast<PointerType>(BC->getType());
      Constant *Repl = ConstantExpr::getBitCast(Fields[0], DstPty);
      BC->replaceAllUsesWith(Repl);
    };
    SmallVector<User *, 8> Users(G->user_begin(), G->user_end());
    for (User *U : Users) {
      if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          rewriteWholeBitcast(CE);
        } else if (CE->getOpcode() == Instruction::GetElementPtr &&
                   CE->getNumOperands() >= 3) {
          auto *I0 = dyn_cast<ConstantInt>(CE->getOperand(1));
          auto *I1 = dyn_cast<ConstantInt>(CE->getOperand(2));
          if (!I0 || !I1 || !I0->isZero()) continue;
          rewriteFieldGep(CE);
        }
      }
      // Direct CallInst user (bind_storage bundle operand `@G` itself):
      // rebuild the call with the bundle pointing at `bitcast(@G.field0 to G->getType())`.
      else if (auto *Call = dyn_cast<CallInst>(U)) {
        if (Call->getIntrinsicID() != Intrinsic::sideeffect) continue;
        IRBuilder<> BIR(Call);
        SmallVector<OperandBundleDef, 2> NewBundles;
        bool Replaced = false;
        for (unsigned i = 0; i < Call->getNumOperandBundles(); ++i) {
          auto B = Call->getOperandBundleAt(i);
          SmallVector<Value *, 4> Inputs(B.Inputs.begin(), B.Inputs.end());
          for (Value *&In : Inputs) {
            if (In == G) {
              In = ConstantExpr::getBitCast(Fields[0], G->getType());
              Replaced = true;
            }
          }
          NewBundles.emplace_back(B.getTagName().str(), Inputs);
        }
        if (!Replaced) continue;
        SmallVector<Value *, 4> Args(Call->arg_begin(), Call->arg_end());
        CallInst *NewCall = BIR.CreateCall(Call->getCalledFunction(), Args,
                                            NewBundles);
        NewCall->copyMetadata(*Call);
        Call->eraseFromParent();
      }
      (void)Int32Ty; // kept for future per-field instruction insertion
    }
    G->removeDeadConstantUsers();
    if (G->use_empty()) {
      G->eraseFromParent();
    } else {
      // Some uses (e.g., debug-info references) may remain; redirect them
      // to a bitcast of field 0 so the dead global can still be dropped.
      Constant *F0AsOld = ConstantExpr::getBitCast(Fields[0], G->getType());
      G->replaceAllUsesWith(F0AsOld);
      G->eraseFromParent();
    }
    Total += NumFields;
    Changed = true;
  }
  if (Total > 0)
    vxxDbg() << "vxx: split " << Total
           << " per-field global(s) from struct global(s) for BIND_STORAGE\n";
  return Changed;
}

static bool collapseBitIntNarrowRoundtrip(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> ToErase;
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      if (!AI)
        continue;
      auto *AllocTy = dyn_cast<IntegerType>(AI->getAllocatedType());
      if (!AllocTy)
        continue;
      // The narrow Type<N> alloca tends to have align 16 from i128's
      // alignment requirement on the bitcast — but we don't strictly
      // require that; just walk uses.
      LoadInst *LoadInIN = nullptr;
      StoreInst *StoreI128 = nullptr;
      bool BadUse = false;
      for (User *U : AI->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          if (LI->getType() != AllocTy) {
            BadUse = true;
            break;
          }
          if (LoadInIN) {
            BadUse = true;
            break;
          }
          LoadInIN = LI;
        } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
          auto *DstPty = dyn_cast<PointerType>(BC->getType());
          if (!DstPty)
            continue;
          if (auto *DstIntTy = dyn_cast<IntegerType>(DstPty->getElementType())) {
            if (DstIntTy->getBitWidth() == 128) {
              for (User *BU : BC->users()) {
                if (auto *SI = dyn_cast<StoreInst>(BU)) {
                  if (StoreI128) {
                    BadUse = true;
                    break;
                  }
                  StoreI128 = SI;
                }
              }
            }
            // bitcast back to i8* for lifetime intrinsics is fine
          }
        }
        // Allow lifetime intrinsics on the bitcasted i8* (handled separately)
      }
      if (BadUse || !LoadInIN || !StoreI128)
        continue;

      // Replace the load with `trunc i128 (storedValue) to iN`.
      IRBuilder<> B(LoadInIN);
      Value *Truncated = B.CreateTrunc(StoreI128->getValueOperand(), AllocTy);
      LoadInIN->replaceAllUsesWith(Truncated);
      ToErase.push_back(LoadInIN);
      ToErase.push_back(StoreI128);
      // The bitcast i128* and the alloca + lifetime intrinsics are now
      // dead; subsequent dead-code sweep will pick them up.
      Changed = true;
    }
  }
  for (Instruction *I : ToErase)
    I->eraseFromParent();
  return Changed;
}

static bool collapseBitIntShlAshrPair(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Ashr = dyn_cast<BinaryOperator>(&I);
      if (!Ashr || Ashr->getOpcode() != Instruction::AShr || !Ashr->isExact())
        continue;
      auto *Shl = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
      if (!Shl || Shl->getOpcode() != Instruction::Shl)
        continue;
      auto *AshrAmtC = dyn_cast<ConstantInt>(Ashr->getOperand(1));
      auto *ShlAmtC = dyn_cast<ConstantInt>(Shl->getOperand(1));
      if (!AshrAmtC || !ShlAmtC)
        continue;
      uint64_t AshrK = AshrAmtC->getZExtValue();
      uint64_t ShlK = ShlAmtC->getZExtValue();
      auto *IntTy = dyn_cast<IntegerType>(Ashr->getType());
      if (!IntTy)
        continue;
      uint64_t TotalBits = IntTy->getBitWidth();
      if (ShlK >= TotalBits || AshrK >= TotalBits)
        continue;
      Value *X = Shl->getOperand(0);
      if (AshrK == ShlK) {
        // `shl X, K; ashr exact, K` = sext-extract of low (TotalBits-K)
        // bits. If X = sext iN to i_TotalBits with N <= TotalBits-K,
        // the pair is a no-op — replace with X directly.
        uint64_t KeptBits = TotalBits - ShlK;
        if (auto *Sext = dyn_cast<SExtInst>(X)) {
          unsigned N = Sext->getOperand(0)->getType()->getIntegerBitWidth();
          if (N <= KeptBits) {
            Ashr->replaceAllUsesWith(X);
            Dead.push_back(Ashr);
            if (Shl->use_empty())
              Dead.push_back(Shl);
            Changed = true;
          }
        }
      } else if (ShlK > AshrK) {
        // `shl X, K1; ashr exact, K2` with K1 > K2. The `exact` flag only
        // asserts the ashr's shifted-out LOW bits are zero — it says nothing
        // about the HIGH K1 bits of X the shl destroyed. The composite is
        //   sext(trunc_{TotalBits-K1}(X)) << (K1-K2)
        // which equals `shl X, K1-K2` ONLY when X provably fits signed in
        // TotalBits-K1 bits. Rewriting unconditionally MISCOMPILES the
        // BitFixed sign-normalize idiom `(raw & mask) << K1 >>a K2` in the
        // cpp_proxy adapter (raw carried in a WIDE i32: `& 0x3F` does not fit
        // signed-6, the collapse turned sext-of-low-6 into a plain zext →
        // using_fixed_point RTL computed 0xFFFFFFFF*19 instead of -1*19 on
        // every negative in2 — caught by the faithful TB's cosim post-check).
        uint64_t KeptBits = TotalBits - ShlK;
        auto *SextX = dyn_cast<SExtInst>(X);
        if (SextX &&
            SextX->getOperand(0)->getType()->getIntegerBitWidth() <= KeptBits) {
          // X provably fits: the fast path is sound.
          IRBuilder<> B(Ashr);
          Value *NewShift = B.CreateShl(
              X, ConstantInt::get(IntTy, ShlK - AshrK), "", /*HasNUW=*/false,
              /*HasNSW=*/Shl->hasNoSignedWrap());
          Ashr->replaceAllUsesWith(NewShift);
          Dead.push_back(Ashr);
          if (Shl->use_empty())
            Dead.push_back(Shl);
          Changed = true;
        } else if (KeptBits > 0) {
          // Sound general form: materialize the sign-extract explicitly.
          // This also EXPOSES the narrow value (trunc/sext) to the
          // downstream BitFixed multiply-narrowing peepholes.
          IRBuilder<> B(Ashr);
          Type *NarrowTy = IntegerType::get(F.getContext(), (unsigned)KeptBits);
          Value *T = B.CreateTrunc(X, NarrowTy);
          Value *S = B.CreateSExt(T, IntTy);
          Value *NewShift =
              B.CreateShl(S, ConstantInt::get(IntTy, ShlK - AshrK));
          Ashr->replaceAllUsesWith(NewShift);
          Dead.push_back(Ashr);
          if (Shl->use_empty())
            Dead.push_back(Shl);
          Changed = true;
        }
      }
      // K1 < K2 case: would require trunc/sext rewrite; skip.
    }
  }
  for (Instruction *I : Dead)
    I->eraseFromParent();
  return Changed;
}

static bool collapseBitIntMulShifts(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Ashr = dyn_cast<BinaryOperator>(&I);
      if (!Ashr || Ashr->getOpcode() != Instruction::AShr || !Ashr->isExact())
        continue;
      auto *AshrAmtC = dyn_cast<ConstantInt>(Ashr->getOperand(1));
      if (!AshrAmtC)
        continue;
      uint64_t K = AshrAmtC->getZExtValue();
      auto *Mul = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
      if (!Mul || Mul->getOpcode() != Instruction::Mul)
        continue;
      // Find the shl operand of mul.
      Value *Other = nullptr;
      Value *Y = nullptr;
      bool MulNSW = Mul->hasNoSignedWrap();
      for (unsigned Op = 0; Op < 2; ++Op) {
        auto *Shl = dyn_cast<BinaryOperator>(Mul->getOperand(Op));
        if (!Shl || Shl->getOpcode() != Instruction::Shl)
          continue;
        auto *ShlAmtC = dyn_cast<ConstantInt>(Shl->getOperand(1));
        if (!ShlAmtC || ShlAmtC->getZExtValue() != K)
          continue;
        Y = Shl->getOperand(0);
        Other = Mul->getOperand(1 - Op);
        break;
      }
      if (!Y || !Other)
        continue;
      IRBuilder<> B(Ashr);
      // Order operands as `mul Y, Other` — Y is the operand that was
      // originally inside the `shl` (in BitInt's emission, the
      // shifted operand is the LHS of the source-level multiply, e.g.
      // `a` in `a * b`). Matching that order keeps the diff against
      // golden's commutative pairing tight.
      Value *NewMul = B.CreateMul(Y, Other, "", /*HasNUW=*/false,
                                  /*HasNSW=*/MulNSW);
      Ashr->replaceAllUsesWith(NewMul);
      Dead.push_back(Ashr);
      if (Mul->use_empty())
        Dead.push_back(Mul);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    I->eraseFromParent();
  return Changed;
}

static bool narrowBitIntMulTrunc(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Trunc = dyn_cast<TruncInst>(&I);
      if (!Trunc)
        continue;
      auto *Mul = dyn_cast<BinaryOperator>(Trunc->getOperand(0));
      if (!Mul || Mul->getOpcode() != Instruction::Mul)
        continue;
      auto *XSext = dyn_cast<SExtInst>(Mul->getOperand(0));
      auto *YSext = dyn_cast<SExtInst>(Mul->getOperand(1));
      if (!XSext || !YSext)
        continue;
      Type *DstTy = Trunc->getType();
      auto *DstIntTy = dyn_cast<IntegerType>(DstTy);
      if (!DstIntTy)
        continue;
      unsigned C = DstIntTy->getBitWidth();
      unsigned A = XSext->getOperand(0)->getType()->getIntegerBitWidth();
      unsigned B = YSext->getOperand(0)->getType()->getIntegerBitWidth();
      if (A + B > C)
        continue;
      IRBuilder<> Bldr(Trunc);
      Value *XC = Bldr.CreateSExt(XSext->getOperand(0), DstTy);
      Value *YC = Bldr.CreateSExt(YSext->getOperand(0), DstTy);
      Value *NewMul = Bldr.CreateMul(XC, YC, "", /*HasNUW=*/false,
                                     /*HasNSW=*/Mul->hasNoSignedWrap());
      Trunc->replaceAllUsesWith(NewMul);
      Dead.push_back(Trunc);
      if (Mul->use_empty())
        Dead.push_back(Mul);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    I->eraseFromParent();
  return Changed;
}

// `__vxx_static_apint(ptr, W, signed)` — retype a function-local
// `static [Ap(U)Int<W>; K]` from rustc's [K x i128] CTFE carrier layout to
// the clang [K x iW] form (`static const ap_int<10> A[10]` => [10 x i10]).
// The declared width only exists in the Rust type, so the top macro passes
// it via this marker; every element is truncated (values are normalized) and
// loads are rewritten to `load iW` + sext/zext back to the i128 carrier —
// the existing bitint peepholes collapse the roundtrips.
static bool narrowStaticApIntArrays(Module &M) {
  Function *Marker = M.getFunction("__vxx_static_apint");
  if (!Marker)
    return false;
  LLVMContext &Ctx = M.getContext();
  SmallVector<CallInst *, 4> Calls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  bool Changed = false;
  vxxDbg() << "vxx: static_apint calls=" << Calls.size() << "\n";
  for (CallInst *CI : Calls) {
    Value *Root = CI->getArgOperand(0)->stripPointerCasts();
    // NAME.as_ptr() lowers to a [0,0] GEP constexpr on the global.
    if (auto *CE = dyn_cast<ConstantExpr>(Root))
      if (CE->getOpcode() == Instruction::GetElementPtr)
        Root = CE->getOperand(0)->stripPointerCasts();
    auto *GV = dyn_cast<GlobalVariable>(Root);
    auto *WC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto *SC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!GV || !WC || !SC || !GV->isConstant() || !GV->hasInitializer()) {
      vxxDbg() << "vxx: static_apint bail(root) " << *Root << "\n";
      continue;
    }
    auto *AT = dyn_cast<ArrayType>(GV->getValueType());
    if (!AT || !AT->getElementType()->isIntegerTy(128)) {
      vxxDbg() << "vxx: static_apint bail(type) " << GV->getName() << " "
               << *GV->getValueType() << "\n";
      continue;
    }
    unsigned W = (unsigned)WC->getZExtValue();
    bool Signed = !SC->isZero();
    if (W == 0 || W >= 128)
      continue;
    unsigned K = AT->getNumElements();
    Type *NewElem = IntegerType::get(Ctx, W);
    ArrayType *NewAT = ArrayType::get(NewElem, K);
    SmallVector<Constant *, 16> Elems;
    bool Bad = false;
    for (unsigned i = 0; i < K; ++i) {
      auto *E = dyn_cast_or_null<ConstantInt>(
          GV->getInitializer()->getAggregateElement(i));
      if (!E) { Bad = true; break; }
      Elems.push_back(ConstantInt::get(NewElem, E->getValue().trunc(W)));
    }
    if (Bad)
      continue;
    // All users must be GEP->load (reads) or marker/spec calls (operand swap).
    SmallVector<GetElementPtrInst *, 8> Geps;
    SmallVector<std::pair<CallInst *, unsigned>, 4> CallOps;
    for (User *U : GV->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        for (User *GU : GEP->users())
          if (!isa<LoadInst>(GU)) { Bad = true; break; }
        Geps.push_back(GEP);
      } else if (auto *UC = dyn_cast<CallInst>(U)) {
        for (unsigned oi = 0; oi < UC->arg_size(); ++oi)
          if (UC->getArgOperand(oi) == GV)
            CallOps.push_back({UC, oi});
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        // the marker's own [0,0] GEP constexpr — dropped with the marker
        for (User *CU : CE->users())
          if (CU != CI && !isa<CallInst>(CU)) { Bad = true; break; }
      } else {
        Bad = true;
      }
      if (Bad) break;
    }
    if (Bad) {
      vxxDbg() << "vxx: static_apint bail(users) " << GV->getName() << "\n";
      continue;
    }
    auto *NewGV = new GlobalVariable(M, NewAT, true, GV->getLinkage(),
                                     ConstantArray::get(NewAT, Elems), "");
    NewGV->takeName(GV);
    NewGV->setAlignment(MaybeAlign(512));
    NewGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    for (GetElementPtrInst *GEP : Geps) {
      IRBuilder<> B(GEP);
      SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
      Value *NG = B.CreateInBoundsGEP(NewAT, NewGV, Idx);
      SmallVector<LoadInst *, 4> Loads;
      for (User *GU : GEP->users())
        Loads.push_back(cast<LoadInst>(GU));
      for (LoadInst *LI : Loads) {
        IRBuilder<> LB(LI);
        Value *NL = LB.CreateLoad(NewElem, NG);
        Value *Ext = Signed ? LB.CreateSExt(NL, LI->getType())
                            : LB.CreateZExt(NL, LI->getType());
        LI->replaceAllUsesWith(Ext);
        LI->eraseFromParent();
      }
      GEP->eraseFromParent();
    }
    for (auto &CO : CallOps)
      CO.first->setArgOperand(CO.second, NewGV);
    vxxDbg() << "vxx: narrowed static apint array " << NewGV->getName()
             << " to [" << K << " x i" << W << "]\n";
    Changed = true;
  }
  for (CallInst *CI : Calls)
    CI->eraseFromParent();
  // drop the now-dead old globals (only marker constexprs may remain)
  SmallVector<GlobalVariable *, 4> Dead;
  for (GlobalVariable &G : M.globals())
    if (G.getName().empty() || G.use_empty())
      if (G.hasLocalLinkage() && G.use_empty())
        Dead.push_back(&G);
  for (GlobalVariable *G : Dead)
    G->eraseFromParent();
  if (Marker->use_empty())
    Marker->eraseFromParent();
  return Changed;
}

static bool narrowStaticIntGlobalToBitInt(Module &M) {
  bool Changed = false;
  SmallVector<GlobalVariable *, 4> Targets;
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasInitializer() || G.hasExternalLinkage())
      continue;
    auto *IntTy = dyn_cast<IntegerType>(G.getValueType());
    if (!IntTy || IntTy->getBitWidth() < 32)
      continue; // only consider widish ints
    auto *Init = dyn_cast<ConstantInt>(G.getInitializer());
    if (!Init || !Init->isZero())
      continue; // only handle zero-init for safety
    Targets.push_back(&G);
  }
  for (GlobalVariable *G : Targets) {
    unsigned Bits = cast<IntegerType>(G->getValueType())->getBitWidth();
    Optional<uint64_t> CommonK;
    SmallVector<LoadInst *, 4> Loads;
    SmallVector<StoreInst *, 4> Stores;
    bool Bail = false;
    for (User *U : G->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        // The load's only use must be `ashr exact (shl LD, K), K`.
        if (!LI->hasOneUse()) {
          Bail = true;
          break;
        }
        auto *Shl = dyn_cast<BinaryOperator>(*LI->user_begin());
        if (!Shl || Shl->getOpcode() != Instruction::Shl) {
          Bail = true;
          break;
        }
        if (!Shl->hasOneUse()) {
          Bail = true;
          break;
        }
        auto *Ashr = dyn_cast<BinaryOperator>(*Shl->user_begin());
        if (!Ashr || Ashr->getOpcode() != Instruction::AShr ||
            !Ashr->isExact()) {
          Bail = true;
          break;
        }
        auto *ShlAmt = dyn_cast<ConstantInt>(Shl->getOperand(1));
        auto *AshrAmt = dyn_cast<ConstantInt>(Ashr->getOperand(1));
        if (!ShlAmt || !AshrAmt || ShlAmt->getValue() != AshrAmt->getValue()) {
          Bail = true;
          break;
        }
        uint64_t K = ShlAmt->getZExtValue();
        if (CommonK && *CommonK != K) {
          Bail = true;
          break;
        }
        CommonK = K;
        Loads.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() != G) {
          Bail = true;
          break;
        }
        // The stored value must be `ashr exact (shl V, K), K`.
        auto *Ashr = dyn_cast<BinaryOperator>(SI->getValueOperand());
        if (!Ashr || Ashr->getOpcode() != Instruction::AShr ||
            !Ashr->isExact()) {
          Bail = true;
          break;
        }
        auto *Shl = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
        if (!Shl || Shl->getOpcode() != Instruction::Shl) {
          Bail = true;
          break;
        }
        auto *ShlAmt = dyn_cast<ConstantInt>(Shl->getOperand(1));
        auto *AshrAmt = dyn_cast<ConstantInt>(Ashr->getOperand(1));
        if (!ShlAmt || !AshrAmt || ShlAmt->getValue() != AshrAmt->getValue()) {
          Bail = true;
          break;
        }
        uint64_t K = ShlAmt->getZExtValue();
        if (CommonK && *CommonK != K) {
          Bail = true;
          break;
        }
        CommonK = K;
        Stores.push_back(SI);
      } else {
        Bail = true;
        break;
      }
    }
    if (Bail || !CommonK)
      continue;
    unsigned NarrowBits = Bits - *CommonK;
    if (NarrowBits == 0 || NarrowBits >= Bits)
      continue;
    IntegerType *NarrowTy = IntegerType::get(M.getContext(), NarrowBits);
    // Create the new narrow global (zero-init).
    auto *NewG = new GlobalVariable(
        M, NarrowTy, /*isConstant=*/false,
        G->getLinkage(),
        ConstantInt::get(NarrowTy, 0),
        "", G, G->getThreadLocalMode());
    NewG->setUnnamedAddr(G->getUnnamedAddr());
    NewG->setAlignment(MaybeAlign(G->getAlignment()));
    NewG->takeName(G);
    // Rewrite each load: replace `ashr exact (shl LD, K), K` with
    // `sext (load NewG) to iN`.
    for (LoadInst *LI : Loads) {
      auto *Shl = cast<BinaryOperator>(*LI->user_begin());
      auto *Ashr = cast<BinaryOperator>(*Shl->user_begin());
      IRBuilder<> B(LI);
      auto *NewLI = B.CreateLoad(NarrowTy, NewG);
      NewLI->setAlignment(LI->getAlign());
      Value *Sexted = B.CreateSExt(NewLI, LI->getType());
      Ashr->replaceAllUsesWith(Sexted);
      Ashr->eraseFromParent();
      Shl->eraseFromParent();
      LI->eraseFromParent();
    }
    // Rewrite each store: take the value before the clamp (the `V` in
    // `ashr (shl V, K), K`) and store `trunc V` to NewG.
    for (StoreInst *SI : Stores) {
      auto *Ashr = cast<BinaryOperator>(SI->getValueOperand());
      auto *Shl = cast<BinaryOperator>(Ashr->getOperand(0));
      Value *V = Shl->getOperand(0);
      IRBuilder<> B(SI);
      Value *Trunc = B.CreateTrunc(V, NarrowTy);
      auto *NewSI = B.CreateStore(Trunc, NewG);
      NewSI->setAlignment(SI->getAlign());
      SI->eraseFromParent();
      if (Ashr->use_empty())
        Ashr->eraseFromParent();
      if (Shl->use_empty())
        Shl->eraseFromParent();
    }
    G->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static bool narrowSextClampedPhi(Function &F) {
  bool Changed = false;
  // Collect candidate phis.
  SmallVector<PHINode *, 4> Cands;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Phi = dyn_cast<PHINode>(&I);
      if (!Phi)
        break; // PHIs are at start of block
      auto *IntTy = dyn_cast<IntegerType>(Phi->getType());
      if (!IntTy || IntTy->getBitWidth() != 128)
        continue;
      Cands.push_back(Phi);
    }
  }
  for (PHINode *Phi : Cands) {
    // Each incoming value must be either:
    //  - a constant int that fits in i(128-K)
    //  - `ashr exact (shl V, K), K` for the same K
    //  - another already-narrowed-PHI (not handled here; bail)
    Optional<uint64_t> CommonK;
    SmallVector<Value *, 4> Sources; // narrowed sources (V or const)
    SmallVector<bool, 4> IsConst;
    bool Bail = false;
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      Value *V = Phi->getIncomingValue(i);
      if (auto *CI = dyn_cast<ConstantInt>(V)) {
        Sources.push_back(CI);
        IsConst.push_back(true);
        continue;
      }
      // Accept either `ashr exact (shl V, K), K` (the BitInt clamp
      // pattern, V is i128 — will be trunc'd later) or `sext iN to
      // i128` where N <= 128 (already-clamped value, e.g. an outer
      // phi narrowed in a previous iteration). For the latter the
      // effective K is 128-N and the source is already iN.
      uint64_t K = 0;
      Value *NarrowSrc = nullptr;
      if (auto *Ashr = dyn_cast<BinaryOperator>(V)) {
        if (Ashr->getOpcode() == Instruction::AShr && Ashr->isExact()) {
          auto *AshrAmt = dyn_cast<ConstantInt>(Ashr->getOperand(1));
          auto *Shl = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
          if (AshrAmt && Shl && Shl->getOpcode() == Instruction::Shl) {
            auto *ShlAmt = dyn_cast<ConstantInt>(Shl->getOperand(1));
            if (ShlAmt && ShlAmt->getValue() == AshrAmt->getValue()) {
              K = AshrAmt->getZExtValue();
              if (K > 0 && K < 128)
                NarrowSrc = Shl->getOperand(0);
            }
          }
        }
      }
      if (!NarrowSrc) {
        if (auto *Sext = dyn_cast<SExtInst>(V)) {
          unsigned N = Sext->getSrcTy()->getIntegerBitWidth();
          if (N > 0 && N < 128) {
            K = 128 - N;
            NarrowSrc = Sext->getOperand(0);
          }
        }
      }
      if (!NarrowSrc) {
        Bail = true;
        break;
      }
      if (CommonK && *CommonK != K) {
        Bail = true;
        break;
      }
      CommonK = K;
      Sources.push_back(NarrowSrc);
      IsConst.push_back(false);
    }
    if (Bail || !CommonK)
      continue;
    unsigned NarrowBits = 128 - *CommonK;
    IntegerType *NarrowTy = IntegerType::get(F.getContext(), NarrowBits);
    // Verify constant incoming values fit signedly into NarrowTy.
    for (unsigned i = 0; i < Sources.size(); ++i) {
      if (!IsConst[i])
        continue;
      auto *CI = cast<ConstantInt>(Sources[i]);
      APInt V = CI->getValue();
      if (V.getMinSignedBits() > NarrowBits) {
        Bail = true;
        break;
      }
    }
    if (Bail)
      continue;
    // Build the new phi.
    IRBuilder<> EB(&*F.getEntryBlock().getFirstInsertionPt());
    PHINode *NewPhi =
        PHINode::Create(NarrowTy, Phi->getNumIncomingValues(), "", Phi);
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      BasicBlock *Pred = Phi->getIncomingBlock(i);
      Value *NewIn = nullptr;
      if (IsConst[i]) {
        auto *CI = cast<ConstantInt>(Sources[i]);
        NewIn = ConstantInt::get(NarrowTy, CI->getValue().trunc(NarrowBits));
      } else {
        // Sources[i] is the V before shl/ashr. Insert trunc to NarrowTy
        // at the end of the predecessor block (before its terminator).
        IRBuilder<> PB(Pred->getTerminator());
        Value *V = Sources[i];
        if (V->getType() == NarrowTy)
          NewIn = V;
        else if (V->getType()->isIntegerTy(128))
          NewIn = PB.CreateTrunc(V, NarrowTy);
        else {
          Bail = true;
          break;
        }
      }
      NewPhi->addIncoming(NewIn, Pred);
    }
    if (Bail) {
      NewPhi->eraseFromParent();
      continue;
    }
    // Replace uses of the old phi with sext(new_phi → i128).
    IRBuilder<> AfterPhi(Phi->getParent(), Phi->getParent()->getFirstInsertionPt());
    Value *Sexted = AfterPhi.CreateSExt(NewPhi, Phi->getType());
    Phi->replaceAllUsesWith(Sexted);
    Phi->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// fwd decl: shared low-W-bit rebuilder (defined just below).
static llvm::Value *narrowBitIntToWidth(llvm::Value *, unsigned, llvm::IRBuilder<> &);

// Iteratively erase trivially-dead instructions. The BitInt narrowing passes
// rebuild low-W-bit values natively, leaving the original wide (i128) producers
// dead; VXXPrep runs after the standard DCE, so sweep them here to match C++'s
// native-iN datapath in the emitted IR.
static bool eraseTriviallyDeadInsts(Function &F) {
  bool Changed = false, Any = true;
  while (Any) {
    Any = false;
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if (isInstructionTriviallyDead(I)) {
          I->eraseFromParent();
          Any = true;
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

// Narrow an in-loop sign-clamp `ashr exact (shl X, K), K` (= sext of the low
// W = BW-K bits) when X is a narrowable binop, by rebuilding the low W bits
// natively: `sext(narrowBitIntToWidth(X, W), BW)`. This frees the wide (i128)
// BitInt accumulator add that demoteBitIntBinopTrunc / demoteEscapingBitIntPhi
// left live because the per-iteration clamp still referenced it. X = PHI is
// left to demoteEscapingBitIntPhi.
static bool narrowBitIntSignClamp(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Ashr = dyn_cast<BinaryOperator>(&I);
      if (!Ashr || Ashr->getOpcode() != Instruction::AShr || !Ashr->isExact())
        continue;
      auto *Shl = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
      if (!Shl || Shl->getOpcode() != Instruction::Shl || !Shl->hasOneUse())
        continue;
      auto *K1 = dyn_cast<ConstantInt>(Ashr->getOperand(1));
      auto *K2 = dyn_cast<ConstantInt>(Shl->getOperand(1));
      if (!K1 || !K2 || K1->getValue() != K2->getValue())
        continue;
      auto *Ty = dyn_cast<IntegerType>(Ashr->getType());
      if (!Ty || Ty->getBitWidth() <= 64)
        continue;
      unsigned BW = Ty->getBitWidth();
      uint64_t K = K1->getZExtValue();
      if (K == 0 || K >= BW)
        continue;
      unsigned W = BW - (unsigned)K;
      if (W > 64)
        continue;
      Value *X = Shl->getOperand(0);
      if (!isa<BinaryOperator>(X))
        continue; // only narrowable binop producers (PHI → escaping pass)
      IRBuilder<> B(Ashr);
      Value *S = B.CreateSExt(narrowBitIntToWidth(X, W, B), Ty);
      Ashr->replaceAllUsesWith(S);
      Dead.push_back(Ashr);
      if (Shl->use_empty())
        Dead.push_back(Shl);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    if (I->use_empty())
      I->eraseFromParent();
  return Changed;
}

// Build a value equal to the low `W` bits of `V`, typed `iW`, *without*
// referencing V's full width when V is itself a narrowable binop or extend — so
// a wide (i128) producer of V can become dead afterwards. Recurses through
// add/sub/mul/and/or/xor (low W bits depend only on operands' low W bits) and
// sext/zext (the source carries the low bits). Falls back to `trunc V`.
static llvm::Value *narrowBitIntToWidth(llvm::Value *V, unsigned W,
                                        llvm::IRBuilder<> &B) {
  auto *NTy = llvm::IntegerType::get(V->getContext(), W);
  if (auto *Ext = llvm::dyn_cast<llvm::CastInst>(V)) {
    if (Ext->getOpcode() == llvm::Instruction::SExt ||
        Ext->getOpcode() == llvm::Instruction::ZExt) {
      llvm::Value *X = Ext->getOperand(0);
      if (auto *XTy = llvm::dyn_cast<llvm::IntegerType>(X->getType())) {
        unsigned Src = XTy->getBitWidth();
        if (Src == W)
          return X;
        if (Src < W)
          return Ext->getOpcode() == llvm::Instruction::SExt ? B.CreateSExt(X, NTy)
                                                             : B.CreateZExt(X, NTy);
        return B.CreateTrunc(X, NTy);
      }
    }
  }
  if (auto *Bin = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
    switch (Bin->getOpcode()) {
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::And:
      case llvm::Instruction::Or:
      case llvm::Instruction::Xor: {
        llvm::Value *La = narrowBitIntToWidth(Bin->getOperand(0), W, B);
        llvm::Value *Lb = narrowBitIntToWidth(Bin->getOperand(1), W, B);
        return B.CreateBinOp(Bin->getOpcode(), La, Lb);
      }
      default:
        break;
    }
  }
  return B.CreateTrunc(V, NTy);
}

// Demote `trunc_W( binop_i128( extA, extB ) )` to a native `iW` binop, matching
// C++ ap_int<W>'s arbitrary-precision datapath. Rust's BitInt<W> does the
// arithmetic in i128 then truncates, so reflow / HLS otherwise sees a 128-bit
// adder. Valid for add/sub/mul/and/or/xor: the low W bits of the result depend
// only on the low W bits of the operands. Unlike narrowMaskedBinopTrunc this
// needs no `and` mask and handles operands wider than W (truncated to W).
static bool demoteBitIntBinopTrunc(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Trunc = dyn_cast<TruncInst>(&I);
      if (!Trunc)
        continue;
      auto *DstIntTy = dyn_cast<IntegerType>(Trunc->getType());
      if (!DstIntTy)
        continue;
      unsigned C = DstIntTy->getBitWidth();
      auto *Bin = dyn_cast<BinaryOperator>(Trunc->getOperand(0));
      if (!Bin)
        continue;
      auto *SrcIntTy = dyn_cast<IntegerType>(Bin->getType());
      if (!SrcIntTy || SrcIntTy->getBitWidth() <= C)
        continue; // only a genuine narrowing
      switch (Bin->getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
          break;
        default:
          continue;
      }
      IRBuilder<> B(Trunc);
      auto MakeNarrow = [&](Value *V) -> Value * {
        if (auto *Ext = dyn_cast<CastInst>(V)) {
          if (Ext->getOpcode() == Instruction::SExt ||
              Ext->getOpcode() == Instruction::ZExt) {
            Value *X = Ext->getOperand(0);
            if (auto *XTy = dyn_cast<IntegerType>(X->getType())) {
              unsigned Src = XTy->getBitWidth();
              if (Src == C)
                return X;
              if (Src < C)
                return Ext->getOpcode() == Instruction::SExt
                           ? B.CreateSExt(X, DstIntTy)
                           : B.CreateZExt(X, DstIntTy);
              return B.CreateTrunc(X, DstIntTy); // Src > C: low C bits only
            }
          }
        }
        return B.CreateTrunc(V, DstIntTy);
      };
      Value *La = MakeNarrow(Bin->getOperand(0));
      Value *Lb = MakeNarrow(Bin->getOperand(1));
      // Drop nsw/nuw: they assert about the i128 op; BitInt<W> arithmetic wraps
      // mod 2^W, which is exactly what the truncation expresses.
      Value *NewBin = B.CreateBinOp(Bin->getOpcode(), La, Lb);
      Trunc->replaceAllUsesWith(NewBin);
      Dead.push_back(Trunc);
      if (Bin->use_empty())
        Dead.push_back(Bin);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    I->eraseFromParent();
  return Changed;
}

// Demote a loop-exit (lcssa-style) wide integer PHI whose *only* uses are
// sign-clamps `ashr exact (shl P, K), K` (= sign-extend the low W = BW-K bits).
// This is the BitInt<W> loop accumulator escaping the loop: demoteBitIntBinopTrunc
// already narrowed the in-loop `trunc(add)` to an iW add, but the i128 add stays
// live because its result flows out through this PHI. Narrowing the PHI to iW
// (incoming = trunc of each value to iW) and rewriting each clamp to
// `sext iW P' to BW` lets the wide add die, matching C++'s native-iN datapath.
static bool demoteEscapingBitIntPhi(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &P : BB.phis())
      Phis.push_back(&P);
    for (PHINode *P : Phis) {
      auto *PTy = dyn_cast<IntegerType>(P->getType());
      if (!PTy || PTy->getBitWidth() <= 64)
        continue; // only wide (i128) accumulators
      unsigned BW = PTy->getBitWidth();
      // Every use must be a clamp `ashr exact (shl P, K), K` with the same K.
      unsigned W = 0;
      bool ok = true;
      SmallVector<std::pair<BinaryOperator *, BinaryOperator *>, 4> Clamps;
      for (User *U : P->users()) {
        auto *Shl = dyn_cast<BinaryOperator>(U);
        if (!Shl || Shl->getOpcode() != Instruction::Shl || !Shl->hasOneUse()) {
          ok = false;
          break;
        }
        auto *ShlK = dyn_cast<ConstantInt>(Shl->getOperand(1));
        auto *Ashr = dyn_cast<BinaryOperator>(*Shl->user_begin());
        if (!ShlK || !Ashr || Ashr->getOpcode() != Instruction::AShr ||
            !Ashr->isExact()) {
          ok = false;
          break;
        }
        auto *AshrK = dyn_cast<ConstantInt>(Ashr->getOperand(1));
        if (!AshrK || AshrK->getValue() != ShlK->getValue()) {
          ok = false;
          break;
        }
        uint64_t K = ShlK->getZExtValue();
        if (K == 0 || K >= BW) {
          ok = false;
          break;
        }
        unsigned ThisW = BW - (unsigned)K;
        if (W == 0)
          W = ThisW;
        else if (W != ThisW) {
          ok = false;
          break;
        }
        Clamps.push_back({Shl, Ashr});
      }
      if (!ok || W == 0 || W >= BW || W > 64 || Clamps.empty())
        continue;
      IntegerType *NTy = IntegerType::get(F.getContext(), W);
      PHINode *NP = PHINode::Create(NTy, P->getNumIncomingValues(), "", P);
      for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
        Value *V = P->getIncomingValue(i);
        BasicBlock *Pred = P->getIncomingBlock(i);
        IRBuilder<> B(Pred->getTerminator());
        // narrowBitIntToWidth rebuilds the low W bits from a narrowable producer
        // (binop/ext) without referencing the wide V, so the wide accumulator add
        // becomes dead once this PHI no longer carries it.
        NP->addIncoming(narrowBitIntToWidth(V, W, B), Pred);
      }
      // Each clamp = sext(low W bits) → replace with sext(NP) to BW.
      for (auto &C : Clamps) {
        IRBuilder<> B(C.second);
        Value *S = B.CreateSExt(NP, PTy);
        C.second->replaceAllUsesWith(S);
        Dead.push_back(C.second); // ashr (uses shl)
        Dead.push_back(C.first);  // shl  (uses P)
      }
      Dead.push_back(P); // old wide PHI, now unused
      Changed = true;
    }
  }
  // Erase clamps (ashr before shl, already ordered), then any now-dead PHIs.
  for (Instruction *I : Dead)
    if (I->use_empty())
      I->eraseFromParent();
  return Changed;
}

static bool narrowMaskedBinopTrunc(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Trunc = dyn_cast<TruncInst>(&I);
      if (!Trunc)
        continue;
      auto *DstIntTy = dyn_cast<IntegerType>(Trunc->getType());
      if (!DstIntTy)
        continue;
      unsigned C = DstIntTy->getBitWidth();
      // Trunc operand must be `and i128 X, MASK` where MASK = (1<<C)-1.
      auto *And = dyn_cast<BinaryOperator>(Trunc->getOperand(0));
      if (!And || And->getOpcode() != Instruction::And)
        continue;
      auto *MaskC = dyn_cast<ConstantInt>(And->getOperand(1));
      if (!MaskC)
        continue;
      APInt Expected = APInt::getLowBitsSet(
          MaskC->getType()->getIntegerBitWidth(), C);
      if (MaskC->getValue() != Expected)
        continue;
      auto *Bin = dyn_cast<BinaryOperator>(And->getOperand(0));
      if (!Bin)
        continue;
      // Only commutative-or-associative arithmetic where masking the
      // result to C bits is the same as masking the inputs first.
      switch (Bin->getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::Or:
        case Instruction::And:
        case Instruction::Xor:
          break;
        default:
          continue;
      }
      // Both operands must be sext/zext from a width <= C.
      auto MakeNarrow = [&](Value *V, IRBuilder<> &B) -> Value * {
        if (auto *Ext = dyn_cast<CastInst>(V)) {
          if (Ext->getOpcode() == Instruction::SExt ||
              Ext->getOpcode() == Instruction::ZExt) {
            unsigned Src = Ext->getSrcTy()->getIntegerBitWidth();
            if (Src > C)
              return nullptr;
            if (Src == C)
              return Ext->getOperand(0);
            return Ext->getOpcode() == Instruction::SExt
                       ? B.CreateSExt(Ext->getOperand(0), DstIntTy)
                       : B.CreateZExt(Ext->getOperand(0), DstIntTy);
          }
        }
        return nullptr;
      };
      IRBuilder<> Bldr(Trunc);
      Value *La = MakeNarrow(Bin->getOperand(0), Bldr);
      Value *Lb = MakeNarrow(Bin->getOperand(1), Bldr);
      if (!La || !Lb)
        continue;
      Value *NewBin =
          Bldr.CreateBinOp(Bin->getOpcode(), La, Lb);
      if (auto *NewBinI = dyn_cast<BinaryOperator>(NewBin)) {
        if (Bin->hasNoSignedWrap())
          NewBinI->setHasNoSignedWrap();
        if (Bin->hasNoUnsignedWrap())
          NewBinI->setHasNoUnsignedWrap();
      }
      Trunc->replaceAllUsesWith(NewBin);
      Dead.push_back(Trunc);
      if (And->use_empty())
        Dead.push_back(And);
      if (Bin->use_empty())
        Dead.push_back(Bin);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    if (I->getParent())
      I->eraseFromParent();
  return Changed;
}

static bool narrowBitFixedMulShiftTrunc(Function &F) {
  bool Changed = false;
  SmallVector<Instruction *, 8> Dead;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Trunc = dyn_cast<TruncInst>(&I);
      if (!Trunc)
        continue;
      auto *Ashr = dyn_cast<BinaryOperator>(Trunc->getOperand(0));
      if (!Ashr || Ashr->getOpcode() != Instruction::AShr || !Ashr->isExact())
        continue;
      auto *AshrAmt = dyn_cast<ConstantInt>(Ashr->getOperand(1));
      if (!AshrAmt)
        continue;
      auto *Mul = dyn_cast<BinaryOperator>(Ashr->getOperand(0));
      if (!Mul || Mul->getOpcode() != Instruction::Mul)
        continue;
      auto *ShlA = dyn_cast<BinaryOperator>(Mul->getOperand(0));
      auto *ShlB = dyn_cast<BinaryOperator>(Mul->getOperand(1));
      if (!ShlA || ShlA->getOpcode() != Instruction::Shl ||
          !ShlB || ShlB->getOpcode() != Instruction::Shl)
        continue;
      auto *K1C = dyn_cast<ConstantInt>(ShlA->getOperand(1));
      auto *K2C = dyn_cast<ConstantInt>(ShlB->getOperand(1));
      if (!K1C || !K2C)
        continue;
      // Both operands must come from sext/zext iN → i128.
      auto *ExtA = dyn_cast<CastInst>(ShlA->getOperand(0));
      auto *ExtB = dyn_cast<CastInst>(ShlB->getOperand(0));
      if (!ExtA || !ExtB)
        continue;
      if ((ExtA->getOpcode() != Instruction::SExt &&
           ExtA->getOpcode() != Instruction::ZExt) ||
          (ExtB->getOpcode() != Instruction::SExt &&
           ExtB->getOpcode() != Instruction::ZExt))
        continue;
      auto *DstIntTy = dyn_cast<IntegerType>(Trunc->getType());
      if (!DstIntTy)
        continue;
      unsigned C = DstIntTy->getBitWidth();
      unsigned A = ExtA->getSrcTy()->getIntegerBitWidth();
      unsigned B = ExtB->getSrcTy()->getIntegerBitWidth();
      uint64_t K1 = K1C->getZExtValue();
      uint64_t K2 = K2C->getZExtValue();
      uint64_t K3 = AshrAmt->getZExtValue();
      // Effective post-shift width: A+K1+B+K2-K3 bits of the value
      // survive the truncation. Need that to fit in C.
      if ((uint64_t)A + (uint64_t)B + K1 + K2 < K3)
        continue;
      uint64_t NetShift = (uint64_t)A + (uint64_t)B + K1 + K2 - K3;
      if (NetShift > C)
        continue;
      // We require K1 + K2 >= K3 (the simpler case). Otherwise we'd
      // need an extra ashr after the mul, which Vitis doesn't emit.
      if (K1 + K2 < K3)
        continue;
      uint64_t LeftoverShift = (K1 + K2) - K3;
      IRBuilder<> Bldr(Trunc);
      // Re-extend operands to iC.
      Value *Aa = ExtA->getOpcode() == Instruction::SExt
                      ? Bldr.CreateSExt(ExtA->getOperand(0), DstIntTy)
                      : Bldr.CreateZExt(ExtA->getOperand(0), DstIntTy);
      Value *Bb = ExtB->getOpcode() == Instruction::SExt
                      ? Bldr.CreateSExt(ExtB->getOperand(0), DstIntTy)
                      : Bldr.CreateZExt(ExtB->getOperand(0), DstIntTy);
      Value *NewMul = Bldr.CreateMul(Aa, Bb, "", /*HasNUW=*/false,
                                     /*HasNSW=*/Mul->hasNoSignedWrap());
      Value *NewVal = NewMul;
      if (LeftoverShift)
        NewVal =
            Bldr.CreateShl(NewMul, ConstantInt::get(DstIntTy, LeftoverShift));
      Trunc->replaceAllUsesWith(NewVal);
      Dead.push_back(Trunc);
      if (Ashr->use_empty())
        Dead.push_back(Ashr);
      if (Mul->use_empty())
        Dead.push_back(Mul);
      if (ShlA->use_empty())
        Dead.push_back(ShlA);
      if (ShlB->use_empty())
        Dead.push_back(ShlB);
      Changed = true;
    }
  }
  for (Instruction *I : Dead)
    if (I->getParent())
      I->eraseFromParent();
  return Changed;
}

struct NarrowParamInfo {
  unsigned Idx;
  unsigned Width;
  bool Signed;
};
struct NarrowReturnInfo {
  unsigned Width;
  bool Signed;
};

static void narrowArrayPointerUses(Argument *OldArg, Argument *NewArg,
                                   bool Signed) {
  DenseMap<Value *, Value *> ValMap;
  ValMap[OldArg] = NewArg;

  SmallVector<Instruction *, 16> Worklist;
  for (User *U : OldArg->users())
    if (auto *I = dyn_cast<Instruction>(U))
      Worklist.push_back(I);

  SmallVector<Instruction *, 16> Dead;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      Value *NewPtr = ValMap.lookup(GEP->getPointerOperand());
      if (!NewPtr)
        continue;
      Type *NewSrcElemTy =
          cast<PointerType>(NewPtr->getType())->getElementType();
      SmallVector<Value *, 4> Idxs(GEP->idx_begin(), GEP->idx_end());
      auto *NewGEP = GetElementPtrInst::Create(
          NewSrcElemTy, NewPtr, Idxs, GEP->getName() + ".narrow", GEP);
      NewGEP->setIsInBounds(GEP->isInBounds());
      ValMap[GEP] = NewGEP;
      for (User *U : GEP->users())
        if (auto *UI = dyn_cast<Instruction>(U))
          Worklist.push_back(UI);
      Dead.push_back(GEP);
    } else if (auto *LI = dyn_cast<LoadInst>(I)) {
      Value *NewPtr = ValMap.lookup(LI->getPointerOperand());
      if (!NewPtr)
        continue;
      Type *NewElTy = cast<PointerType>(NewPtr->getType())->getElementType();
      auto *NewLI = new LoadInst(NewElTy, NewPtr, LI->getName() + ".narrow",
                                 LI->isVolatile(), LI);
      IRBuilder<> LB(LI);
      Value *Ext = Signed ? LB.CreateSExt(NewLI, LI->getType(), "load.widen")
                          : LB.CreateZExt(NewLI, LI->getType(), "load.widen");
      LI->replaceAllUsesWith(Ext);
      Dead.push_back(LI);
    } else if (auto *SI = dyn_cast<StoreInst>(I)) {
      Value *NewPtr = ValMap.lookup(SI->getPointerOperand());
      if (!NewPtr)
        continue;
      Type *NarrowTy = cast<PointerType>(NewPtr->getType())->getElementType();
      IRBuilder<> SB(SI);
      Value *Trunc =
          SB.CreateTrunc(SI->getValueOperand(), NarrowTy, "store.narrow");
      new StoreInst(Trunc, NewPtr, SI->isVolatile(), SI);
      Dead.push_back(SI);
    }
  }
  // Dead is built def-before-use (a GEP is appended before the loads/stores
  // that reference it). Erase in reverse so users are destroyed before the
  // values they use -- otherwise eraseFromParent() trips the LLVM
  // "Uses remain when a value is destroyed" assertion.
  for (auto It = Dead.rbegin(), E = Dead.rend(); It != E; ++It)
    (*It)->eraseFromParent();
}

static void narrowPointerUses(Argument *OldArg, Argument *NewArg, bool Signed) {
  Type *NarrowTy =
      cast<PointerType>(NewArg->getType())->getElementType();
  SmallVector<Instruction *, 8> Dead;
  SmallVector<User *, 8> Users(OldArg->users());
  for (User *U : Users) {
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() != OldArg)
        continue;
      IRBuilder<> SB(SI);
      Value *Trunc =
          SB.CreateTrunc(SI->getValueOperand(), NarrowTy, "store.narrow");
      new StoreInst(Trunc, NewArg, SI->isVolatile(), SI);
      Dead.push_back(SI);
    } else if (auto *LI = dyn_cast<LoadInst>(U)) {
      if (LI->getPointerOperand() != OldArg)
        continue;
      IRBuilder<> LB(LI);
      LoadInst *NewLI = new LoadInst(NarrowTy, NewArg, LI->getName(),
                                     LI->isVolatile(), LI);
      Value *Ext = Signed ? LB.CreateSExt(NewLI, LI->getType(), "load.widen")
                          : LB.CreateZExt(NewLI, LI->getType(), "load.widen");
      LI->replaceAllUsesWith(Ext);
      Dead.push_back(LI);
    }
  }
  for (Instruction *I : Dead)
    I->eraseFromParent();

  // For any remaining uses of OldArg (e.g. bitcast to `i8*` for the
  // AxisStreamView constructor, marker-call operands like
  // `__vxx_axis_disagg(p_data, …)`), insert a bitcast of NewArg back
  // to OldArg's pointer type at each use's insertion point. AxisStreamView
  // then sees a `*mut u8` view over the i4/i1 storage; volatile loads
  // through that pointer read 8 bits but cosim's csim phase runs the
  // C++ source (not our IR), so the host-side garbage is irrelevant.
  if (!OldArg->use_empty()) {
    // Old function's body has been spliced into NewF (see
    // narrowOneTopSignature). NewArg is the corresponding parameter of
    // NewF — insert the cast at NewF's entry so it dominates all remaining
    // uses (which are all inside NewF after the splice).
    Function *F = NewArg->getParent();
    BasicBlock &Entry = F->getEntryBlock();
    Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
    IRBuilder<> B(InsertBefore);
    Value *Cast = B.CreateBitCast(NewArg, OldArg->getType(),
                                   OldArg->getName() + ".oldcast");
    OldArg->replaceAllUsesWith(Cast);
  }
}

static Function *narrowOneTopSignature(Function *Old,
                                        ArrayRef<NarrowParamInfo> Params,
                                        Optional<NarrowReturnInfo> Ret) {
  LLVMContext &Ctx = Old->getContext();
  FunctionType *OldFT = Old->getFunctionType();
  unsigned NParams = OldFT->getNumParams();
  SmallVector<bool, 8> IsNarrow(NParams, false);
  SmallVector<bool, 8> IsPointer(NParams, false);
  SmallVector<bool, 8> IsArray(NParams, false);
  SmallVector<bool, 8> Signed(NParams, false);
  SmallVector<unsigned, 8> Widths(NParams, 0);
  for (const NarrowParamInfo &P : Params) {
    if (P.Idx >= NParams)
      return nullptr;
    IsNarrow[P.Idx] = true;
    Signed[P.Idx] = P.Signed;
    Widths[P.Idx] = P.Width;
  }

  // Wrap by-value scalar params in named single-field structs to match
  // cosim TB stub C++ ABI (which passes ap_int<N> as `struct __cosim_sN__`
  // by value, sized to sizeof(ap_int<N>)). Bare iN scalar args mismatch
  // the cosim TB stub → SIGSEGV at CALL_C_DUT during cosim.
  SmallVector<bool, 8> IsScalarStruct(NParams, false);
  SmallVector<StructType *, 8> ParamStructTys(NParams, nullptr);
  SmallVector<Type *, 8> NewParamTys;
  for (unsigned i = 0; i < NParams; ++i) {
    Type *OldPT = OldFT->getParamType(i);
    if (!IsNarrow[i]) {
      NewParamTys.push_back(OldPT);
      continue;
    }
    Type *NarrowTy = Type::getIntNTy(Ctx, Widths[i]);
    if (OldPT->isPointerTy()) {
      Type *PtrElem = cast<PointerType>(OldPT)->getElementType();
      unsigned AS = OldPT->getPointerAddressSpace();
      if (auto *OldArrTy = dyn_cast<ArrayType>(PtrElem)) {
        IsArray[i] = true;
        Type *NewArrTy = ArrayType::get(NarrowTy, OldArrTy->getNumElements());
        NewParamTys.push_back(PointerType::get(NewArrTy, AS));
      } else {
        IsPointer[i] = true;
        NewParamTys.push_back(PointerType::get(NarrowTy, AS));
      }
    } else {
      // By-value scalar: wrap in named single-field struct.
      IsScalarStruct[i] = true;
      std::string Name =
          (Signed[i] ? "narrow.ApInt" : "narrow.ApUint") + std::to_string(Widths[i]);
      StructType *Wrapped = StructType::create(Ctx, {NarrowTy}, Name);
      ParamStructTys[i] = Wrapped;
      NewParamTys.push_back(Wrapped);
    }
  }

  Type *NewRetTy = OldFT->getReturnType();
  Type *NarrowRetIntTy = nullptr;
  StructType *RetStructTy = nullptr;
  if (Ret.hasValue()) {
    NarrowRetIntTy = Type::getIntNTy(Ctx, Ret->Width);
    // Wrap return in a NAMED single-field struct so the bc kernel sig
    // matches the cosim TB's `_hw_stub` C++ ABI (which uses `struct __cosim_sN__*`
    // for ap_int<N> by-value return). Anonymous `{ iN }` triggers rustc
    // codegen SIGSEGV — named structs work (rustc itself uses `%ApIntN`).
    std::string Name =
        (Ret->Signed ? "narrow.ApInt" : "narrow.ApUint") + std::to_string(Ret->Width);
    RetStructTy = StructType::create(Ctx, {NarrowRetIntTy}, Name);
    NewRetTy = RetStructTy;
  }

  FunctionType *NewFT =
      FunctionType::get(NewRetTy, NewParamTys, OldFT->isVarArg());
  Function *NewF = Function::Create(NewFT, Old->getLinkage(),
                                    Old->getName() + ".narrow",
                                    Old->getParent());
  NewF->copyAttributesFrom(Old);
  NewF->getBasicBlockList().splice(NewF->end(), Old->getBasicBlockList());

  BasicBlock &Entry = NewF->getEntryBlock();
  Instruction *InsertBefore = &*Entry.getFirstInsertionPt();
  IRBuilder<> B(InsertBefore);
  auto OldArg = Old->arg_begin();
  auto NewArg = NewF->arg_begin();
  for (unsigned i = 0; i < NParams; ++i, ++OldArg, ++NewArg) {
    NewArg->takeName(&*OldArg);
    if (IsNarrow[i]) {
      NewArg->removeAttr(Attribute::ZExt);
      NewArg->removeAttr(Attribute::SExt);
      NewArg->removeAttr(Attribute::Alignment);
      NewArg->removeAttr(Attribute::Dereferenceable);
      NewArg->removeAttr(Attribute::DereferenceableOrNull);
    }
    if (IsArray[i]) {
      narrowArrayPointerUses(&*OldArg, &*NewArg, Signed[i]);
    } else if (IsPointer[i]) {
      narrowPointerUses(&*OldArg, &*NewArg, Signed[i]);
    } else if (IsScalarStruct[i]) {
      // Extract the iN field then widen back to the OldArg width.
      Value *Inner = B.CreateExtractValue(&*NewArg, {0u}, "extract");
      Value *Replacement =
          Signed[i] ? B.CreateSExt(Inner, OldArg->getType(), "widen")
                    : B.CreateZExt(Inner, OldArg->getType(), "widen");
      OldArg->replaceAllUsesWith(Replacement);
    } else if (IsNarrow[i]) {
      Value *Replacement =
          Signed[i] ? B.CreateSExt(&*NewArg, OldArg->getType(), "widen")
                    : B.CreateZExt(&*NewArg, OldArg->getType(), "widen");
      OldArg->replaceAllUsesWith(Replacement);
    } else {
      OldArg->replaceAllUsesWith(&*NewArg);
    }
  }

  if (Ret.hasValue()) {
    for (BasicBlock &BB : *NewF) {
      auto *RI = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!RI || !RI->getReturnValue())
        continue;
      IRBuilder<> RB(RI);
      Value *Narrowed =
          RB.CreateTrunc(RI->getReturnValue(), NarrowRetIntTy, "ret.narrow");
      // Wrap into the single-field struct.
      Value *Wrapped =
          RB.CreateInsertValue(UndefValue::get(RetStructTy), Narrowed, {0u},
                               "ret.wrap");
      ReturnInst::Create(Ctx, Wrapped, RI);
      RI->eraseFromParent();
    }
  }

  std::string Name = Old->getName().str();
  // The old kernel may still be referenced (e.g. by @llvm.used /
  // @llvm.compiler.used or other constant uses) even though it has no direct
  // callers. Erasing it with uses remaining trips the LLVM
  // "Uses remain when a value is destroyed" assertion (and leaves a dangling
  // reference in non-assert builds). Replace those uses with NewF bitcast to
  // the old function type (the canonical "swap a function for one of a
  // different type" idiom) before erasing.
  // NOT hlsrs::vxx::replaceFunctionKeepingName: that helper bare-erases, but here we
  // must replaceAllUsesWith(bitcast) first (Old still has uses).
  if (!Old->use_empty())
    Old->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, Old->getType()));
  Old->eraseFromParent();
  NewF->setName(Name);
  return NewF;
}

static bool narrowBitInts(Module &M) {
  Function *ParamMarker = M.getFunction("__vxx_top_param");
  Function *ReturnMarker = M.getFunction("__vxx_top_return");
  if (!ParamMarker && !ReturnMarker)
    return false;

  DenseMap<Function *, SmallVector<NarrowParamInfo, 4>> Params;
  DenseMap<Function *, NarrowReturnInfo> Returns;
  SmallVector<CallInst *, 16> AllCalls;

  if (ParamMarker) {
    for (User *U : ParamMarker->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      auto *CIdx = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      auto *CWidth = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      auto *CSign = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      if (!CIdx || !CWidth || !CSign) {
        AllCalls.push_back(CI);
        continue;
      }
      NarrowParamInfo P{(unsigned)CIdx->getZExtValue(),
                        (unsigned)CWidth->getZExtValue(),
                        CSign->getZExtValue() != 0};
      if (P.Width >= 1 && P.Width <= 128)
        Params[CI->getParent()->getParent()].push_back(P);
      AllCalls.push_back(CI);
    }
  }
  if (ReturnMarker) {
    for (User *U : ReturnMarker->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      auto *CWidth = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      auto *CSign = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (!CWidth || !CSign) {
        AllCalls.push_back(CI);
        continue;
      }
      NarrowReturnInfo R{(unsigned)CWidth->getZExtValue(),
                         CSign->getZExtValue() != 0};
      if (R.Width >= 1 && R.Width <= 128)
        Returns[CI->getParent()->getParent()] = R;
      AllCalls.push_back(CI);
    }
  }

  for (CallInst *CI : AllCalls)
    CI->eraseFromParent();

  SmallPtrSet<Function *, 8> Targets;
  for (auto &KV : Params)
    Targets.insert(KV.first);
  for (auto &KV : Returns)
    Targets.insert(KV.first);

  bool Changed = false;
  for (Function *Old : Targets) {
    SmallVector<NarrowParamInfo, 4> Empty;
    const SmallVectorImpl<NarrowParamInfo> &Ps =
        Params.count(Old) ? Params[Old] : Empty;
    Optional<NarrowReturnInfo> Ret;
    auto RIt = Returns.find(Old);
    if (RIt != Returns.end())
      Ret = RIt->second;
    if (narrowOneTopSignature(Old, Ps, Ret))
      Changed = true;
  }
  return Changed;
}

static bool elideDeadUninitMemcpy(Module &M) {
  bool Changed = false;
  auto baseAlloca = [](Value *V) -> AllocaInst * {
    while (true) {
      if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
      if (auto *G = dyn_cast<GetElementPtrInst>(V)) { V = G->getPointerOperand(); continue; }
      if (auto *BCC = dyn_cast<BitCastOperator>(V)) { V = BCC->getOperand(0); continue; }
      if (auto *GO = dyn_cast<GEPOperator>(V)) { V = GO->getPointerOperand(); continue; }
      break;
    }
    return dyn_cast<AllocaInst>(V);
  };
  auto neverStored = [&](AllocaInst *A) -> bool {
    SmallVector<Value *, 8> WL{A};
    SmallPtrSet<Value *, 8> Seen;
    while (!WL.empty()) {
      Value *V = WL.pop_back_val();
      for (User *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V) return false; // stored into
          continue;
        }
        if (auto *CI = dyn_cast<CallInst>(U)) {
          Function *F = CI->getCalledFunction();
          if (F && F->getName().startswith("llvm.lifetime")) continue;
          if (F && F->getName().startswith("llvm.memcpy")) {
            // dest (arg0) = written; src (arg1) = only read.
            if (CI->arg_size() >= 1 && CI->getArgOperand(0) == V) return false;
            continue;
          }
          return false; // passed to some call → assume written
        }
        if (isa<BitCastInst>(U) || isa<GetElementPtrInst>(U)) {
          if (Seen.insert(U).second) WL.push_back(U);
          continue;
        }
        if (isa<LoadInst>(U)) continue;
        return false; // unknown user → be conservative
      }
    }
    return true;
  };
  bool Again = true;
  unsigned Guard = 0;
  while (Again && Guard++ < 6) {
    Again = false;
    SmallVector<CallInst *, 8> Dead;
    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (Function *Cal = CI->getCalledFunction())
              if (Cal->getName().startswith("llvm.memcpy") && CI->arg_size() >= 2)
                if (AllocaInst *Src = baseAlloca(CI->getArgOperand(1)))
                  if (neverStored(Src))
                    Dead.push_back(CI);
    }
    for (CallInst *CI : Dead) { CI->eraseFromParent(); Changed = Again = true; }
  }
  if (Changed)
    vxxDbg() << "vxx: elided dead uninit memcpy (-O0 residual top loop)\n";
  return Changed;
}

static bool promoteKernelAllocas(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    legacy::FunctionPassManager FPM(&M);
    FPM.add(createSROAPass());
    FPM.add(createPromoteMemoryToRegisterPass());
    FPM.doInitialization();
    if (FPM.run(F)) Changed = true;
    FPM.doFinalization();
  }
  return Changed;
}

static bool foldExtractValue(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    bool Again = true;
    unsigned Guard = 0;
    while (Again && Guard++ < 8) {
      Again = false;
      // (0) fold `icmp pred (select c, ConstInt A, ConstInt B), ConstInt K`
      //     -> c / !c / const, collapsing the iterator's disc-switch onto the
      //     loop test (so `icmp ult (select v1,1,0),1` becomes `not v1`).
      {
        SmallVector<ICmpInst *, 16> IWL;
        for (BasicBlock &BB : F)
          for (Instruction &I : BB)
            if (auto *IC = dyn_cast<ICmpInst>(&I))
              IWL.push_back(IC);
        for (ICmpInst *IC : IWL) {
          auto *Sel = dyn_cast<SelectInst>(IC->getOperand(0));
          auto *K = dyn_cast<ConstantInt>(IC->getOperand(1));
          if (!Sel || !K) continue;
          auto *TA = dyn_cast<ConstantInt>(Sel->getTrueValue());
          auto *FB = dyn_cast<ConstantInt>(Sel->getFalseValue());
          if (!TA || !FB) continue;
          auto *RT = dyn_cast<ConstantInt>(
              ConstantExpr::getICmp(IC->getPredicate(), TA, K));
          auto *RF = dyn_cast<ConstantInt>(
              ConstantExpr::getICmp(IC->getPredicate(), FB, K));
          if (!RT || !RF) continue;
          bool bt = RT->isOne();
          bool bf = RF->isOne();
          Type *I1 = IC->getType();
          Value *C = Sel->getCondition();
          if (bt == bf) {
            IC->replaceAllUsesWith(ConstantInt::get(I1, bt));
            IC->eraseFromParent(); Changed = Again = true;
          } else if (bt && !bf) {
            IC->replaceAllUsesWith(C);
            IC->eraseFromParent(); Changed = Again = true;
          } else { // !bt && bf -> result is `not C`
            // If every user is a conditional branch on this icmp, swap each
            // branch's successors and use C directly — avoids a residual `xor`
            // on the loop-exit test (clang emits a plain icmp branch; the xor
            // form perturbs Vitis loop/pipeline analysis of the merged loop).
            SmallVector<BranchInst *, 4> Brs;
            bool AllBr = true;
            for (User *U : IC->users()) {
              auto *BI = dyn_cast<BranchInst>(U);
              if (BI && BI->isConditional() && BI->getCondition() == IC)
                Brs.push_back(BI);
              else { AllBr = false; break; }
            }
            if (AllBr && !Brs.empty()) {
              for (BranchInst *BI : Brs) {
                BI->setCondition(C);
                BI->swapSuccessors();
              }
              IC->eraseFromParent(); Changed = Again = true;
            } else {
              IRBuilder<> B(IC);
              Value *NotC = B.CreateXor(C, ConstantInt::get(I1, 1));
              IC->replaceAllUsesWith(NotC);
              IC->eraseFromParent(); Changed = Again = true;
            }
          }
        }
      }
      SmallVector<ExtractValueInst *, 32> WL;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *EV = dyn_cast<ExtractValueInst>(&I))
            WL.push_back(EV);
      for (ExtractValueInst *EV : WL) {
        Value *Agg = EV->getAggregateOperand();
        ArrayRef<unsigned> Idx = EV->getIndices();
        if (Idx.size() != 1)
          continue;
        // (1) extractvalue(insertvalue(A, V, Idx), Idx) -> V; disjoint -> base.
        if (auto *IV = dyn_cast<InsertValueInst>(Agg)) {
          if (IV->getNumIndices() == 1 && IV->getIndices()[0] == Idx[0]) {
            EV->replaceAllUsesWith(IV->getInsertedValueOperand());
            EV->eraseFromParent(); Changed = Again = true; continue;
          }
          if (IV->getNumIndices() == 1 && IV->getIndices()[0] != Idx[0]) {
            EV->setOperand(0, IV->getAggregateOperand());
            Changed = Again = true; continue;
          }
        }
        // (2) extractvalue(constant-agg, Idx) -> element.
        if (auto *C = dyn_cast<Constant>(Agg)) {
          if (Constant *E = C->getAggregateElement(Idx[0])) {
            EV->replaceAllUsesWith(E);
            EV->eraseFromParent(); Changed = Again = true; continue;
          }
        }
        // (2b) extractvalue(select(c, A, B), Idx) -> select(c, ev A, ev B)
        //      (SROA/if-conversion turns the iterator's some/none into a select).
        if (auto *SI = dyn_cast<SelectInst>(Agg)) {
          Value *TV = SI->getTrueValue(), *FV = SI->getFalseValue();
          bool TS = isa<InsertValueInst>(TV) || isa<Constant>(TV);
          bool FS = isa<InsertValueInst>(FV) || isa<Constant>(FV);
          if (TS && FS) {
            IRBuilder<> B(EV);
            Value *ET = isa<Constant>(TV)
                            ? cast<Constant>(TV)->getAggregateElement(Idx[0])
                            : B.CreateExtractValue(TV, Idx);
            Value *EF = isa<Constant>(FV)
                            ? cast<Constant>(FV)->getAggregateElement(Idx[0])
                            : B.CreateExtractValue(FV, Idx);
            if (ET && EF) {
              Value *NS = B.CreateSelect(SI->getCondition(), ET, EF);
              EV->replaceAllUsesWith(NS);
              EV->eraseFromParent(); Changed = Again = true; continue;
            }
          }
        }
        // (3) extractvalue(phi, Idx) -> phi(extractvalue) when every incoming is
        //     an insertvalue or constant (so the pushed EVs fold next round).
        if (auto *PN = dyn_cast<PHINode>(Agg)) {
          bool AllSimple = true;
          for (Value *In : PN->incoming_values())
            if (!isa<InsertValueInst>(In) && !isa<Constant>(In)) {
              AllSimple = false; break;
            }
          if (!AllSimple)
            continue;
          PHINode *NewPN = PHINode::Create(EV->getType(),
                                           PN->getNumIncomingValues(), "ev.phi",
                                           &PN->getParent()->front());
          for (unsigned k = 0; k < PN->getNumIncomingValues(); ++k) {
            BasicBlock *Pred = PN->getIncomingBlock(k);
            Value *In = PN->getIncomingValue(k);
            Value *E = nullptr;
            if (auto *Cst = dyn_cast<Constant>(In))
              E = Cst->getAggregateElement(Idx[0]);
            if (!E) {
              IRBuilder<> PB(Pred->getTerminator());
              E = PB.CreateExtractValue(In, Idx);
            }
            NewPN->addIncoming(E, Pred);
          }
          EV->replaceAllUsesWith(NewPN);
          EV->eraseFromParent(); Changed = Again = true; continue;
        }
      }
    }
  }
  return Changed;
}

static bool markCountedLoopIVNoWrap(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> Tops;
  if (Function *TM = M.getFunction("__vxx_top_kernel"))
    for (User *U : TM->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *F = CI->getFunction()) Tops.insert(F);
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!Tops.count(&F) && !F.hasFnAttribute("fpga.top.func")) continue;
    for (BasicBlock &BB : F) {
      for (PHINode &PN : BB.phis()) {
        if (!PN.getType()->isIntegerTy()) continue;
        // Find the backedge value that is `add PN, +C` (canonical step).
        for (Value *IV : PN.incoming_values()) {
          auto *Add = dyn_cast<BinaryOperator>(IV);
          if (!Add || Add->getOpcode() != Instruction::Add) continue;
          Value *Other = nullptr;
          if (Add->getOperand(0) == &PN) Other = Add->getOperand(1);
          else if (Add->getOperand(1) == &PN) Other = Add->getOperand(0);
          else continue;
          auto *C = dyn_cast<ConstantInt>(Other);
          if (!C || C->isNegative() || C->isZero()) continue;
          if (!Add->hasNoSignedWrap())   { Add->setHasNoSignedWrap(true);   Changed = true; }
          if (!Add->hasNoUnsignedWrap()) { Add->setHasNoUnsignedWrap(true); Changed = true; }
        }
      }
    }
  }
  if (Changed)
    vxxDbg() << "vxx: marked counted-loop IV no-wrap\n";
  return Changed;
}

static bool canonicalizeRangeLoopStep(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> Tops;
  if (Function *TM = M.getFunction("__vxx_top_kernel"))
    for (User *U : TM->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *F = CI->getFunction()) Tops.insert(F);
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!Tops.count(&F) && !F.hasFnAttribute("fpga.top.func")) continue;
    for (BasicBlock &BB : F) {
      auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
      if (!Br || !Br->isConditional()) continue;
      auto *Cond = dyn_cast<ICmpInst>(Br->getCondition());
      if (!Cond) continue;
      BasicBlock *TrueSucc = Br->getSuccessor(0);
      for (Instruction &I : BB) {
        auto *Add = dyn_cast<BinaryOperator>(&I);
        if (!Add || Add->getOpcode() != Instruction::Add) continue;
        // Locate a `zext(X)` step operand (either side of the add).
        unsigned StepIdx = 2; ZExtInst *Z = nullptr;
        if ((Z = dyn_cast<ZExtInst>(Add->getOperand(1)))) StepIdx = 1;
        else if ((Z = dyn_cast<ZExtInst>(Add->getOperand(0)))) StepIdx = 0;
        if (StepIdx == 2) continue;
        Value *ZOp = Z->getOperand(0);
        bool AlwaysOne = false;
        if (ZOp == Cond) {
          // Form (a): step = zext(branch-cond). Safe to force +1 iff the
          // true-edge is the backedge carrying this add (so zext==1 every taken
          // iteration) AND every use is a loop-carried PHI (so the dead +0 exit
          // value is never observed). Provably semantics-preserving.
          bool SafeBackedge = false, AllPhiUses = true;
          for (User *U : Add->users()) {
            auto *PN = dyn_cast<PHINode>(U);
            if (!PN) { AllPhiUses = false; break; }
            if (PN->getParent() == TrueSucc &&
                PN->getIncomingValueForBlock(&BB) == Add)
              SafeBackedge = true;
          }
          AlwaysOne = SafeBackedge && AllPhiUses;
        } else if (auto *P = dyn_cast<PHINode>(ZOp)) {
          // Form (b): step = zext(%p) where %p is a self-loop look-ahead flag
          //   %p = phi i1 [true, PH], [Cond, BB]    (Cond = continue branch cond)
          // %p is ALWAYS true while the body executes (init true; backedge value
          // = Cond, which was true to re-enter), so zext(%p)==1 unconditionally —
          // replacing with 1 is exact (not just dead-value-safe). This is rustc's
          // saturating form for a trivial-body counted loop (stencil coeff fill).
          if (P->getParent() == &BB && P->getType()->isIntegerTy(1) &&
              P->getNumIncomingValues() == 2 && TrueSucc == &BB) {
            bool hasTrue = false, hasCondBack = false;
            for (unsigned i = 0; i < 2; ++i) {
              Value *Iv = P->getIncomingValue(i);
              if (auto *CI = dyn_cast<ConstantInt>(Iv)) { if (CI->isOne()) hasTrue = true; }
              else if (Iv == Cond && P->getIncomingBlock(i) == &BB) hasCondBack = true;
            }
            AlwaysOne = hasTrue && hasCondBack;
          }
        }
        if (!AlwaysOne) continue;
        Add->setOperand(StepIdx, ConstantInt::get(Add->getType(), 1));
        Add->setHasNoSignedWrap(true);
        Add->setHasNoUnsignedWrap(true);
        Changed = true;
      }
    }
  }
  if (Changed)
    vxxDbg() << "vxx: canonicalized rustc Range loop step to +1\n";
  return Changed;
}

static bool collapseRangeLookaheadPhi(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> Tops;
  if (Function *TM = M.getFunction("__vxx_top_kernel"))
    for (User *U : TM->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *F = CI->getFunction()) Tops.insert(F);
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!Tops.count(&F) && !F.hasFnAttribute("fpga.top.func")) continue;
    for (BasicBlock &BB : F) {
      // Find a canonical look-ahead IV `%la` in this header.
      for (PHINode &LA : BB.phis()) {
        if (LA.getNumIncomingValues() != 2 || !LA.getType()->isIntegerTy())
          continue;
        // Identify the (latch, step-add) pair for LA.
        BasicBlock *Latch = nullptr;
        BinaryOperator *Step = nullptr;
        for (unsigned k = 0; k < 2; ++k) {
          auto *Add = dyn_cast<BinaryOperator>(LA.getIncomingValue(k));
          if (Add && Add->getOpcode() == Instruction::Add &&
              (Add->getOperand(0) == &LA || Add->getOperand(1) == &LA)) {
            Value *O = Add->getOperand(0) == &LA ? Add->getOperand(1)
                                                 : Add->getOperand(0);
            if (auto *C = dyn_cast<ConstantInt>(O))
              if (C->equalsInt(1)) { Step = Add; Latch = LA.getIncomingBlock(k); }
          }
        }
        if (!Step || !Latch) continue;
        // Find the dependent `%iter` PHI whose latch value is `%la` itself.
        for (PHINode &Iter : BB.phis()) {
          if (&Iter == &LA || Iter.getType() != LA.getType()) continue;
          int LatchIdx = Iter.getBasicBlockIndex(Latch);
          if (LatchIdx < 0) continue;
          if (Iter.getIncomingValue(LatchIdx) != &LA) continue;
          // %iter.cnext = add nsw nuw %iter, 1, inserted before the step-add so
          // it dominates the step's use below. Numerically %iter.cnext == %la
          // at the latch (both = iter+1).
          IRBuilder<> B(Step);
          Value *IterNext = B.CreateAdd(&Iter, ConstantInt::get(Iter.getType(), 1),
                                        Iter.getName() + ".cnext", true, true);
          // (1) %iter's backedge becomes self-contained {init,+,1}.
          Iter.setIncomingValue(LatchIdx, IterNext);
          // (2) Redirect the exit-counter step (`add %la, 1`) to read %iter.cnext
          //     instead of %la. Now nothing reads %la except its own (dead) phi
          //     backedge + leftover dead icmp → IndVarSimplify/DCE deletes the
          //     redundant look-ahead PHI, leaving one canonical IV like C++.
          Step->replaceUsesOfWith(&LA, IterNext);
          Changed = true;
        }
      }
    }
  }
  if (Changed)
    vxxDbg() << "vxx: decoupled rustc look-ahead dual-counter PHIs\n";
  return Changed;
}

static bool unrotateCountedLoops(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  SmallPtrSet<Function *, 4> StencilFns;
  if (Function *ASten = M.getFunction("__vxx_array_stencil"))
    for (User *U : ASten->users())
      if (auto *SCI = dyn_cast<CallInst>(U))
        if (Function *F = SCI->getFunction()) StencilFns.insert(F);
  if (StencilFns.empty()) return false;

  for (Function *F : StencilFns) {
    // Snapshot latches first (we mutate the CFG).
    SmallVector<BasicBlock *, 8> Latches;
    for (BasicBlock &BB : *F) Latches.push_back(&BB);
    for (BasicBlock *L : Latches) {
      auto *Br = dyn_cast<BranchInst>(L->getTerminator());
      if (!Br || !Br->isConditional()) continue;
      auto *EC = dyn_cast<ICmpInst>(Br->getCondition());
      if (!EC) continue;
      // Continue-condition `icmp <ne|ult|slt> (add iv,1), N` (true → backedge).
      // rustc emits NE for some loops, ULT for the trivial coeff-fill inner loop.
      // For all three the body runs iv=0..N-1, so the un-rotated entry test is
      // `icmp ult iv, N` regardless.
      { auto P = EC->getPredicate();
        if (P != ICmpInst::ICMP_NE && P != ICmpInst::ICMP_ULT &&
            P != ICmpInst::ICMP_SLT) continue; }
      // ec = icmp <pred> (add iv,1), N
      Value *Op0 = EC->getOperand(0), *Op1 = EC->getOperand(1);
      auto *N = dyn_cast<ConstantInt>(Op1);
      auto *Add = dyn_cast<BinaryOperator>(Op0);
      if (!N) { N = dyn_cast<ConstantInt>(Op0); Add = dyn_cast<BinaryOperator>(Op1); }
      if (!N || !Add || Add->getOpcode() != Instruction::Add || Add->getParent() != L)
        continue;
      ConstantInt *One = dyn_cast<ConstantInt>(Add->getOperand(1));
      auto *IV = dyn_cast<PHINode>(Add->getOperand(0));
      if (!One || !One->isOne() || !IV) continue;
      BasicBlock *H = IV->getParent();
      // H must be one of Br's successors (the continue edge) and IV's backedge
      // value (from L) must be this Add.
      if (Br->getSuccessor(0) != H && Br->getSuccessor(1) != H) continue;
      BasicBlock *E = Br->getSuccessor(0) == H ? Br->getSuccessor(1)
                                               : Br->getSuccessor(0);
      if (IV->getBasicBlockIndex(L) < 0) continue;
      if (IV->getIncomingValueForBlock(L) != Add) continue;
      int PHIdx = -1;
      for (unsigned i = 0; i < IV->getNumIncomingValues(); ++i)
        if (IV->getIncomingBlock(i) != L) { PHIdx = i; break; }
      if (PHIdx < 0) continue;
      BasicBlock *PH = IV->getIncomingBlock(PHIdx);
      // PH must branch unconditionally to H (clean preheader).
      auto *PHBr = dyn_cast<BranchInst>(PH->getTerminator());
      if (!PHBr || PHBr->isConditional() || PHBr->getSuccessor(0) != H) continue;

      // --- Build the Cond block ---
      BasicBlock *Cond = BasicBlock::Create(Ctx, H->getName() + ".unrot", F, H);
      IRBuilder<> CB(Cond);
      // Move every PHI of H into Cond (same incomings: PH + L predecessors).
      SmallVector<PHINode *, 8> HdrPhis;
      for (PHINode &P : H->phis()) HdrPhis.push_back(&P);
      PHINode *NewIV = nullptr;
      for (PHINode *P : HdrPhis) {
        PHINode *NP = CB.CreatePHI(P->getType(), P->getNumIncomingValues(),
                                   P->getName());
        for (unsigned i = 0; i < P->getNumIncomingValues(); ++i)
          NP->addIncoming(P->getIncomingValue(i), P->getIncomingBlock(i));
        P->replaceAllUsesWith(NP);
        if (P == IV) NewIV = NP;
      }
      for (PHINode *P : HdrPhis) P->eraseFromParent();
      // Entry test `icmp ult %iv, N` (iv runs 0..N-1) + br to body H or exit E.
      Value *Ent = CB.CreateICmpULT(NewIV, N, "unrot.ec");
      CB.CreateCondBr(Ent, H, E);
      // PH → Cond
      PHBr->setSuccessor(0, Cond);
      // L: unconditional → Cond; carry the loop's !llvm.loop metadata
      // (pipeline/name/tripcount) onto the new backedge branch; drop the dead
      // exit icmp.
      MDNode *LoopMD = Br->getMetadata("llvm.loop");
      BranchInst *NewLatchBr = BranchInst::Create(Cond, L);
      if (LoopMD) NewLatchBr->setMetadata("llvm.loop", LoopMD);
      Br->eraseFromParent();
      if (EC->use_empty()) EC->eraseFromParent();
      // E's PHIs: the in-loop edge now comes from Cond, carrying the header
      // (loop-carried) value. Map the L-incoming value to its Cond PHI.
      for (PHINode &EP : E->phis()) {
        int Idx = EP.getBasicBlockIndex(L);
        if (Idx < 0) continue;
        Value *V = EP.getIncomingValue(Idx);
        Value *Mapped = V;
        for (PHINode &CP : Cond->phis())
          if (CP.getIncomingValueForBlock(L) == V) { Mapped = &CP; break; }
        EP.setIncomingBlock(Idx, Cond);
        EP.setIncomingValue(Idx, Mapped);
      }
      Changed = true;
    }
  }
  if (Changed)
    vxxDbg() << "vxx: un-rotated stencil counted loops to clang for.cond\n";
  return Changed;
}

static bool simplifyKernelLoops(Module &M) {
  bool Changed = false;
  // Detect top kernel(s) EARLY via the __vxx_top_kernel marker: this pass runs
  // BEFORE injectKernelTopAttribute sets fpga.top.func, so gating on that
  // attribute alone makes it a silent no-op. The LoopRotate/JumpThreading/
  // CFGSimplify passes below
  // canonicalize the rustc loop CFG enough for the HLS backend to form
  // recognized loops and infer m_axi bursts; without them lmem_2rw exports an
  // EMPTY component.
  SmallPtrSet<Function *, 4> EarlyTops;
  if (Function *TopMarker = M.getFunction("__vxx_top_kernel"))
    for (User *U : TopMarker->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (Function *F = CI->getFunction()) EarlyTops.insert(F);
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!EarlyTops.count(&F) && !F.hasFnAttribute("fpga.top.func")) continue;
    legacy::FunctionPassManager FPM(&M);
    // LoopSimplify (dedicated preheader + single latch) and IndVarSimplify
    // (canonical induction variable) are what Clang's own pipeline produces;
    // rustc's loop output is NOT in that form, so the HLS backend can't make
    // the loops countable and exports an empty component. The minimal LoopRotate/
    // JumpThreading/CFGSimplify trio is insufficient — IndVarSimplify is the
    // load-bearing addition. This is canonicalization, not behaviour change.
    // SROA + mem2reg FIRST: at -O0 locals (incl. the synthesized Range struct
    // {start,end}) stay in allocas, so the loop bound is a `load` not a constant
    // -> Vitis sees a variable trip count (214-187) and can't unroll constant
    // window loops (stencil 214-332). Promoting them exposes the constant end
    // (e.g. 15) so the trip count is constant. No-op at -O1 (already SSA).
    FPM.add(createSROAPass());
    FPM.add(createPromoteMemoryToRegisterPass());
    FPM.add(createLoopSimplifyPass());
    FPM.add(createLoopRotatePass());
    FPM.add(createIndVarSimplifyPass());
    FPM.add(createJumpThreadingPass());
    FPM.add(createCFGSimplificationPass());
    FPM.add(createLoopSimplifyPass());
    FPM.doInitialization();
    if (FPM.run(F)) Changed = true;
    FPM.doFinalization();
  }
  if (Changed)
    vxxDbg() << "vxx: simplifyKernelLoops applied\n";
  return Changed;
}

static bool lowerWithOverflowIntrinsics(Module &M) {
  bool Changed = false;
  unsigned Lowered = 0;
  SmallVector<CallInst *, 8> Calls;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Cal = CI->getCalledFunction();
          if (!Cal) continue;
          Intrinsic::ID ID = Cal->getIntrinsicID();
          if (ID == Intrinsic::uadd_with_overflow ||
              ID == Intrinsic::sadd_with_overflow ||
              ID == Intrinsic::usub_with_overflow ||
              ID == Intrinsic::ssub_with_overflow ||
              ID == Intrinsic::umul_with_overflow ||
              ID == Intrinsic::smul_with_overflow)
            Calls.push_back(CI);
        }
  }
  for (CallInst *CI : Calls) {
    Function *Cal = CI->getCalledFunction();
    Intrinsic::ID ID = Cal->getIntrinsicID();
    IRBuilder<> B(CI);
    Value *A = CI->getArgOperand(0);
    Value *Bv = CI->getArgOperand(1);
    auto *IntT = cast<IntegerType>(A->getType());
    unsigned BW = IntT->getBitWidth();
    Value *Res = nullptr, *Ovf = nullptr;
    switch (ID) {
      case Intrinsic::uadd_with_overflow:
        Res = B.CreateAdd(A, Bv);
        Ovf = B.CreateICmpULT(Res, A);
        break;
      case Intrinsic::usub_with_overflow:
        Res = B.CreateSub(A, Bv);
        Ovf = B.CreateICmpULT(A, Bv);
        break;
      case Intrinsic::sadd_with_overflow: {
        Res = B.CreateAdd(A, Bv);
        Value *NotXorAB = B.CreateNot(B.CreateXor(A, Bv));
        Value *XorARes = B.CreateXor(A, Res);
        Value *Conj = B.CreateAnd(NotXorAB, XorARes);
        Value *SignBit = ConstantInt::get(IntT, APInt::getSignMask(BW));
        Ovf = B.CreateICmpNE(B.CreateAnd(Conj, SignBit),
                              ConstantInt::get(IntT, 0));
        break;
      }
      case Intrinsic::ssub_with_overflow: {
        Res = B.CreateSub(A, Bv);
        Value *XorAB = B.CreateXor(A, Bv);
        Value *XorARes = B.CreateXor(A, Res);
        Value *Conj = B.CreateAnd(XorAB, XorARes);
        Value *SignBit = ConstantInt::get(IntT, APInt::getSignMask(BW));
        Ovf = B.CreateICmpNE(B.CreateAnd(Conj, SignBit),
                              ConstantInt::get(IntT, 0));
        break;
      }
      case Intrinsic::umul_with_overflow: {
        IntegerType *WideT = IntegerType::get(M.getContext(), BW * 2);
        Value *WA = B.CreateZExt(A, WideT);
        Value *WB = B.CreateZExt(Bv, WideT);
        Value *WRes = B.CreateMul(WA, WB);
        Res = B.CreateTrunc(WRes, IntT);
        Value *Shifted = B.CreateLShr(WRes, ConstantInt::get(WideT, BW));
        Ovf = B.CreateICmpNE(Shifted, ConstantInt::get(WideT, 0));
        break;
      }
      case Intrinsic::smul_with_overflow: {
        IntegerType *WideT = IntegerType::get(M.getContext(), BW * 2);
        Value *WA = B.CreateSExt(A, WideT);
        Value *WB = B.CreateSExt(Bv, WideT);
        Value *WRes = B.CreateMul(WA, WB);
        Res = B.CreateTrunc(WRes, IntT);
        Value *SExtBack = B.CreateSExt(Res, WideT);
        Ovf = B.CreateICmpNE(WRes, SExtBack);
        break;
      }
      default:
        continue;
    }
    // Replace `extractvalue` users directly with Res/Ovf instead of rebuilding
    // an `insertvalue {Res,Ovf}` aggregate. At -O1 instcombine folds
    // extractvalue(insertvalue,N)->operand, but at -O0 (no instcombine) the
    // aggregate survives, leaving the index as extractvalue(insertvalue,0)
    // rather than a plain `add` — which the HLS backend's stencil access-pattern
    // recogniser cannot analyse (HLS 214-332). Direct replacement yields the
    // plain add/sub at any opt level (matches clang's `add`/`sub nsw`).
    auto *RetST = cast<StructType>(CI->getType());
    SmallVector<ExtractValueInst *, 4> EVs;
    bool HasOther = false;
    for (User *U : CI->users()) {
      auto *EV = dyn_cast<ExtractValueInst>(U);
      if (EV && EV->getNumIndices() == 1)
        EVs.push_back(EV);
      else
        HasOther = true;
    }
    for (ExtractValueInst *EV : EVs) {
      Value *Rep = (EV->getIndices()[0] == 0) ? Res : Ovf;
      EV->replaceAllUsesWith(Rep);
      EV->eraseFromParent();
    }
    if (HasOther) {
      Value *Agg = UndefValue::get(RetST);
      Agg = B.CreateInsertValue(Agg, Res, {0});
      Agg = B.CreateInsertValue(Agg, Ovf, {1});
      CI->replaceAllUsesWith(Agg);
    }
    CI->eraseFromParent();
    Lowered++;
    Changed = true;
  }
  if (Lowered)
    vxxDbg() << "vxx: lowered " << Lowered
           << " with.overflow intrinsic call(s)\n";
  return Changed;
}

// Classify whether ST is one of rustc's zero-size-array "sandwich" structs
// that HLS 214-309 rejects: the alternating `[ZSA, field, ZSA, ...]` layout
// `struct_llfields` emits (shapes a/b/c below), plus the documented 3-field
// AxisDisabled / Vector<T,N> / ConfigT variants. Reads only ST (no module
// state). Extracted from stripZeroSizeArrayStructFields for readability —
// this shape dispatch was the bulk of that function's cyclomatic complexity.
static bool isRustcZsaSandwichStruct(StructType *ST) {
  if (ST->isOpaque())
    return false;
  bool HasZSA = false;
  int NonZSACount = 0;
  for (Type *FT : ST->elements()) {
    bool IsZSA = false;
    if (auto *AT = dyn_cast<ArrayType>(FT))
      if (AT->getNumElements() == 0)
        IsZSA = true;
    if (IsZSA)
      HasZSA = true;
    else
      NonZSACount++;
  }
  // Only strip when the struct has multiple real fields. Single-field
  // wrappers like `Stream<T>` (`{ [0 x T], T, [0 x T] }`) survive
  // strip-correctness but mutating them confuses downstream rewriters
  // (rewriteStreamReads, etc.) that expect the [0 x T] padding shape.
  // The struct_ii_issue / nested-struct cases that hit HLS 214-309
  // are always 2+ field aggregates.
  //
  // In addition: rustc's `struct_llfields` always emits an alternating
  // `[ZSA, F, ZSA, F, ..., ZSA]` layout with `2k+1` total fields. We
  // recognise that exact shape — if a struct has ZSAs in any other
  // position (e.g., a hand-written struct with explicit `[0 x T]` for
  // some unrelated reason), skip it to avoid misinterpreting intent.
  bool IsRustcShape = false;
  auto isZSAFieldAt = [&](unsigned idx) {
    auto *AT = dyn_cast<ArrayType>(ST->getElementType(idx));
    return AT && AT->getNumElements() == 0;
  };
  auto isSmallI8PadAt = [&](unsigned idx) {
    // Trailing alignment pad like `[N x i8]` for small N (rustc's
    // structure-size rounding up to the alignment of the largest field).
    auto *AT = dyn_cast<ArrayType>(ST->getElementType(idx));
    return AT && AT->getNumElements() > 0 && AT->getNumElements() < 16 &&
           AT->getElementType()->isIntegerTy(8);
  };
  // Recognised shapes (kernel-module-only — see KernelMarker check in the
  // caller — so we trust all `[0 x ?]` here are rustc-generated pad markers,
  // not hand-rolled intent):
  //   (a) `[ZSA, F, ZSA, F, ..., ZSA]` — `2k+1` fields, strict alternation
  //   (b) `[ZSA, F, ZSA, F, ..., ZSA, F, <[N x i8] pad>]` — `2k+2` fields
  //   (c) general case: starts AND ends with ZSA-or-i8-pad, with real
  //       fields interleaved by ZSA, but allows one or more intermediate
  //       `[N x i8]` alignment pads (e.g. before a u16/u32 field in a
  //       mixed-alignment struct). NonZSACount counts real + i8-pad
  //       fields together; NF = sum of all field positions.
  auto isPadOrZSAAt = [&](unsigned idx) {
    return isZSAFieldAt(idx) || isSmallI8PadAt(idx);
  };
  // Single-field shape `[ZSA, i8, ZSA]` — Rust's `#[repr(C)]`
  // single-byte placeholder structs like `AxisDisabled`. Restrict to
  // structs explicitly named `AxisDisabled` so we don't strip
  // unrelated 1-byte types (`BitUint<8>`, etc.).
  if (HasZSA && NonZSACount == 1) {
    unsigned NF = ST->getNumElements();
    StringRef SN = ST->hasName() ? ST->getName() : StringRef();
    if (NF == 3 && isZSAFieldAt(0) && isZSAFieldAt(2) &&
        (SN.contains("AxisDisabled") ||
         SN == "struct.hls::axis_disabled_signal")) {
      IsRustcShape = true;
    }
    // Vector<T, N> shape: `{ [0 x T'], [N x T], [0 x T'] }` where the
    // middle real field is an array. Stream<T> is `{ [0 x T], T, [0 x T] }`
    // (scalar middle) and must NOT be stripped — downstream Stream
    // rewriters depend on the wrapper shape. We disambiguate by middle
    // being an ArrayType.
    if (NF == 3 && isZSAFieldAt(0) && isZSAFieldAt(2) &&
        isa<ArrayType>(ST->getElementType(1))) {
      IsRustcShape = true;
    }
    // ConfigT / StatusT pattern: `{ [0 x T], T_scalar, [0 x T] }` where
    // middle is a scalar AND the struct is NOT a Stream<T> wrapper.
    // Detection: name doesn't contain "Stream<" but DOES match Rust's
    // `{ZSA, scalar, ZSA}` ZSA-sandwich layout. These appear as
    // single-field newtype wrappers (Stream<ConfigT>'s inner T) and
    // HLS rejects the ZSA fields even on local allocas.
    if (NF == 3 && isZSAFieldAt(0) && isZSAFieldAt(2) &&
        !SN.contains("Stream<") && !SN.contains("class.hls::stream") &&
        (ST->getElementType(1)->isIntegerTy() ||
         ST->getElementType(1)->isFloatingPointTy() ||
         ST->getElementType(1)->isArrayTy() ||
         ST->getElementType(1)->isVectorTy())) {
      // Also covers `#[repr(transparent)] Vector<T, N>` = `{ [0 x i32],
      // [N x T], [0 x i32] }` (barista_hls::Vector, C++ hls::vector<T, N>):
      // the middle real field is an array/vector, not a scalar. Stripping the
      // ZSAs leaves `{ [N x T] }`, avoiding HLS 214-309 in sub-functions that
      // take the vector by value/ref (e.g. compute/store in using_vectors).
      IsRustcShape = true;
    }
    // (Stream<T> local-only strip attempted but causes llvm.lifetime
    // type mismatch — see git log. Need separate alloca-type-fixing
    // pass for safe Stream<T> stripping.)
  }
  if (HasZSA && NonZSACount >= 2) {
    unsigned NF = ST->getNumElements();
    if (NF == 2 * NonZSACount + 1) {
      IsRustcShape = true;
      for (unsigned i = 0; i < NF; ++i) {
        bool ExpectZSA = (i % 2 == 0);
        if (ExpectZSA != isZSAFieldAt(i)) {
          IsRustcShape = false;
          break;
        }
      }
    } else if (NF == 2 * NonZSACount - 1 && isSmallI8PadAt(NF - 1)) {
      IsRustcShape = true;
      for (unsigned i = 0; i < NF - 1; ++i) {
        bool ExpectZSA = (i % 2 == 0);
        if (ExpectZSA != isZSAFieldAt(i)) {
          IsRustcShape = false;
          break;
        }
      }
    } else {
      // Shape (c) general: even-idx must be ZSA-or-i8-pad, odd-idx must
      // be a real field. NF must be odd (starts and ends with ZSA/pad).
      if (NF % 2 == 1) {
        IsRustcShape = true;
        for (unsigned i = 0; i < NF; ++i) {
          bool IsEven = (i % 2 == 0);
          if (IsEven) {
            if (!isPadOrZSAAt(i)) { IsRustcShape = false; break; }
          } else {
            if (isPadOrZSAAt(i)) { IsRustcShape = false; break; }
          }
        }
      }
    }
  }
  return IsRustcShape;
}

bool stripZeroSizeArrayStructFields(Module &M) {
  // Run on any kernel module: detect by presence of `__vxx_top_*`
  // markers (kernel, m_axi, top_param etc) — the proc-macro emits
  // at least one for every `#[barista_hls::top]` function. This is
  // broader than just `__vxx_top_kernel` (which can be optimized away
  // when the function body is empty) and matches all kernel modules
  // while still excluding libcore monomorphisations.
  // When invoked from the cpp_proxy AXIS path (`__vxx_axis_packed`) the caller has
  // already established this is a kernel module, and by the phase it runs the
  // `__vxx_*` markers may already be consumed — so skip the marker probe.
  // cpp_proxy cosim builds (`__vxx_cpp_proxy` marker) synthesise the user kernel as a
  // SUB-function of the generated `rust_<top>`/cosim_top, so the `#[top]`
  // `__vxx_top_*` marker is absent (the kernel isn't the top) and the
  // marker-probe below misses it — but the module IS a kernel module and its
  // ZSA-sandwich structs (e.g. barista_hls::Vector) still trip HLS 214-309 in
  // the sub-functions. Treat it as a kernel unconditionally (same as PACKED).
  bool IsKernel = hlsrs::vxx::markerUsed(M, "__vxx_axis_packed") ||
                  hlsrs::vxx::markerUsed(M, "__vxx_cpp_proxy");
  for (Function &F : M) {
    if (IsKernel) break;
    if (F.getName().startswith("__vxx_top_") ||
        F.getName().startswith("__vxx_m_axi") ||
        F.getName().startswith("__vxx_ap_") ||
        F.getName().startswith("__vxx_axis") ||
        F.getName().startswith("__vxx_s_axilite")) {
      if (!F.user_empty()) { IsKernel = true; break; }
    }
  }
  if (!IsKernel) return false;

  SmallVector<StructType *, 8> Targets;
  for (StructType *ST : M.getIdentifiedStructTypes())
    if (isRustcZsaSandwichStruct(ST))
      Targets.push_back(ST);

  // Shape (d): AXIS Packet structs (= inner element of a `class.hls::stream<...>`
  // wrapper) with mixed real fields and `[1 x i8]` alignment pads — typical
  // for axis_user-style packets where disabled channels (1-byte placeholders)
  // sit next to a wider real channel (u16/u32). Rust's `#[repr(C)]` inserts
  // explicit `[1 x i8]` pad fields between mismatched-alignment fields, which
  // proprietary Vitis HLS clang's "Dump HBM driver helper code" pass crashes
  // on (segfault). Stripping the explicit pads lets natural struct alignment
  // recreate the same byte layout via implicit gaps, matching the C++
  // axis<>'s 7-field shape.
  //
  // Detection: walk all named structs starting with `class.hls::stream<` (the
  // post-rename AXIS stream wrapper). For each such 1-field wrapper, mark
  // its inner struct as a Packet target if it has at least one `[1 x i8]`
  // pad field that wasn't already caught by shapes (a-c).
  // Identify "AXIS Packet" structs: the inner element of any
  // `class.hls::stream<...>` / `barista_hls::Stream<...>` 1-field wrapper.
  // For these, the strip-mapping logic below will remove `[1 x i8]`
  // alignment pads in addition to ZSAs — natural alignment of the
  // surviving fields recreates the same byte layout via implicit gaps,
  // matching the C++ axis<>'s 7-field IR shape and avoiding the
  // proprietary HLS clang's "Dump HBM driver helper code" segfault.
  SmallPtrSet<StructType *, 4> PacketTargets;
  for (StructType *ST : M.getIdentifiedStructTypes()) {
    if (ST->isOpaque() || !ST->hasName()) continue;
    StringRef SN = ST->getName();
    if (!SN.startswith("class.hls::stream<") &&
        !SN.startswith("barista_hls::Stream<")) continue;
    if (ST->getNumElements() != 1) continue;
    auto *Inner = dyn_cast<StructType>(ST->getElementType(0));
    if (!Inner || Inner->isOpaque()) continue;
    PacketTargets.insert(Inner);
    // Make sure Inner is in Targets (may already be there via shapes a-c).
    bool AlreadyTarget = false;
    for (StructType *T : Targets) if (T == Inner) { AlreadyTarget = true; break; }
    if (!AlreadyTarget) Targets.push_back(Inner);
  }
  if (Targets.empty())
    return false;

  // Build oldIdx → newIdx mapping (-1 for stripped) per target struct.
  // For shape (d) AXIS Packet targets, also strip `[1 x i8]` pad fields
  // (natural alignment of the surviving fields recreates the same byte
  // layout via implicit gaps).
  DenseMap<StructType *, std::vector<int>> Mapping;
  for (StructType *ST : Targets) {
    bool IsPacket = PacketTargets.count(ST) != 0;
    SmallVector<Type *, 8> NewFields;
    std::vector<int> Map(ST->getNumElements(), -1);
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *FT = ST->getElementType(i);
      if (auto *AT = dyn_cast<ArrayType>(FT)) {
        if (AT->getNumElements() == 0)
          continue;
        if (IsPacket && AT->getNumElements() == 1 && AT->getElementType()->isIntegerTy(8))
          continue;
      }
      Map[i] = (int)NewFields.size();
      NewFields.push_back(FT);
    }
    Mapping[ST] = std::move(Map);
  }

  // Capture global initializers BEFORE mutating struct types. After
  // `setBody`, any pre-existing `ConstantStruct` initializing a global
  // becomes structurally-mismatched (operand count != new field count)
  // and LLVM may invalidate or re-interpret it lossily. Walk every
  // initializer recursively, recording (a) which globals/positions
  // reference our targets, (b) the surviving field operands. Then
  // mutate the structs, and finally rebuild new ConstantStructs from
  // the saved operand pointers (which remain valid since they're
  // separate Constant objects — ConstantInt, ConstantDataArray, etc.).
  struct CapturedStruct {
    StructType *Type;
    SmallVector<Constant *, 8> Ops; // already filtered (non-stripped only)
  };
  std::function<Constant *(Constant *)> captureConst;
  captureConst = [&](Constant *C) -> Constant * {
    if (!C)
      return C;
    if (auto *CS = dyn_cast<ConstantStruct>(C)) {
      auto *ST = CS->getType();
      auto It = Mapping.find(ST);
      if (It != Mapping.end()) {
        // Capture filtered operands in a placeholder; resolve to a real
        // ConstantStruct AFTER struct mutation. We use a tagged wrapper
        // by stashing the captured data in a side table keyed off the
        // Constant pointer.
        // For now, materialise immediately by walking the operands and
        // recursing (operands are still valid pre-mutation).
        SmallVector<Constant *, 8> NewOps;
        const auto &Map = It->second;
        for (unsigned i = 0; i < Map.size(); ++i) {
          if (Map[i] == -1)
            continue;
          Constant *Op = cast<Constant>(CS->getOperand(i));
          NewOps.push_back(captureConst(Op));
        }
        // Cache on the side: we'll need ST + NewOps to rebuild after mutation.
        // Use ConstantStruct::getAnon to create an ANON struct constant for
        // now (since the named struct is about to mutate); we'll fix it up
        // post-mutation by building the proper ConstantStruct(ST, ops).
        // Simpler: create a placeholder by using an anonymous struct
        // immediately, store its operands, and then below we'll reassemble
        // with the mutated ST.
        // Store as anon struct keyed by original CS for tracking.
        return ConstantStruct::getAnon(NewOps, ST->isPacked());
      }
      // Non-target struct: recurse into fields.
      SmallVector<Constant *, 8> NewOps;
      bool changed = false;
      for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
        Constant *Op = cast<Constant>(CS->getOperand(i));
        Constant *NewOp = captureConst(Op);
        NewOps.push_back(NewOp);
        if (NewOp != Op)
          changed = true;
      }
      if (!changed)
        return CS;
      return ConstantStruct::get(ST, NewOps);
    }
    if (auto *CA = dyn_cast<ConstantArray>(C)) {
      SmallVector<Constant *, 16> NewOps;
      bool changed = false;
      for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
        Constant *Op = cast<Constant>(CA->getOperand(i));
        Constant *NewOp = captureConst(Op);
        NewOps.push_back(NewOp);
        if (NewOp != Op)
          changed = true;
      }
      if (!changed)
        return CA;
      // Array element type may need to change if it referenced a target
      // struct. Compute the new element type from the rewritten ops.
      Type *NewElemTy = NewOps.empty() ? CA->getType()->getElementType()
                                       : NewOps[0]->getType();
      auto *NewArrTy = ArrayType::get(NewElemTy, NewOps.size());
      return ConstantArray::get(NewArrTy, NewOps);
    }
    return C;
  };

  // Pass 1: capture & rewrite all initializers (PRE-mutation).
  // The rewritten constants reference anonymous structs (since the named
  // structs aren't mutated yet); we'll fix the type names in pass 3.
  DenseMap<GlobalVariable *, Constant *> NewInits;
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasInitializer())
      continue;
    Constant *Init = G.getInitializer();
    Constant *NewInit = captureConst(Init);
    if (NewInit != Init)
      NewInits[&G] = NewInit;
  }

  // Pass 2 (LLVM-rule-compliant replacement). The original code mutated each
  // target struct's body in place via StructType::setBody — which is ILLEGAL
  // on a non-opaque struct (setBody asserts isOpaque()) and only "worked"
  // under NDEBUG. Instead, build a NEW struct per target with the surviving
  // fields and remap the whole module (signatures, instruction types, GEP
  // field indices, globals + initializers) old → new, so nothing references
  // the old type. `Mapping` already holds oldIdx → newIdx (-1 = stripped),
  // computed above (and `Pass 1`/`Pass 3`/the GEP-rewrite below are now dead
  // — remapStructsInModule subsumes them).
  {
    DenseMap<Type *, Type *> TypeMap;
    DenseMap<StructType *, std::vector<int>> FieldMaps;
    // Phase 1: create OPAQUE shells for every target and register them in
    // TypeMap first. This is the LLVM-rule-compliant pattern for cyclic /
    // nested type rewrites — a target struct may have a field whose type is
    // ANOTHER target struct (e.g. `%S` contains `%T`). We must not bake the
    // OLD nested type into the new struct's body, or GEPs through it would
    // index the stale layout (yielding `[0 x i32]` where `i32` is meant).
    for (StructType *ST : Targets) {
      std::string Nm = ST->hasName() ? ST->getName().str() : std::string();
      if (!Nm.empty())
        ST->setName(Nm + ".pad"); // free the name for the replacement
      StructType *NewST = StructType::create(M.getContext(), Nm); // opaque
      TypeMap[ST] = NewST;
      FieldMaps[ST] = Mapping[ST];
    }
    // Phase 2: now that every target is in TypeMap, set each new body with
    // surviving fields remapped so nested target references resolve to their
    // (new) replacements. setBody is legal here — the shells are opaque.
    for (StructType *ST : Targets) {
      const std::vector<int> &Map = Mapping[ST];
      SmallVector<Type *, 8> NewFields;
      for (unsigned i = 0; i < ST->getNumElements(); ++i)
        if (i < Map.size() && Map[i] != -1)
          NewFields.push_back(
              hlsrs::vxx::remapTypeRecursive(ST->getElementType(i), TypeMap, M.getContext()));
      cast<StructType>(TypeMap[ST])->setBody(NewFields, ST->isPacked());
    }
    hlsrs::vxx::remapStructsInModule(M, TypeMap, &FieldMaps);
    vxxDbg() << "VXXPrep: stripZeroSizeArrayStructFields: replaced "
           << TypeMap.size() << " struct type(s) via new-struct remap\n";
    return !TypeMap.empty();
  }

  // Pass 3: re-bless captured initializers' anonymous structs as the
  // (now-mutated) named structs by recursing once more. We can identify
  // anon-struct values where the field count matches a target struct's
  // new field count and the field types align — re-cast to the named ST.
  // Simpler approach: walk top-level init types; if global type was
  // [N x %ST] but new init is [N x {anon-struct}], coerce.
  std::function<Constant *(Constant *, Type *)> rebless;
  rebless = [&](Constant *C, Type *ExpectedTy) -> Constant * {
    if (!C || !ExpectedTy)
      return C;
    if (auto *CAZ = dyn_cast<ConstantAggregateZero>(C))
      return ConstantAggregateZero::get(ExpectedTy);
    if (auto *CS = dyn_cast<ConstantStruct>(C)) {
      if (auto *ExpectedST = dyn_cast<StructType>(ExpectedTy)) {
        if (CS->getType() != ExpectedST &&
            CS->getNumOperands() == ExpectedST->getNumElements()) {
          SmallVector<Constant *, 8> NewOps;
          for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
            Constant *Op = cast<Constant>(CS->getOperand(i));
            NewOps.push_back(rebless(Op, ExpectedST->getElementType(i)));
          }
          return ConstantStruct::get(ExpectedST, NewOps);
        }
      }
      // Same type or non-struct expected: recurse normally.
      auto *ST = CS->getType();
      SmallVector<Constant *, 8> NewOps;
      bool changed = false;
      for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
        Constant *Op = cast<Constant>(CS->getOperand(i));
        Constant *NewOp = rebless(Op, ST->getElementType(i));
        NewOps.push_back(NewOp);
        if (NewOp != Op)
          changed = true;
      }
      if (!changed)
        return CS;
      return ConstantStruct::get(ST, NewOps);
    }
    if (auto *CA = dyn_cast<ConstantArray>(C)) {
      auto *ExpectedAT = dyn_cast<ArrayType>(ExpectedTy);
      Type *ElemExpected = ExpectedAT ? ExpectedAT->getElementType()
                                      : CA->getType()->getElementType();
      SmallVector<Constant *, 16> NewOps;
      bool changed = false;
      for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
        Constant *Op = cast<Constant>(CA->getOperand(i));
        Constant *NewOp = rebless(Op, ElemExpected);
        NewOps.push_back(NewOp);
        if (NewOp != Op)
          changed = true;
      }
      if (!changed)
        return CA;
      auto *NewArrTy = ArrayType::get(ElemExpected, NewOps.size());
      return ConstantArray::get(NewArrTy, NewOps);
    }
    return C;
  };
  for (auto &KV : NewInits) {
    GlobalVariable *G = KV.first;
    Constant *NewInit = rebless(KV.second, G->getValueType());
    G->setInitializer(NewInit);
  }

  unsigned NumRewritten = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 32> ToRewrite;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          ToRewrite.push_back(GEP);

    for (GetElementPtrInst *GEP : ToRewrite) {
      Type *CurT = GEP->getSourceElementType();
      SmallVector<Value *, 8> NewIndices;
      bool Rewrote = false;
      bool Skip = false;

      for (unsigned k = 1; k < GEP->getNumOperands(); ++k) {
        Value *Idx = GEP->getOperand(k);

        // First index walks the source pointee — type doesn't change, no
        // struct field mapping applies (it's a pointer-walk, not a struct
        // selector). Keep the index as-is.
        if (k == 1) {
          NewIndices.push_back(Idx);
          continue;
        }

        if (auto *ST = dyn_cast<StructType>(CurT)) {
          auto It = Mapping.find(ST);
          if (It != Mapping.end()) {
            auto *CIdx = dyn_cast<ConstantInt>(Idx);
            if (!CIdx) {
              Skip = true;
              break;
            }
            int OldI = (int)CIdx->getZExtValue();
            const auto &Map = It->second;
            if (OldI < 0 || OldI >= (int)Map.size()) {
              Skip = true;
              break;
            }
            int NewI = Map[OldI];
            if (NewI == -1) {
              // GEP indexes into a stripped [0 x T] field. Find the next
              // real field — `&S.zsa_field` and `&S.next_field` share an
              // address since [0 x T] is zero-size.
              int NextReal = -1;
              for (int j = OldI + 1; j < (int)Map.size(); ++j)
                if (Map[j] != -1) {
                  NextReal = Map[j];
                  break;
                }
              if (NextReal == -1) {
                Skip = true;
                break;
              }
              NewIndices.push_back(
                  ConstantInt::get(Idx->getType(), (uint64_t)NextReal));
              Type *NextT = ST->getElementType((unsigned)NextReal);
              // The inner GEP index (if any) was the "into [0 x T]" degenerate
              // access. After strip it's only valid if the next real field is
              // an array/vector type — then it becomes a real element index.
              // Otherwise (scalar field) drop the inner index.
              bool NextIsAggregate =
                  isa<ArrayType>(NextT) || isa<VectorType>(NextT);
              if (!NextIsAggregate && k + 1 < GEP->getNumOperands())
                k++;
              CurT = NextT;
              Rewrote = true;
              continue;
            }
            NewIndices.push_back(
                ConstantInt::get(Idx->getType(), (uint64_t)NewI));
            if (NewI != OldI)
              Rewrote = true;
            CurT = ST->getElementType((unsigned)NewI);
            continue;
          }
        }

        // Non-target type: keep the index, advance through the type.
        NewIndices.push_back(Idx);
        if (auto *ST2 = dyn_cast<StructType>(CurT)) {
          auto *CIdx = dyn_cast<ConstantInt>(Idx);
          if (!CIdx) {
            Skip = true;
            break;
          }
          CurT = ST2->getElementType(CIdx->getZExtValue());
        } else if (auto *AT = dyn_cast<ArrayType>(CurT)) {
          CurT = AT->getElementType();
        } else if (auto *VT = dyn_cast<VectorType>(CurT)) {
          CurT = VT->getElementType();
        } else {
          Skip = true;
          break;
        }
      }

      if (Skip || !Rewrote)
        continue;

      IRBuilder<> B(GEP);
      Value *NewGEP = B.CreateGEP(GEP->getSourceElementType(),
                                  GEP->getPointerOperand(), NewIndices);
      if (auto *NewGEPI = dyn_cast<GetElementPtrInst>(NewGEP)) {
        if (GEP->isInBounds())
          NewGEPI->setIsInBounds(true);
      }
      NewGEP->takeName(GEP);
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
      NumRewritten++;
    }
  }
  if (NumRewritten || !Targets.empty()) {
    vxxDbg() << "vxx: stripZeroSizeArrayStructFields: stripped "
           << Targets.size() << " struct type(s), rewrote " << NumRewritten
           << " GEP(s)\n";
  }
  return true;
}

static bool dropAxisDisabledPacketFields(Module &M) {
  // Intact `stripZeroSizeArrayStructFields` handles most of this. Stub.
  (void)M; return false;
}

static bool stripAxisPacketTrailingPad(Module &M) {
  (void)M; return false;
}

static bool stripStreamWrapperPadding(Module &M) {
  // Intact `stripZeroSizeArrayStructFields` covers the wrapper-pad case.
  (void)M; return false;
}

static bool narrowBlackboxExtern(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();

  SmallVector<Function *, 4> Candidates;
  for (Function &F : M) {
    if (!F.isDeclaration() || F.isIntrinsic() || F.use_empty()) continue;
    StringRef Name = F.getName();
    if (Name.startswith("_ZN") || Name.startswith("_Z") ||
        Name.startswith("__vxx_") || Name.startswith("_ssdm_") ||
        Name.startswith("llvm.") || Name.startswith("fpga.") ||
        Name.startswith("_fpc_") ||
        Name.startswith("__rust") || Name.startswith("rust_"))
      continue;
    bool allCompat = true, hasI128 = false;
    for (auto &Arg : F.args()) {
      Type *T = Arg.getType();
      if (T->isIntegerTy(128)) { hasI128 = true; continue; }
      if (auto *PT = dyn_cast<PointerType>(T)) {
        if (PT->getElementType()->isIntegerTy(128)) { hasI128 = true; continue; }
      }
      allCompat = false; break;
    }
    if (allCompat && hasI128) Candidates.push_back(&F);
  }
  for (Function *F : Candidates) {
    unsigned NArgs = F->arg_size();
    SmallVector<unsigned, 16> ArgN(NArgs, 0);
    SmallVector<bool, 16>     ArgIsPtr(NArgs, false);
    SmallVector<bool, 16>     ArgIsSigned(NArgs, true);

    unsigned ix = 0;
    for (auto &Arg : F->args()) {
      ArgIsPtr[ix] = Arg.getType()->isPointerTy();
      ix++;
    }

    SmallVector<CallInst *, 4> Calls;
    bool ok = true;
    for (User *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI) { ok = false; break; }
      Calls.push_back(CI);
    }
    if (!ok || Calls.empty()) continue;

    // Two-pass inference: value args get N from sext/zext source. Ptr args
    // assume same N as the value args (BitInt<N> extern fns are typically
    // uniform across in/out — caller body doesn't trunc the load output
    // because it stays in i128 storage form). If no value args exist,
    // ptr arg N can't be inferred and we bail.
    unsigned UniformN = 0; bool UniformSigned = true;
    for (CallInst *CI : Calls) {
      for (unsigned i = 0; i < NArgs && ok; ++i) {
        if (ArgIsPtr[i]) continue;  // handled in second pass
        Value *V = CI->getArgOperand(i);
        unsigned w = 0; bool sgnd = true;
        if (auto *S = dyn_cast<SExtInst>(V)) { w = S->getSrcTy()->getIntegerBitWidth(); sgnd = true; }
        else if (auto *Z = dyn_cast<ZExtInst>(V)) { w = Z->getSrcTy()->getIntegerBitWidth(); sgnd = false; }
        else { ok = false; break; }
        if (ArgN[i] == 0) { ArgN[i] = w; ArgIsSigned[i] = sgnd; }
        else if (ArgN[i] != w || ArgIsSigned[i] != sgnd) { ok = false; break; }
        if (UniformN == 0) { UniformN = w; UniformSigned = sgnd; }
      }
      if (!ok) break;
    }
    if (!ok) continue;
    if (UniformN == 0) continue;  // no value args → can't infer ptr N
    // Second pass: fill ptr arg N with UniformN. Verify the ptr arg
    // is an alloca i128.
    for (CallInst *CI : Calls) {
      for (unsigned i = 0; i < NArgs && ok; ++i) {
        if (!ArgIsPtr[i]) continue;
        Value *V = CI->getArgOperand(i)->stripPointerCasts();
        auto *AI = dyn_cast<AllocaInst>(V);
        if (!AI || !AI->getAllocatedType()->isIntegerTy(128)) { ok = false; break; }
        ArgN[i] = UniformN;
        ArgIsSigned[i] = UniformSigned;
      }
      if (!ok) break;
    }
    if (!ok) continue;

    // Build new function type and declaration.
    // Value args use exact iN width. Ptr args use the power-of-2 storage
    // width (e.g. BitInt<10>* → i16*): an `ap_int<N>&` reference goes through
    // the underlying int16_t / int32_t storage pointer, not the i10 value
    // type. Without this, HLS rejects the blackbox bind (`add_files -blackbox`)
    // due to an rtl_model signature mismatch.
    SmallVector<Type *, 16> NewArgTys;
    SmallVector<unsigned, 16> ArgPtrN(NArgs, 0); // pow2 width for ptr args
    for (unsigned i = 0; i < NArgs; ++i) {
      if (ArgIsPtr[i]) {
        unsigned PtrN = llvm::PowerOf2Ceil(ArgN[i] < 8 ? 8 : ArgN[i]);
        ArgPtrN[i] = PtrN;
        NewArgTys.push_back(PointerType::get(Type::getIntNTy(Ctx, PtrN), 0));
      } else {
        NewArgTys.push_back(Type::getIntNTy(Ctx, ArgN[i]));
      }
    }
    FunctionType *NewFT = FunctionType::get(F->getReturnType(), NewArgTys, false);
    std::string OldName = F->getName().str();
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                       OldName + ".narrowed", &M);
    NewF->copyAttributesFrom(F);
    for (unsigned i = 0; i < NArgs; ++i) {
      NewF->removeParamAttr(i, Attribute::Alignment);
      NewF->removeParamAttr(i, Attribute::Dereferenceable);
      NewF->removeParamAttr(i, Attribute::DereferenceableOrNull);
      NewF->removeParamAttr(i, Attribute::NoAlias);
    }

    // Rewrite each call site.
    SmallVector<CallInst *, 4> DeadCalls;
    SmallVector<AllocaInst *, 16> DeadAllocas;
    for (CallInst *CI : Calls) {
      IRBuilder<> B(CI);
      SmallVector<Value *, 16> NewArgs;
      bool callOk = true;
      for (unsigned i = 0; i < NArgs && callOk; ++i) {
        if (ArgIsPtr[i]) {
          auto *OldAI = cast<AllocaInst>(CI->getArgOperand(i)->stripPointerCasts());
          // Create new alloca with pow2-storage width at OldAI's slot.
          unsigned PtrN = ArgPtrN[i];
          IRBuilder<> AB(OldAI);
          unsigned BytesAlign = (PtrN + 7) / 8;
          if (BytesAlign == 0) BytesAlign = 1;
          AllocaInst *NewAI = AB.CreateAlloca(Type::getIntNTy(Ctx, PtrN),
                                              nullptr, OldAI->getName());
          NewAI->setAlignment(Align(llvm::PowerOf2Ceil(BytesAlign)));
          // Rewrite OldAI uses except for the call CI itself.
          SmallVector<Use *, 8> UsesToRewrite;
          for (Use &U : OldAI->uses()) {
            if (U.getUser() == CI) continue;
            UsesToRewrite.push_back(&U);
          }
          for (Use *U : UsesToRewrite) {
            Instruction *UI = dyn_cast<Instruction>(U->getUser());
            if (!UI) { callOk = false; break; }
            if (auto *BC = dyn_cast<BitCastInst>(UI)) {
              IRBuilder<> BB2(BC);
              Value *NewBC = BB2.CreateBitCast(NewAI, BC->getType());
              BC->replaceAllUsesWith(NewBC);
              BC->eraseFromParent();
            } else if (auto *SI2 = dyn_cast<StoreInst>(UI)) {
              if (U->getOperandNo() != 1) { callOk = false; break; }
              IRBuilder<> SB2(SI2);
              Value *Val = SI2->getValueOperand();
              Value *NewVal = SB2.CreateTrunc(Val, Type::getIntNTy(Ctx, PtrN));
              SB2.CreateStore(NewVal, NewAI);
              SI2->eraseFromParent();
            } else if (auto *LI2 = dyn_cast<LoadInst>(UI)) {
              IRBuilder<> LB2(LI2);
              Value *NLI = LB2.CreateLoad(Type::getIntNTy(Ctx, PtrN), NewAI);
              Value *Ext = ArgIsSigned[i]
                  ? LB2.CreateSExt(NLI, LI2->getType())
                  : LB2.CreateZExt(NLI, LI2->getType());
              LI2->replaceAllUsesWith(Ext);
              LI2->eraseFromParent();
            } else {
              callOk = false; break;
            }
          }
          if (!callOk) break;
          NewArgs.push_back(NewAI);
          DeadAllocas.push_back(OldAI);
        } else {
          Value *V = CI->getArgOperand(i);
          if (auto *S = dyn_cast<SExtInst>(V)) NewArgs.push_back(S->getOperand(0));
          else if (auto *Z = dyn_cast<ZExtInst>(V)) NewArgs.push_back(Z->getOperand(0));
          else { callOk = false; break; }
        }
      }
      if (!callOk) {
        // Bail: leave this call (and undo nothing — narrow attempt was IR-mutating).
        // To be safe, bail out of the whole function narrowing.
        NewF->eraseFromParent();
        ok = false;
        break;
      }
      CallInst *NewCI = B.CreateCall(NewF, NewArgs);
      NewCI->setCallingConv(CI->getCallingConv());
      DeadCalls.push_back(CI);
    }
    if (!ok) continue;

    // Erase old calls then old allocas (now use-empty).
    for (CallInst *CI : DeadCalls) CI->eraseFromParent();
    for (AllocaInst *AI : DeadAllocas) {
      if (AI->use_empty()) AI->eraseFromParent();
    }
    // Erase old function and rename new one to original name.
    F->eraseFromParent();
    NewF->setName(OldName);
    Changed = true;
  }
  return Changed;
}

static bool dropLlvmAssume(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<Instruction *, 16> Dead;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getIntrinsicID() == Intrinsic::assume)
            Dead.push_back(CI);
    for (Instruction *I : Dead) { I->eraseFromParent(); Changed = true; }
  }
  // Drop the declaration if no remaining users.
  if (Function *F = M.getFunction("llvm.assume"))
    if (F->use_empty()) { F->eraseFromParent(); Changed = true; }
  return Changed;
}

static bool inlineFromIdentity(Module &M) {
  bool Changed = false;
  SmallVector<Function *, 4> Dead;
  for (Function &F : M) {
    StringRef N = F.getName();
    // Match mangled `<T as core::convert::From<T>>::from` or `core::convert::From::from`.
    if (!N.contains("core..convert..From") && !N.contains("core::convert::From")) continue;
    if (!N.contains("from")) continue;
    if (F.arg_size() != 1) continue;
    if (F.getReturnType() != F.getArg(0)->getType()) continue;
    SmallVector<CallInst *, 8> Calls;
    for (User *U : F.users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      CI->replaceAllUsesWith(CI->getArgOperand(0));
      CI->eraseFromParent();
      Changed = true;
    }
    if (F.use_empty()) Dead.push_back(&F);
  }
  for (Function *F : Dead) F->eraseFromParent();
  return Changed;
}

static bool stripOpenclLeakedMetadata(Module &M) {
  static const char *Names[] = {
      "opencl.ocl.version", "opencl.spir.version", "opencl.kernels",
      "opencl.used.optional.core.features", "opencl.used.extensions",
      "opencl.compiler.options"};
  bool Changed = false;
  for (const char *N : Names)
    if (NamedMDNode *NMD = M.getNamedMetadata(N)) {
      NMD->eraseFromParent();
      Changed = true;
    }
  return Changed;
}

static bool stripFreeze(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<Instruction *, 8> Dead;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I)) {
          FI->replaceAllUsesWith(FI->getOperand(0));
          Dead.push_back(FI);
          Changed = true;
        }
    for (Instruction *I : Dead)
      I->eraseFromParent();
  }
  return Changed;
}


// ===== EarlyPrep passes =====
// These 6 passes + the vxxEarlyShape entry shape rustc -O0 IR before
// the LLVM mid-end runs. They are self-contained (only llvm/std and
// each other) and belong on the OPEN side.

static bool stripAllocaInlineSuffix(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    // Collect base → alloca list. Base = greedy strip of `\.\d+$`, then
    // greedy strip of all trailing `.i` segments.
    std::map<std::string, SmallVector<AllocaInst *, 2>> ByBase;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *AI = dyn_cast<AllocaInst>(&I);
        if (!AI || !AI->hasName()) continue;
        StringRef N = AI->getName();
        // Skip allocas already with no inline-noise suffix.
        if (!N.contains(".i")) continue;
        std::string Base = N.str();
        // Strip trailing `\.\d+`.
        size_t LastDot = Base.find_last_of('.');
        if (LastDot != std::string::npos) {
          bool AllDigits = LastDot + 1 < Base.size();
          for (size_t i = LastDot + 1; i < Base.size(); ++i) {
            if (!isdigit((unsigned char)Base[i])) { AllDigits = false; break; }
          }
          if (AllDigits) Base.resize(LastDot);
        }
        // Strip trailing `.i` segments (greedy).
        while (Base.size() >= 2 &&
               Base[Base.size() - 2] == '.' && Base[Base.size() - 1] == 'i') {
          Base.resize(Base.size() - 2);
        }
        if (Base.empty() || Base == N.str()) continue;
        ByBase[Base].push_back(AI);
      }
    }
    // Build set of names currently in use in F's value symbol table to
    // detect collisions with existing non-suffixed allocas.
    std::set<std::string> InUse;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          if (AI->hasName()) InUse.insert(AI->getName().str());
      }
    }
    for (auto &P : ByBase) {
      const std::string &Base = P.first;
      auto &List = P.second;
      if (List.size() != 1) continue;
      // If base is already in use by a different alloca, leave alone.
      if (InUse.count(Base)) continue;
      List[0]->setName(Base);
      Changed = true;
    }
  }
  return Changed;
}

// Undefined function. The body is a simple stateful step:
//   fn next(&mut self) -> Option<T> {
//     if self.start < self.end { let r=self.start; self.start+=1; Some(r) }
//     else { None }
//   }
// returned as { iX disc, iX value } with disc 1=Some / 0=None (Rust's
// Option<int> layout). We synthesize it so forceInlineIteratorTrampolines can
// inline it and the subsequent IndVarSimplify (simplifyKernelLoops) makes the
// loop countable — i.e. a Rust `for i in 0..N` becomes HLS-synthesizable
// regardless of opt level (at -O1 rustc inlines this itself).
static bool synthesizeRangeNextBody(Module &M) {
  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  unsigned N = 0;
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;
    StringRef Nm = F.getName();
    if (!(Nm.contains("ops..range") && Nm.contains("next")))
      continue;
    FunctionType *FT = F.getFunctionType();
    if (FT->getNumParams() != 1) continue;
    auto *RetST = dyn_cast<StructType>(FT->getReturnType());
    if (!RetST || RetST->getNumElements() != 2) continue;
    auto *ITy = dyn_cast<IntegerType>(RetST->getElementType(0));
    if (!ITy || RetST->getElementType(1) != ITy) continue;
    auto *PT = dyn_cast<PointerType>(FT->getParamType(0));
    if (!PT || PT->getElementType() != RetST) continue;

    BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", &F);
    BasicBlock *Some = BasicBlock::Create(Ctx, "some", &F);
    BasicBlock *None = BasicBlock::Create(Ctx, "none", &F);
    IRBuilder<> B(Entry);
    Value *Self = &*F.arg_begin();
    Value *Sp = B.CreateConstInBoundsGEP2_32(RetST, Self, 0, 0);
    Value *Start = B.CreateLoad(ITy, Sp);
    Value *Ep = B.CreateConstInBoundsGEP2_32(RetST, Self, 0, 1);
    Value *End = B.CreateLoad(ITy, Ep);
    B.CreateCondBr(B.CreateICmpSLT(Start, End), Some, None);

    B.SetInsertPoint(Some);
    Value *Nx = B.CreateAdd(Start, ConstantInt::get(ITy, 1));
    B.CreateStore(Nx, Sp);
    Value *R0 = B.CreateInsertValue(UndefValue::get(RetST),
                                    ConstantInt::get(ITy, 1), {0});
    Value *R1 = B.CreateInsertValue(R0, Start, {1});
    B.CreateRet(R1);

    B.SetInsertPoint(None);
    Value *N0 = B.CreateInsertValue(UndefValue::get(RetST),
                                    ConstantInt::get(ITy, 0), {0});
    B.CreateRet(N0);

    F.setLinkage(GlobalValue::InternalLinkage);
    Changed = true;
    N++;
  }
  if (N)
    vxxDbg() << "vxx: synthesized Range::next body x" << N
           << " (-O0 for-loop support)\n";
  return Changed;
}

static bool synthesizeSliceTrampolines(Module &M) {
  bool Changed = false;
  unsigned N = 0;
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;
    StringRef Nm = F.getName();
    bool IsAsPtr = Nm.contains("5slice") && Nm.contains("as_ptr");
    bool IsLen = Nm.contains("5slice") && Nm.contains("3len17");
    if (!IsAsPtr && !IsLen) continue;
    FunctionType *FT = F.getFunctionType();
    if (FT->getNumParams() != 2) continue;
    auto *P0 = dyn_cast<PointerType>(FT->getParamType(0));
    if (!P0) continue;
    BasicBlock *Entry = BasicBlock::Create(M.getContext(), "entry", &F);
    IRBuilder<> B(Entry);
    if (IsLen) {
      // len(ptr, i64 n) -> n   (return the length argument)
      if (FT->getReturnType() != FT->getParamType(1)) continue;
      B.CreateRet(&*std::next(F.arg_begin()));
    } else {
      // as_ptr([0 x E]* p, i64 n) -> E*  (data pointer = &p[0])
      auto *RetPT = dyn_cast<PointerType>(FT->getReturnType());
      auto *AT = dyn_cast<ArrayType>(P0->getElementType());
      if (!RetPT || !AT) { Entry->eraseFromParent(); continue; }
      Value *GEP = B.CreateConstInBoundsGEP2_64(P0->getElementType(),
                                                &*F.arg_begin(), 0, 0);
      if (GEP->getType() != RetPT)
        GEP = B.CreateBitCast(GEP, RetPT);
      B.CreateRet(GEP);
    }
    F.setLinkage(GlobalValue::InternalLinkage);
    Changed = true; N++;
  }
  if (N)
    vxxDbg() << "vxx: synthesized slice as_ptr/len body x" << N
           << " (-O0 for-loop support)\n";
  return Changed;
}

// At -O0 rustc does not inline the trivial `#[inline]` libcore volatile
// accessors, and their bodies can be dropped, leaving an undefined
// `declare` (e.g. `core::ptr::read_volatile`). The HLS backend then rejects the
// design (HLS 214-194 "Undefined function"). clang has no equivalent issue:
// it lowers `*p` straight to a volatile load even with -disable-llvm-optzns.
// We restore that by giving the declared-only wrappers a body — a single
// volatile load/store — so synthesis works regardless of opt level. (At -O1
// rustc already inlines these, so this is a no-op there.)
static bool synthesizeVolatileAccessors(Module &M) {
  bool Changed = false;
  unsigned N = 0;
  for (Function &F : M) {
    if (!F.isDeclaration()) continue;
    StringRef Nm = F.getName();
    bool IsRead = Nm.contains("3ptr") && Nm.contains("read_volatile");
    bool IsWrite = Nm.contains("3ptr") && Nm.contains("write_volatile");
    if (!IsRead && !IsWrite) continue;
    FunctionType *FT = F.getFunctionType();
    BasicBlock *Entry = BasicBlock::Create(M.getContext(), "entry", &F);
    IRBuilder<> B(Entry);
    if (IsRead) {
      // read_volatile(T* %p) -> T  ==  load volatile T, T* %p
      if (FT->getNumParams() != 1) { Entry->eraseFromParent(); continue; }
      auto *PT = dyn_cast<PointerType>(FT->getParamType(0));
      if (!PT || PT->getElementType() != FT->getReturnType()) {
        Entry->eraseFromParent(); continue;
      }
      LoadInst *L = B.CreateLoad(FT->getReturnType(), &*F.arg_begin());
      L->setVolatile(true);
      B.CreateRet(L);
    } else {
      // write_volatile(T* %dst, T %src)  ==  store volatile T %src, T* %dst
      if (FT->getNumParams() != 2 || !FT->getReturnType()->isVoidTy()) {
        Entry->eraseFromParent(); continue;
      }
      auto *PT = dyn_cast<PointerType>(FT->getParamType(0));
      if (!PT || PT->getElementType() != FT->getParamType(1)) {
        Entry->eraseFromParent(); continue;
      }
      StoreInst *S = B.CreateStore(&*std::next(F.arg_begin()), &*F.arg_begin());
      S->setVolatile(true);
      B.CreateRetVoid();
    }
    F.setLinkage(GlobalValue::InternalLinkage);
    Changed = true; N++;
  }
  if (N)
    vxxDbg() << "vxx: synthesized volatile accessor body x" << N
           << " (-O0 support)\n";
  return Changed;
}

static bool stripLlvmExpect(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> Dead;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Cal = CI->getCalledFunction())
            if (Cal->getName().startswith("llvm.expect")) {
              CI->replaceAllUsesWith(CI->getArgOperand(0));
              Dead.push_back(CI);
            }
  for (CallInst *CI : Dead) {
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static bool forceInlineIteratorTrampolines(Module &M) {
  bool Changed = false;
  unsigned Replaced = 0, Inlined = 0;
  SmallVector<CallInst *, 32> ToReplace;
  SmallVector<CallInst *, 32> ToInline;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Cal = CI->getCalledFunction();
          if (!Cal) continue;
          StringRef Name = Cal->getName();
          if (!Name.contains("IntoIterator") &&
              !(Name.contains("ops..range") && Name.contains("next")) &&
              !Name.contains("..iter..traits..iterator..Iterator") &&
              !Name.contains("Iterator..next") &&
              // newtype-ish iterator adapter constructors (libcore decls
              // we can't link in). rustc uses legacy mangling for outer
              // functions: `Rev<T>` → `Rev$LT$T$GT$`. Catch both legacy
              // (`Rev$LT$`) and v0-style (`rev..Rev`) forms.
              !Name.contains("Rev$LT$") &&
              !Name.contains("rev..Rev") &&
              !Name.contains("StepBy$LT$") &&
              !Name.contains("step_by..StepBy") &&
              // `<T as From<T>>::from` blanket impl (identity). Called
              // from `Into::into` inside StepBy iteration's next() path
              // (e.g., usize → usize convert). libcore decls only.
              !(Name.contains("convert..From") && Name.contains("4from")) &&
              !(Name.contains("convert..Into") && Name.contains("4into")))
            continue;
          // Try direct replacement first: into_iter on Range<T> is identity.
          // Pattern: callee returns `{T, T}` or `{T, T}*` (sret) given args
          // `(T, T)` or `(sret_ptr, T, T)`.
          if (Cal->isDeclaration()) {
            ToReplace.push_back(CI);
          } else {
            ToInline.push_back(CI);
          }
        }
  }
  for (CallInst *CI : ToInline) {
    InlineFunctionInfo IFI;
    if (InlineFunction(*CI, IFI).isSuccess()) {
      Inlined++;
      Changed = true;
    }
  }
  for (CallInst *CI : ToReplace) {
    Function *Cal = CI->getCalledFunction();
    Type *RetTy = Cal->getReturnType();
    // Case 0: `T from(T self)` / `T into(T self)` — identity. libcore
    // `From<T> for T` / `Into::into` blanket impls. 1 arg, same type
    // returned as taken.
    if (CI->arg_size() == 1 && CI->getArgOperand(0)->getType() == RetTy) {
      CI->replaceAllUsesWith(CI->getArgOperand(0));
      CI->eraseFromParent();
      Replaced++;
      Changed = true;
      continue;
    }
    // Case A: `{T, T} into_iter(T start, T end)` — identity, return {a, b}.
    if (auto *ST = dyn_cast<StructType>(RetTy)) {
      if (ST->getNumElements() == 2 && CI->arg_size() == 2 &&
          ST->getElementType(0) == CI->getArgOperand(0)->getType() &&
          ST->getElementType(1) == CI->getArgOperand(1)->getType()) {
        IRBuilder<> B(CI);
        Value *S = UndefValue::get(ST);
        S = B.CreateInsertValue(S, CI->getArgOperand(0), {0});
        S = B.CreateInsertValue(S, CI->getArgOperand(1), {1});
        CI->replaceAllUsesWith(S);
        CI->eraseFromParent();
        Replaced++;
        Changed = true;
        continue;
      }
    }
    // Case B: `void into_iter({T, T}* sret, T start, T end)` — store {a, b}.
    if (RetTy->isVoidTy() && CI->arg_size() == 3) {
      Value *Sret = CI->getArgOperand(0);
      if (auto *PT = dyn_cast<PointerType>(Sret->getType())) {
        if (auto *ST = dyn_cast<StructType>(PT->getElementType())) {
          if (ST->getNumElements() == 2 &&
              ST->getElementType(0) == CI->getArgOperand(1)->getType() &&
              ST->getElementType(1) == CI->getArgOperand(2)->getType()) {
            IRBuilder<> B(CI);
            Value *S = UndefValue::get(ST);
            S = B.CreateInsertValue(S, CI->getArgOperand(1), {0});
            S = B.CreateInsertValue(S, CI->getArgOperand(2), {1});
            B.CreateStore(S, Sret);
            CI->eraseFromParent();
            Replaced++;
            Changed = true;
            continue;
          }
        }
      }
    }
  }

  // Case C: rebuild StepBy::new function bodies. stage1 rustc emits a
  // malformed first store inside `StepBy::new` for fpga64 — `store i64
  // %iter.0, {i64,i64}* %v1` where %v1 was GEP'd one level too shallow
  // (stops at the inner Range struct ptr instead of descending to its
  // first i64 field). LLVM 11 lets this slide; Vitis HLS LLVM 7 llvm-as
  // rejects it. Detect StepBy::new defs by name + signature shape and
  // regenerate the body in canonical form.
  //
  // Signature: void @StepBy::new(StepBy* sret, T iter.0, T iter.1, T step)
  // Struct:    { {T,T}, T, i8, [padding] }
  // Body:      store iter.0 to .0.0;  store iter.1 to .0.1;
  //            store (step-1) to .1;  store 1 to .2;  ret void
  unsigned StepByRebuilt = 0;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    StringRef Name = F.getName();
    if (!Name.contains("StepBy$LT$") && !Name.contains("step_by..StepBy"))
      continue;
    if (!Name.contains("3new")) continue;
    if (!F.getReturnType()->isVoidTy()) continue;
    if (F.arg_size() != 4) continue;
    auto *SretPT = dyn_cast<PointerType>(F.getArg(0)->getType());
    if (!SretPT) continue;
    auto *OuterST = dyn_cast<StructType>(SretPT->getElementType());
    if (!OuterST || OuterST->getNumElements() < 3) continue;
    auto *RangeST = dyn_cast<StructType>(OuterST->getElementType(0));
    if (!RangeST || RangeST->getNumElements() != 2) continue;
    auto *I0T = dyn_cast<IntegerType>(RangeST->getElementType(0));
    auto *I1T = dyn_cast<IntegerType>(RangeST->getElementType(1));
    auto *StepT = dyn_cast<IntegerType>(OuterST->getElementType(1));
    auto *FlagT = dyn_cast<IntegerType>(OuterST->getElementType(2));
    if (!I0T || !I1T || !StepT || !FlagT) continue;
    if (F.getArg(1)->getType() != I0T) continue;
    if (F.getArg(2)->getType() != I1T) continue;
    if (F.getArg(3)->getType() != StepT) continue;

    Value *Sret = F.getArg(0);
    Value *IterStart = F.getArg(1);
    Value *IterEnd = F.getArg(2);
    Value *StepArg = F.getArg(3);
    // Preserve arg names across deleteBody — some LLVM 11 paths clear
    // them, which leaves the sret arg printed as `%0` instead of `%a0`.
    // Vitis HLS LLVM 7 llvm-as parser rejects unnamed sret args.
    SmallVector<std::string, 4> ArgNames;
    for (Argument &A : F.args()) ArgNames.push_back(A.getName().str());
    vxxDbg() << "vxx: StepBy::new arg names pre: ["
           << ArgNames[0] << "," << ArgNames[1] << "," << ArgNames[2]
           << "," << ArgNames[3] << "]\n";
    LLVMContext &Ctx = M.getContext();
    F.deleteBody();
    for (unsigned i = 0; i < F.arg_size(); ++i) {
      StringRef Cur = F.getArg(i)->getName();
      vxxDbg() << "vxx: StepBy::new arg " << i
             << " name post-delete: '" << Cur << "'\n";
      if (Cur.empty() && !ArgNames[i].empty()) {
        // Use a unique name to avoid auto-numbering collision.
        F.getArg(i)->setName(Twine("a") + Twine(i));
      }
    }
    // Re-acquire pointers in case deleteBody invalidated them.
    Sret = F.getArg(0);
    IterStart = F.getArg(1);
    IterEnd = F.getArg(2);
    StepArg = F.getArg(3);
    BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", &F);
    IRBuilder<> B(Entry);
    // .0.0 = iter.0  (start)
    Value *F00 = B.CreateConstInBoundsGEP2_32(OuterST, Sret, 0, 0);
    Value *F000 = B.CreateConstInBoundsGEP2_32(RangeST, F00, 0, 0);
    B.CreateStore(IterStart, F000);
    // .0.1 = iter.1  (end)
    Value *F001 = B.CreateConstInBoundsGEP2_32(RangeST, F00, 0, 1);
    B.CreateStore(IterEnd, F001);
    // .1 = step - 1
    Value *F1 = B.CreateConstInBoundsGEP2_32(OuterST, Sret, 0, 1);
    Value *StepM1 = B.CreateSub(StepArg, ConstantInt::get(StepT, 1));
    B.CreateStore(StepM1, F1);
    // .2 = 1  (first_take)
    Value *F2 = B.CreateConstInBoundsGEP2_32(OuterST, Sret, 0, 2);
    B.CreateStore(ConstantInt::get(FlagT, 1), F2);
    B.CreateRetVoid();
    StepByRebuilt++;
    Changed = true;
  }
  if (StepByRebuilt)
    vxxDbg() << "vxx: StepBy::new bodies rebuilt: " << StepByRebuilt
           << "\n";

  if (Changed)
    vxxDbg() << "vxx: Iterator trampoline: inlined=" << Inlined
           << " replaced=" << Replaced << "\n";
  return Changed;
}

static bool stripPanicCalls(Module &M) {
  bool Changed = false;
  SmallVector<Function *, 8> Targets;
  for (Function &F : M) {
    StringRef N = F.getName();
    if (N.contains("panic_bounds_check") ||
        N.contains("panic_fmt") ||
        N.contains("panic_cannot_unwind") ||
        N.contains("rust_begin_unwind") ||
        N.contains("4core9panicking5panic")) {
      Targets.push_back(&F);
    }
  }
  for (Function *F : Targets) {
    SmallVector<CallInst *, 4> Calls;
    for (User *U : F->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      IRBuilder<> B(CI);
      // Replace the panic call with unreachable.
      B.CreateUnreachable();
      // Remove all instructions after the call in its BB.
      BasicBlock *BB = CI->getParent();
      auto It = CI->getIterator();
      ++It;
      while (It != BB->end()) {
        Instruction *Next = &*It;
        ++It;
        if (Next->isTerminator() && Next != &BB->back()) continue;
        if (!Next->use_empty()) Next->replaceAllUsesWith(UndefValue::get(Next->getType()));
        Next->eraseFromParent();
      }
      CI->eraseFromParent();
      Changed = true;
    }
    if (F->use_empty()) { F->eraseFromParent(); Changed = true; }
  }
  return Changed;
}

} } // namespace hlsrs::vxx

using namespace hlsrs::vxx;

extern "C" void vxxShapeIR(LLVMModuleRef MRef) {
  Module &M = *llvm::unwrap(MRef);

  // Phase 0: scrub rustc-leaked named metadata that confuses the HLS backend.
  stripOpenclLeakedMetadata(M);

  // Phase 0.5: reassemble the phase-2 struct-layout / storage pragma builders
  // (`array_partition()` / `array_reshape()` / `bind_storage()`) from their
  // token chains into the legacy markers. Runs on pristine rustc output —
  // before any consumer (arg-decay, disaggCompletePartitionKernelSig,
  // redirectWholeStructBindStorage) and before SROA — so the legacy marker is
  // byte-identical to the old Drop-builder emission. Both marker forms are
  // accepted during the migration.
  injectArrayPartitionChain(M);
  injectArrayReshapeChain(M);
  injectBindStorageChain(M);
  injectArrayStencilChain(M);
  // NB: aggregate() / disaggregate() / alias() are NOT token-chain — they stay
  // Drop-builders (rustc emits the legacy marker directly). Their consumers need
  // the marker's port operand to resolve to the already-retyped AXIS / m_axi /
  // ap_fifo arg, which a reassembled chain marker fails to provide at any stage
  // (disagg → port unsplit HLS 214-244; alias → VXXPrep crash; aggregate on an
  // ap_fifo struct port → FIFO degrades to BRAM, Stage A DIFF).

  // Phase 1: post-opt IR cleanup (Rust idioms → plain LLVM).
  inlineFromIdentity(M);
  dropLlvmAssume(M);
  stripFreeze(M);
  lowerWithOverflowIntrinsics(M);
  elideDeadUninitMemcpy(M); // drop uninit_array copy -> no residual top loop
  promoteKernelAllocas(M); // SROA+mem2reg first so Option struct becomes select/phi
  foldExtractValue(M);     // collapse Range::next Option struct -> canonical loop test
  canonicalizeRangeLoopStep(M); // rustc 0..N saturating step -> +1 so IndVarSimplify forms a canonical IV (stencil burst)
  simplifyKernelLoops(M);       // SROA+mem2reg+IndVarSimplify: materializes the clean dual-counter PHI form
  if (collapseRangeLookaheadPhi(M)) // decouple the look-ahead dual-counter (single canonical IV like C++)
    simplifyKernelLoops(M);     // re-run IndVarSimplify to drop the now-redundant look-ahead PHI
  markCountedLoopIVNoWrap(M);    // restore clang's nsw/nuw on the canonical step for downstream SCEV burst-inference
  unrotateCountedLoops(M);        // rustc rotated do-while → clang for.cond, enabling downstream y+x flatten/merge (stencil)
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    collapseBitIntNarrowRoundtrip(F);
    collapseBitIntShlAshrPair(F);
    collapseBitIntMulShifts(F);
    narrowBitIntMulTrunc(F);
    narrowSextClampedPhi(F);
    narrowMaskedBinopTrunc(F);
    demoteBitIntBinopTrunc(F);
    demoteEscapingBitIntPhi(F);
    narrowBitIntSignClamp(F);
    narrowBitFixedMulShiftTrunc(F);
    eraseTriviallyDeadInsts(F);
  }
  narrowStaticIntGlobalToBitInt(M);
  narrowBitInts(M);
  // Re-run the BitFixed multiply-narrowing peephole now that narrowBitInts has
  // rewritten the kernel boundary to native iN (clean `sext/zext iA → i128`
  // operand shapes). Before the narrowing, the i128 fixed-point multiply's
  // operands are rustc's masked/shifted i128 normal forms and the pattern
  // can't match — so the `(sext i6 <<3) * (zext i10 <<90) ashr exact 92 →
  // trunc i36` form survived to the emitted bc, and reflow's bit-width
  // minimization MISCOMPILES it (the negative product's sign extension is
  // lost above bit ~33: using_fixed_point cosim mismatch on every in2<0
  // transaction, RTL = golden + 3*2^27). The narrowed form (mul in the
  // destination width + small shl) is the same staged shape clang emits for
  // ap_fixed, which reflow handles correctly.
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (narrowBitFixedMulShiftTrunc(F))
      eraseTriviallyDeadInsts(F);
  }
  narrowBlackboxExtern(M);

  // Phase 2: static globals + struct layout cleanup.
  renameSingleUserStaticGlobals(M);
  rewriteStaticIntGlobals(M);
  evaluateRomInitCalls(M); // clang-style compile-time ROM-init folding — needs the typed-array global shape from the rewrite above
  narrowStaticApIntArrays(M); // fn-local static [Ap(U)Int<W>; K] -> [K x iW] (also needs the typed-array shape)
  redirectWholeStructBindStorage(M);
  stripStreamWrapperPadding(M);
  stripZeroSizeArrayStructFields(M);
  // Dissolve named single-scalar-field wrapper structs (C++ dat_t<T>) into
  // their field type — AFTER the ZSA strip (rustc pads the wrapper with
  // `[0 x T]` fields, so the single-field shape only exists post-strip).
  // The C++ frontend flattens these pre-reflow, so all downstream machinery
  // (stream renames, fifo ops, axis, m_axi) must see the scalar.
  hlsrs::vxx::dissolveSingleFieldStructs(M);
  dropAxisDisabledPacketFields(M);
  stripAxisPacketTrailingPad(M);
  // -O0: strip dead function declarations left after lowering (e.g. an unused
  // llvm.umul.with.overflow decl whose calls were already lowered). HLS SYNCHK
  // rejects the mere presence of unsupported intrinsic declarations; at -O1
  // LLVM's StripDeadPrototypes removes them, so replicate it here.
  {
    SmallVector<Function *, 16> DeadDecls;
    for (Function &F : M)
      if (F.isDeclaration() && F.use_empty())
        DeadDecls.push_back(&F);
    unsigned NStripped = 0;
    for (Function *F : DeadDecls) { F->eraseFromParent(); ++NStripped; }
    if (NStripped)
      vxxDbg() << "vxx: stripped " << NStripped
             << " dead function declaration(s) (-O0)\n";
  }
  // -O0: strip dead rustc-mangled libcore/std function DEFINITIONS that the top
  // kernel doesn't reach. At -O0 rustc emits & keeps these monomorphisations
  // (core::ptr::{read,write}, core::mem::swap, Range::next, slice::{len,as_ptr},

}

// EarlyPrep entry (backs VXXEarlyPrep). Runs before std/middle-end optimizations.
// Vitis' proprietary `reflow` backend mis-synthesises a ROTATED (do-while)
// loop that pops/pushes a FIFO in both arms of a branch under a variable trip
// count (reproduced in pure C++). The burst_maxi cosim compute core (BurstMaxi
// read()/write() lowered to the mc_fifo_pop/mc_fifo_push helpers) is exactly
// that shape, so when the module contains those helpers we pin LLVM's
// LoopRotate threshold to 0 here — BEFORE the opt pipeline runs — so the loop
// reaches reflow in canonical while (header-compare) form. Detected from the
// IR, not a build env flag. The real m_axi kernel bc has no such helpers.
static void vxxMaybeDisableLoopRotate(Module &M) {
  bool IsBurstComputeCore = false;
  for (Function &F : M)
    if (F.getName().contains("mc_fifo_pop") ||
        F.getName().contains("mc_fifo_push")) {
      IsBurstComputeCore = true;
      break;
    }
  if (!IsBurstComputeCore)
    return;
  auto &Opts = llvm::cl::getRegisteredOptions();
  auto It = Opts.find("rotation-max-header-size");
  if (It != Opts.end())
    if (auto *O = static_cast<llvm::cl::opt<unsigned> *>(It->second))
      O->setValue(0);
}

// Cache the ap_axis_user user width (e.g. 13) from the `__vxx_axis_w` marker's
// USER arg (index 3) into a module flag, BEFORE the opt/axis passes consume and
// erase the marker. The AXIS canonicalisation reads it back much later via
// axisUserWidthFromMarker. barista_hls' own TU has the marker unused → no flag.
static void cacheAxisUserWidth(Module &M) {
  Function *WM = M.getFunction("__vxx_axis_user_w");
  if (!WM)
    return;
  for (User *U : WM->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (CI->arg_size() == 1)
        if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
          if (unsigned W = (unsigned)C->getZExtValue()) {
            M.addModuleFlag(Module::Warning, "vxx.axis_user_w", W);
            return;
          }
}

extern "C" void vxxEarlyShape(LLVMModuleRef MRef) {
  Module &M = *llvm::unwrap(MRef);
  vxxMaybeDisableLoopRotate(M);
  cacheAxisUserWidth(M);
  // Runs before std/middle-end optimizations.
  // Job: shape rustc IR so subsequent opts (inlining, GVN, instcombine)
  // can simplify aggressively. Anything that depends on flattened
  // closures or inlined iterator trampolines belongs here.
  synthesizeRangeNextBody(M);   // -O0: give Range::next a body so it inlines
  synthesizeSliceTrampolines(M);// -O0: give slice as_ptr/len bodies
  synthesizeVolatileAccessors(M);// -O0: give read/write_volatile a load/store body
  stripLlvmExpect(M);           // -O0: drop llvm.expect bounds-check hints
  forceInlineIteratorTrampolines(M);
  stripAllocaInlineSuffix(M);
  stripPanicCalls(M);
}
