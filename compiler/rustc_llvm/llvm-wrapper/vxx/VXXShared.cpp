// VXXShared.cpp — definitions of the low-level helpers shared across the prep
// stages (see VXXShared.h). Carved out of VXXPrep.cpp.

#include "VXXShared.h"

#include "VXXPrep.h"

#include <cstdlib>
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

namespace hlsrs { namespace vxx {

raw_ostream &vxxDbg() {
  // Silent by default; `VXX_DEBUG=1` routes pass diagnostics to stderr.
  static const bool On = ::getenv("VXX_DEBUG") != nullptr;
  return On ? errs() : nulls();
}

bool markerUsed(Module &M, StringRef Name) {
  Function *F = M.getFunction(Name);
  // A kernel module references the marker as an EXTERNAL declaration (its body
  // lives in barista_hls); barista_hls' own translation unit has the DEFINITION
  // (empty body). Keying on `isDeclaration()` detects the kernel case robustly —
  // the declaration persists even after the opt pipeline drops the side-effect-
  // free marker CALL — while excluding barista_hls (where the paths must not
  // fire, else the library IR is mangled).
  return F && F->isDeclaration();
}

void replaceFunctionKeepingName(Function *Old, Function *New) {
  std::string Name = Old->getName().str();
  Old->eraseFromParent();
  New->setName(Name);
}

std::string mangleForIntrinsic(Type *T) {
  if (auto *I = dyn_cast<IntegerType>(T)) {
    std::string S;
    raw_string_ostream OS(S);
    OS << "i" << I->getBitWidth();
    return OS.str();
  }
  if (auto *P = dyn_cast<PointerType>(T)) {
    std::string S;
    raw_string_ostream OS(S);
    OS << "p" << P->getAddressSpace() << mangleForIntrinsic(P->getElementType());
    return OS.str();
  }
  if (auto *S = dyn_cast<StructType>(T)) {
    if (S->hasName()) {
      // LLVM intrinsic mangling drops the `struct.`/`class.` prefix and
      // replaces `:`/spaces with `_`. We just keep the raw name; the only
      // requirement is that the same struct type produces the same string.
      std::string N = S->getName().str();
      for (auto &c : N)
        if (c == '.' || c == ':' || c == ' ' || c == '<' || c == '>' || c == ',')
          c = '_';
      return std::string("s_") + N;
    }
    return std::string("s_anon");
  }
  std::string S;
  raw_string_ostream OS(S);
  T->print(OS);
  return OS.str();
}

Function *getOrInsertFifoPopAny(Module &M, Type *ElemTy) {
  Type *PtrTy = PointerType::getUnqual(ElemTy);
  std::string Name = std::string("llvm.fpga.fifo.pop.") +
                     mangleForIntrinsic(ElemTy) + "." +
                     mangleForIntrinsic(PtrTy);
  FunctionType *FT = FunctionType::get(ElemTy, {PtrTy}, false);
  return cast<Function>(M.getOrInsertFunction(Name, FT).getCallee());
}

Function *getOrInsertFifoPushAny(Module &M, Type *ElemTy) {
  Type *PtrTy = PointerType::getUnqual(ElemTy);
  std::string Name = std::string("llvm.fpga.fifo.push.") +
                     mangleForIntrinsic(ElemTy) + "." +
                     mangleForIntrinsic(PtrTy);
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), {ElemTy, PtrTy}, false);
  return cast<Function>(M.getOrInsertFunction(Name, FT).getCallee());
}

Type *remapTypeRecursive(Type *T,
                                const DenseMap<Type *, Type *> &TypeMap,
                                LLVMContext &Ctx) {
  auto It = TypeMap.find(T);
  if (It != TypeMap.end()) return It->second;
  if (auto *PT = dyn_cast<PointerType>(T)) {
    Type *Pointee = remapTypeRecursive(PT->getElementType(), TypeMap, Ctx);
    if (Pointee != PT->getElementType())
      return PointerType::get(Pointee, PT->getAddressSpace());
  }
  if (auto *AT = dyn_cast<ArrayType>(T)) {
    Type *Elem = remapTypeRecursive(AT->getElementType(), TypeMap, Ctx);
    if (Elem != AT->getElementType())
      return ArrayType::get(Elem, AT->getNumElements());
  }
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    // Remap return + parameter types so function-pointer bitcasts (e.g. a
    // `bitcast(@callee to oldFnTy*)` left by signature rebuild) reference the
    // NEW struct types, not the stale `.pad` shells.
    Type *Ret = remapTypeRecursive(FT->getReturnType(), TypeMap, Ctx);
    SmallVector<Type *, 8> Params;
    bool changed = (Ret != FT->getReturnType());
    for (Type *P : FT->params()) {
      Type *NP = remapTypeRecursive(P, TypeMap, Ctx);
      if (NP != P) changed = true;
      Params.push_back(NP);
    }
    if (changed)
      return FunctionType::get(Ret, Params, FT->isVarArg());
  }
  return T;
}

void substituteTypeInFunction(Function *F,
                                      const DenseMap<Type *, Type *> &TypeMap,
                                      ArrayRef<std::pair<Value *, Value *>> SeedVMap,
                                      const DenseMap<StructType *, std::vector<int>> *FieldMaps) {
  if (TypeMap.empty()) return;
  LLVMContext &Ctx = F->getContext();
  // A DISSOLVED struct maps to a non-struct (single-scalar-field wrappers like
  // C++ dat_t<T> flatten to T): its field-selector indices vanish from GEP /
  // extract / insert chains rather than shifting.
  auto IsDissolved = [&TypeMap](Type *T) -> bool {
    if (auto *ST = dyn_cast<StructType>(T)) {
      auto It = TypeMap.find(ST);
      return It != TypeMap.end() && !isa<StructType>(It->second);
    }
    return false;
  };
  bool AnyDissolved = false;
  for (auto &KV : TypeMap)
    if (isa<StructType>(KV.first) && !isa<StructType>(KV.second)) {
      AnyDissolved = true;
      break;
    }
  ValueToValueMapTy VMap;
  // Pre-seed (e.g. old-arg → new-arg after a BB splice into a retyped
  // function shell) so operand patching below redirects body uses to the
  // new values before instructions are recreated with remapped types.
  for (auto &KV : SeedVMap) VMap[KV.first] = KV.second;
  // FieldMaps is keyed by the OLD struct, but operand-patching below may
  // already swap an instruction's aggregate operand to the NEW struct type
  // before we inspect it (ExtractValue/InsertValue read the operand's *current*
  // type, unlike GEP which stores the old source element type). Mirror each
  // entry under the NEW struct key so the field-index shift fires regardless of
  // which type the operand currently carries. (Instructions being processed
  // always still hold OLD-layout indices, so applying the map is correct.)
  DenseMap<StructType *, std::vector<int>> CombinedFM;
  const DenseMap<StructType *, std::vector<int>> *FM = FieldMaps;
  if (FieldMaps && !FieldMaps->empty()) {
    CombinedFM = *FieldMaps;
    for (auto &KV : *FieldMaps) {
      auto It = TypeMap.find(KV.first);
      if (It != TypeMap.end())
        if (auto *NewST = dyn_cast<StructType>(It->second))
          CombinedFM[NewST] = KV.second;
    }
    FM = &CombinedFM;
  }
  // Walk in BB-then-instruction order. Mutate one inst at a time;
  // RAUW old → new and erase the old. New insts inherit the body's
  // remapped operand types.
  for (BasicBlock &BB : *F) {
    for (auto It = BB.begin(); It != BB.end(); ) {
      Instruction *I = &*It++;
      // Patch operands first (so when we recreate the instruction,
      // operands already point to the substituted values).
      for (Use &U : I->operands()) {
        Value *V = U.get();
        if (auto *MV = dyn_cast_or_null<Value>(VMap.lookup(V))) {
          U.set(MV);
        } else if (auto *CE = dyn_cast<ConstantExpr>(V)) {
          // ConstantExpr operands (e.g. `bitcast(@g to [N x %S.pad]*)`) aren't
          // in VMap; remap their types so they no longer name the stale struct.
          Constant *NC = remapConstantType(CE, TypeMap, Ctx, FM);
          if (NC != CE) U.set(NC);
        }
      }
      Instruction *NewI = nullptr;
      if (auto *AI = dyn_cast<AllocaInst>(I)) {
        Type *NewElTy = remapTypeRecursive(AI->getAllocatedType(), TypeMap, Ctx);
        if (NewElTy != AI->getAllocatedType()) {
          IRBuilder<> B(AI);
          AllocaInst *N = B.CreateAlloca(NewElTy, AI->getArraySize(),
                                         AI->getName());
          N->setAlignment(Align(AI->getAlignment()));
          NewI = N;
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        Type *OldSrcTy = GEP->getSourceElementType();
        // The GEP source element type MUST equal the (already operand-patched)
        // pointer operand's pointee type — IRBuilder asserts this. Derive it
        // from the pointer rather than recomputing via remapTypeRecursive,
        // which can disagree after a field-index shift and trip the assert.
        Type *NewSrcTy = remapTypeRecursive(OldSrcTy, TypeMap, Ctx);
        if (auto *PtrTy =
                dyn_cast<PointerType>(GEP->getPointerOperand()->getType()))
          NewSrcTy = PtrTy->getElementType();
        Type *NewResultPtrTy = remapTypeRecursive(GEP->getType(), TypeMap, Ctx);
        bool typeChanged = (NewSrcTy != OldSrcTy) ||
                           (NewResultPtrTy != GEP->getType());
        // Field-strip index remap: when a struct in the index chain lost
        // fields, shift constant field selectors to surviving positions (a
        // stripped [0 x T] field shares its address with the next real field).
        SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
        bool idxChanged = false, skip = false;
        if ((FM && !FM->empty()) || AnyDissolved) {
          SmallVector<Value *, 8> NI;
          Type *CurT = OldSrcTy;
          for (unsigned k = 0; k < Indices.size(); ++k) {
            Value *Idx = Indices[k];
            if (k == 0) { NI.push_back(Idx); continue; } // pointer walk
            if (auto *ST = dyn_cast<StructType>(CurT)) {
              if (IsDissolved(ST)) {
                // Field selector into a dissolved wrapper: must be the (only)
                // field 0; the index disappears (the wrapper IS its field).
                auto *CIdx = dyn_cast<ConstantInt>(Idx);
                if (!CIdx || !CIdx->isZero() || ST->getNumElements() != 1) {
                  skip = true;
                  break;
                }
                CurT = ST->getElementType(0);
                idxChanged = true;
                continue; // do NOT push the index
              }
              auto It = FM ? FM->find(ST) : DenseMap<StructType *, std::vector<int>>::const_iterator();
              if (FM && It != FM->end()) {
                auto *CIdx = dyn_cast<ConstantInt>(Idx);
                if (!CIdx) { skip = true; break; }
                int OldI = (int)CIdx->getZExtValue();
                const auto &Map = It->second;
                if (OldI < 0 || OldI >= (int)Map.size()) { skip = true; break; }
                int NewIx = Map[OldI];
                if (NewIx == -1) {
                  int NextReal = -1, NextOld = -1;
                  for (int j = OldI + 1; j < (int)Map.size(); ++j)
                    if (Map[j] != -1) { NextReal = Map[j]; NextOld = j; break; }
                  if (NextReal == -1) { skip = true; break; }
                  NI.push_back(ConstantInt::get(Idx->getType(), (uint64_t)NextReal));
                  // Field TYPE must be fetched at its OLD position in the OLD
                  // struct (NextReal is the NEW index — using it here would
                  // desync CurT and yield an illegal index chain).
                  Type *NextT = ST->getElementType((unsigned)NextOld);
                  bool NextIsAgg = isa<ArrayType>(NextT) || isa<VectorType>(NextT);
                  if (!NextIsAgg && k + 1 < Indices.size()) k++;
                  CurT = NextT; idxChanged = true; continue;
                }
                NI.push_back(ConstantInt::get(Idx->getType(), (uint64_t)NewIx));
                if (NewIx != OldI) idxChanged = true;
                // Same field, OLD position gives the correct type.
                CurT = ST->getElementType((unsigned)OldI);
                continue;
              }
            }
            NI.push_back(Idx);
            if (auto *ST2 = dyn_cast<StructType>(CurT)) {
              auto *CIdx = dyn_cast<ConstantInt>(Idx);
              if (!CIdx) { skip = true; break; }
              CurT = ST2->getElementType(CIdx->getZExtValue());
            } else if (auto *AT = dyn_cast<ArrayType>(CurT)) {
              CurT = AT->getElementType();
            } else if (auto *VT = dyn_cast<VectorType>(CurT)) {
              CurT = VT->getElementType();
            } else { skip = true; break; }
          }
          if (!skip && idxChanged) Indices.assign(NI.begin(), NI.end());
        }
        if (!skip && (typeChanged || idxChanged)) {
          // Validate the (possibly index-shifted) walk against the remapped
          // source type before building — an invalid index chain makes
          // getIndexedType() return null and PointerType::get() assert.
          Type *Indexed =
              GetElementPtrInst::getIndexedType(NewSrcTy, Indices);
          if (!Indexed) {
            SmallVector<Value *, 8> OrigIdx(GEP->idx_begin(), GEP->idx_end());
            if (GetElementPtrInst::getIndexedType(NewSrcTy, OrigIdx)) {
              Indices.assign(OrigIdx.begin(), OrigIdx.end());
              Indexed = GetElementPtrInst::getIndexedType(NewSrcTy, Indices);
            }
          }
          if (Indexed) {
            IRBuilder<> B(GEP);
            Value *N = B.CreateInBoundsGEP(NewSrcTy, GEP->getPointerOperand(),
                                           Indices, GEP->getName());
            NewI = dyn_cast<Instruction>(N);
          } else {
            vxxDbg() << "vxx: GEP remap could not validate indices for "
                   << "NewSrcTy=";
            NewSrcTy->print(vxxDbg());
            vxxDbg() << " idxChanged=" << idxChanged << " nIdx=" << Indices.size()
                   << " in " << F->getName() << "\n";
          }
        }
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        // Loaded type must equal the (operand-patched) pointer's pointee.
        Type *NewTy = remapTypeRecursive(LI->getType(), TypeMap, Ctx);
        if (auto *PtrTy =
                dyn_cast<PointerType>(LI->getPointerOperand()->getType()))
          NewTy = PtrTy->getElementType();
        if (NewTy != LI->getType()) {
          IRBuilder<> B(LI);
          LoadInst *N = B.CreateLoad(NewTy, LI->getPointerOperand(),
                                     LI->getName());
          N->setAlignment(Align(LI->getAlignment()));
          N->setVolatile(LI->isVolatile());
          NewI = N;
        }
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {
        Type *NewDestTy = remapTypeRecursive(BC->getDestTy(), TypeMap, Ctx);
        if (NewDestTy != BC->getDestTy()) {
          IRBuilder<> B(BC);
          Value *N = B.CreateBitCast(BC->getOperand(0), NewDestTy,
                                     BC->getName());
          NewI = dyn_cast<Instruction>(N);
        }
      } else if (auto *CI = dyn_cast<CallInst>(I)) {
        // For overloaded LLVM intrinsics whose name encodes a struct
        // type we just substituted (e.g. `llvm.fpga.fifo.pop.sl_*.p0sl_*`),
        // we need a NEW intrinsic decl referencing the new type.
        Function *Callee = CI->getCalledFunction();
        if (Callee && Callee->isIntrinsic() && Callee->getName().contains("fpga.fifo.")) {
          // Check if any arg type or result type would change after remap.
          bool NeedsRetype = false;
          Type *NewRetTy = remapTypeRecursive(CI->getType(), TypeMap, Ctx);
          if (NewRetTy != CI->getType()) NeedsRetype = true;
          SmallVector<Type *, 2> NewArgTys;
          for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
            Type *T = remapTypeRecursive(CI->getArgOperand(i)->getType(),
                                          TypeMap, Ctx);
            NewArgTys.push_back(T);
            if (T != CI->getArgOperand(i)->getType()) NeedsRetype = true;
          }
          if (NeedsRetype) {
            Module *M = CI->getParent()->getParent()->getParent();
            // Build new intrinsic name from the new types via existing
            // `mangleForIntrinsic` helper. The intrinsic signature for
            // pop is `<T> @llvm.fpga.fifo.pop.<T>.p0<T>(<T>* %ptr)`;
            // for push it's `void @llvm.fpga.fifo.push.<T>.p0<T>(<T>, <T>*)`.
            // We rely on `getOrInsertFifoPopAny`/`PushAny` which mangles
            // exactly this way.
            bool IsPop = Callee->getName().contains(".pop.");
            Type *ElTy = IsPop ? NewRetTy : NewArgTys[0];
            Function *NewCallee = IsPop
                ? getOrInsertFifoPopAny(*M, ElTy)
                : getOrInsertFifoPushAny(*M, ElTy);
            IRBuilder<> B(CI);
            SmallVector<Value *, 2> Args(CI->arg_begin(), CI->arg_end());
            // The remap may retype the stream pointer to a struct-element
            // pointer (`Stream<dat_t<u32>>`: the elem stays i32 but the
            // pointer becomes dat_t*) — cast each arg to the callee's param
            // type or llvm-as rejects the ill-typed call.
            FunctionType *NFT = NewCallee->getFunctionType();
            for (unsigned ai = 0; ai < Args.size() && ai < NFT->getNumParams(); ++ai)
              if (Args[ai]->getType() != NFT->getParamType(ai) &&
                  Args[ai]->getType()->isPointerTy() &&
                  NFT->getParamType(ai)->isPointerTy())
                Args[ai] = B.CreateBitCast(Args[ai], NFT->getParamType(ai));
            CallInst *N = B.CreateCall(NewCallee, Args, CI->getName());
            NewI = N;
          }
        }
        // Call whose callee is a constexpr `bitcast(@fn to oldFnTy*)` left by
        // the signature rebuild — its target fn type still names the stale
        // `.pad` struct. Rebuild the call against the remapped fn type so the
        // call's stored FunctionType no longer references the old struct
        // (LLVM 7 llvm-as rejects the constant-expression type mismatch).
        if (!NewI) {
          if (auto *CE = dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
            if (CE->getOpcode() == Instruction::BitCast) {
              Type *NewPtrTy = remapTypeRecursive(CE->getType(), TypeMap, Ctx);
              if (NewPtrTy != CE->getType()) {
                auto *NewFnPtrTy = cast<PointerType>(NewPtrTy);
                auto *NewFnTy =
                    cast<FunctionType>(NewFnPtrTy->getElementType());
                Constant *Underlying = CE->getOperand(0); // already-remapped fn
                Value *NewCallee =
                    Underlying->getType() == NewFnPtrTy
                        ? static_cast<Value *>(Underlying)
                        : ConstantExpr::getBitCast(Underlying, NewFnPtrTy);
                IRBuilder<> B(CI);
                SmallVector<Value *, 8> Args(CI->arg_begin(), CI->arg_end());
                CallInst *N =
                    B.CreateCall(NewFnTy, NewCallee, Args, CI->getName());
                N->setCallingConv(CI->getCallingConv());
                N->setAttributes(CI->getAttributes());
                NewI = N;
              }
            }
          }
        }
      } else if (auto *EV = dyn_cast<ExtractValueInst>(I)) {
        // By-value aggregate read (e.g. extractvalue %BurstMaxi, 1). Remap
        // the result type and shift struct field selectors via FieldMaps.
        Type *AggTy = EV->getAggregateOperand()->getType();
        // Dissolved single-field wrapper: `extractvalue %dat_t %v, 0` — the
        // wrapper IS its field, so the read is the (already remapped)
        // aggregate operand itself.
        if (AnyDissolved &&
            (IsDissolved(AggTy) || !AggTy->isAggregateType()) &&
            EV->getNumIndices() == 1 && *EV->idx_begin() == 0) {
          Value *Rep = EV->getAggregateOperand();
          while (!EV->use_empty()) { Use &U2 = *EV->use_begin(); U2.set(Rep); }
          EV->eraseFromParent();
          continue;
        }
        SmallVector<unsigned, 4> Idxs(EV->idx_begin(), EV->idx_end());
        bool changed = (remapTypeRecursive(AggTy, TypeMap, Ctx) != AggTy);
        bool skipEV = false, deadEV = false;
        if (FM && !FM->empty()) {
          Type *CurT = AggTy;
          for (unsigned &Ix : Idxs) {
            if (auto *ST = dyn_cast<StructType>(CurT)) {
              auto It = FM->find(ST);
              if (It != FM->end()) {
                const auto &Map = It->second;
                if ((int)Ix >= (int)Map.size()) { skipEV = true; break; }
                if (Map[Ix] == -1) { deadEV = true; break; }
                Type *NextT = ST->getElementType(Ix);
                Ix = (unsigned)Map[Ix]; changed = true; CurT = NextT; continue;
              }
              CurT = ST->getElementType(Ix);
            } else if (auto *AT = dyn_cast<ArrayType>(CurT)) {
              CurT = AT->getElementType();
            } else { skipEV = true; break; }
          }
        }
        if (deadEV) {
          // Read of a stripped (zero-size pad) field — it carries no data, so
          // the surviving struct has no such field. Replace with undef of the
          // remapped result type (rustc's by-value FCA split emits these reads
          // for the dropped pad, e.g. `extractvalue %burst_maxi, 1`).
          Value *U =
              UndefValue::get(remapTypeRecursive(EV->getType(), TypeMap, Ctx));
          while (!EV->use_empty()) { Use &U2 = *EV->use_begin(); U2.set(U); }
          EV->eraseFromParent();
        } else if (!skipEV && changed) {
          IRBuilder<> B(EV);
          Value *N = B.CreateExtractValue(EV->getAggregateOperand(), Idxs,
                                          EV->getName());
          NewI = dyn_cast<Instruction>(N);
        }
      } else if (auto *IV = dyn_cast<InsertValueInst>(I)) {
        Type *AggTy = IV->getAggregateOperand()->getType();
        // Dissolved single-field wrapper: `insertvalue %dat_t undef, %x, 0`
        // — the wrapper IS its field, so the result is just the inserted
        // scalar.
        if (AnyDissolved &&
            (IsDissolved(AggTy) || !AggTy->isAggregateType()) &&
            IV->getNumIndices() == 1 && *IV->idx_begin() == 0) {
          Value *Rep = IV->getInsertedValueOperand();
          while (!IV->use_empty()) { Use &U2 = *IV->use_begin(); U2.set(Rep); }
          IV->eraseFromParent();
          continue;
        }
        SmallVector<unsigned, 4> Idxs(IV->idx_begin(), IV->idx_end());
        bool changed = (remapTypeRecursive(AggTy, TypeMap, Ctx) != AggTy);
        bool skipIV = false, deadIV = false;
        if (FM && !FM->empty()) {
          Type *CurT = AggTy;
          for (unsigned &Ix : Idxs) {
            if (auto *ST = dyn_cast<StructType>(CurT)) {
              auto It = FM->find(ST);
              if (It != FM->end()) {
                const auto &Map = It->second;
                if ((int)Ix >= (int)Map.size()) { skipIV = true; break; }
                if (Map[Ix] == -1) { deadIV = true; break; }
                Type *NextT = ST->getElementType(Ix);
                Ix = (unsigned)Map[Ix]; changed = true; CurT = NextT; continue;
              }
              CurT = ST->getElementType(Ix);
            } else if (auto *AT = dyn_cast<ArrayType>(CurT)) {
              CurT = AT->getElementType();
            } else { skipIV = true; break; }
          }
        }
        if (deadIV) {
          // Insert into a stripped (zero-size pad) field — the field is gone in
          // the surviving struct, so the insert is a no-op: forward the
          // aggregate operand to all uses.
          Value *Agg = IV->getAggregateOperand();
          while (!IV->use_empty()) { Use &U2 = *IV->use_begin(); U2.set(Agg); }
          IV->eraseFromParent();
        } else if (!skipIV && changed) {
          IRBuilder<> B(IV);
          Value *N = B.CreateInsertValue(IV->getAggregateOperand(),
                                         IV->getInsertedValueOperand(), Idxs,
                                         IV->getName());
          NewI = dyn_cast<Instruction>(N);
        }
      } else if (isa<PHINode>(I) || isa<SelectInst>(I)) {
        // PHI/Select results equal their incoming/operand types. We can't
        // recreate a PHI in place (addIncoming type-checks against operands
        // that may not be remapped yet on a loop back-edge), so mutate the
        // result type directly — the operands are redirected to their remapped
        // defs by the use-redirect below as those defs are recreated, so the
        // IR converges. Without this, a struct-pointer PHI keeps the OLD
        // (`.pad`) type and every GEP/icmp that derives its type from the PHI
        // inherits the stale type (LLVM 11 tolerates it; LLVM 7 llvm-as does
        // not — e.g. `using_vectors` loop induction over `[N x %Vector]`).
        Type *NewTy = remapTypeRecursive(I->getType(), TypeMap, Ctx);
        if (NewTy != I->getType())
          I->mutateType(NewTy);
      }
      if (NewI) {
        // The replacement has a remapped (different) type, which
        // replaceAllUsesWith() forbids (asserts New->getType()==getType()).
        // Redirect each use directly via Use::set (no type check) — the
        // consuming instructions are themselves recreated with remapped types
        // later in this same walk, so the IR converges to a consistent state.
        while (!I->use_empty()) {
          Use &U = *I->use_begin();
          U.set(NewI);
        }
        VMap[I] = NewI;
        I->eraseFromParent();
      }
    }
  }
}

Constant *remapConstantType(Constant *C,
                                   const DenseMap<Type *, Type *> &TypeMap,
                                   LLVMContext &Ctx,
                                   const DenseMap<StructType *, std::vector<int>> *FieldMaps) {
  if (!C) return C;
  Type *NewTy = remapTypeRecursive(C->getType(), TypeMap, Ctx);
  if (auto *CAZ = dyn_cast<ConstantAggregateZero>(C))
    return NewTy != C->getType() ? ConstantAggregateZero::get(NewTy) : C;
  if (isa<UndefValue>(C))
    return NewTy != C->getType() ? UndefValue::get(NewTy) : C;
  if (auto *CPN = dyn_cast<ConstantPointerNull>(C))
    return NewTy != C->getType()
               ? ConstantPointerNull::get(cast<PointerType>(NewTy))
               : C;
  if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    // Dissolved single-field wrapper: `%dat_t { i32 X }` becomes `i32 X`.
    if (!isa<StructType>(NewTy) && CS->getNumOperands() == 1)
      return remapConstantType(cast<Constant>(CS->getOperand(0)), TypeMap, Ctx,
                               FieldMaps);
    // If this struct lost fields, keep only the surviving-field operands.
    const std::vector<int> *FM = nullptr;
    if (FieldMaps) {
      auto It = FieldMaps->find(CS->getType());
      if (It != FieldMaps->end()) FM = &It->second;
    }
    SmallVector<Constant *, 8> Ops;
    for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
      if (FM && i < FM->size() && (*FM)[i] == -1)
        continue; // stripped field — drop its initializer operand
      Ops.push_back(remapConstantType(cast<Constant>(CS->getOperand(i)), TypeMap, Ctx, FieldMaps));
    }
    auto *NewST = cast<StructType>(remapTypeRecursive(CS->getType(), TypeMap, Ctx));
    return ConstantStruct::get(NewST, Ops);
  }
  if (auto *CA = dyn_cast<ConstantArray>(C)) {
    SmallVector<Constant *, 16> Ops;
    for (unsigned i = 0; i < CA->getNumOperands(); ++i)
      Ops.push_back(remapConstantType(cast<Constant>(CA->getOperand(i)), TypeMap, Ctx, FieldMaps));
    Type *NewEl = Ops.empty() ? remapTypeRecursive(CA->getType()->getElementType(), TypeMap, Ctx)
                              : Ops[0]->getType();
    return ConstantArray::get(ArrayType::get(NewEl, Ops.size()), Ops);
  }
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    // Constant-expression casts/GEPs whose type or operands name a remapped
    // struct (e.g. `bitcast([2 x %S]* @g to [2 x %S.pad]*)` left over after the
    // old struct was renamed). Rebuild with remapped operands; for a bitcast,
    // retarget to the remapped destination type (collapsing to the operand
    // itself when the cast becomes an identity).
    SmallVector<Constant *, 4> Ops;
    bool changed = false;
    for (unsigned i = 0; i < CE->getNumOperands(); ++i) {
      Constant *Op = cast<Constant>(CE->getOperand(i));
      Constant *NewOp = remapConstantType(Op, TypeMap, Ctx, FieldMaps);
      Ops.push_back(NewOp);
      if (NewOp != Op) changed = true;
    }
    if (CE->getOpcode() == Instruction::BitCast) {
      Type *NewDest = remapTypeRecursive(CE->getType(), TypeMap, Ctx);
      if (changed || NewDest != CE->getType()) {
        if (Ops[0]->getType() == NewDest)
          return Ops[0]; // identity bitcast after remap
        return ConstantExpr::getBitCast(Ops[0], NewDest);
      }
      return C;
    }
    if (changed)
      return CE->getWithOperands(Ops); // GEP/other: recompute result type
    return C;
  }
  // Scalars / other constants don't carry a remapped struct type.
  return C;
}

void remapStructsInModule(Module &M,
                                 const DenseMap<Type *, Type *> &TypeMapIn,
                                 const DenseMap<StructType *, std::vector<int>> *FieldMaps) {
  if (TypeMapIn.empty()) return;
  LLVMContext &Ctx = M.getContext();

  // Extend the type map with the TRANSITIVE CLOSURE of containing structs.
  // New-struct replacement (unlike in-place setBody) leaves any OTHER struct
  // that references a remapped struct — e.g. `Stream<TransPkt> = { TransPkt }`
  // — still pointing at the OLD struct, because LLVM struct bodies are
  // immutable. We must therefore ALSO replace every identified struct that
  // (transitively) contains a remapped struct, or downstream passes that read
  // the container's element type get the stale (un-stripped `.pad`) struct —
  // the root cause of HLS 214-244 on AXIS struct streams.
  DenseMap<Type *, Type *> TypeMap = TypeMapIn;
  {
    SmallPtrSet<Type *, 16> Shells;
    for (auto &KV : TypeMap) Shells.insert(KV.second);
    SmallVector<StructType *, 16> ClosureOld;
    bool grew = true;
    while (grew) {
      grew = false;
      for (StructType *ST : M.getIdentifiedStructTypes()) {
        if (ST->isOpaque() || TypeMap.count(ST) || Shells.count(ST))
          continue;
        bool refsMapped = false;
        for (Type *FT : ST->elements())
          if (remapTypeRecursive(FT, TypeMap, Ctx) != FT) { refsMapped = true; break; }
        if (!refsMapped)
          continue;
        std::string Nm = ST->hasName() ? ST->getName().str() : std::string();
        if (!Nm.empty())
          ST->setName(Nm + ".old"); // free the name for the replacement
        StructType *Shell = StructType::create(Ctx, Nm); // opaque
        TypeMap[ST] = Shell;
        Shells.insert(Shell);
        ClosureOld.push_back(ST);
        grew = true;
      }
    }
    // Now that every container is registered, body each closure shell with the
    // remapped field types (the original targets were already bodied by the
    // caller via the opaque-shell→setBody two-phase).
    for (StructType *Old : ClosureOld) {
      SmallVector<Type *, 8> NF;
      for (Type *FT : Old->elements())
        NF.push_back(remapTypeRecursive(FT, TypeMap, Ctx));
      cast<StructType>(TypeMap[Old])->setBody(NF, Old->isPacked());
    }
  }

  // 0. Globals FIRST: recreate any global whose value-type or initializer names
  //    an old type, BEFORE remapping function bodies. Function GEPs derive their
  //    source element type from the (operand-patched) pointer; if the global is
  //    still old-typed when a body is remapped, the GEP keeps the stale `.pad`
  //    source and the later global RAUW bakes in a `bitcast(newG, [N x .pad]*)`
  //    that never gets cleaned (root cause of static-ROM 212-361). Doing globals
  //    first means bodies see new globals; the constexpr-operand remap then
  //    collapses the RAUW bitcast to the new-typed global.
  {
    SmallVector<GlobalVariable *, 8> GVs;
    for (GlobalVariable &G : M.globals()) GVs.push_back(&G);
    for (GlobalVariable *G : GVs) {
      Type *NewVT = remapTypeRecursive(G->getValueType(), TypeMap, Ctx);
      Constant *NewInit = nullptr;
      if (G->hasInitializer())
        NewInit = remapConstantType(G->getInitializer(), TypeMap, Ctx, FieldMaps);
      bool changed = (NewVT != G->getValueType()) ||
                     (NewInit && NewInit != G->getInitializer());
      if (!changed) continue;
      auto *NewG = new GlobalVariable(M, NewVT, G->isConstant(), G->getLinkage(),
                                      NewInit, "", G, G->getThreadLocalMode(),
                                      G->getAddressSpace());
      NewG->copyAttributesFrom(G);
      NewG->takeName(G);
      if (!G->use_empty())
        G->replaceAllUsesWith(ConstantExpr::getBitCast(NewG, G->getType()));
      G->eraseFromParent();
    }
  }

  // 1. Functions whose SIGNATURE references an old type: build a new shell with
  //    the remapped FunctionType, move the body over, and remap it.
  SmallPtrSet<Function *, 16> Remapped; // bodies already remapped in step 1
  SmallVector<Function *, 16> Fns;
  for (Function &F : M) Fns.push_back(&F);
  for (Function *F : Fns) {
    FunctionType *FT = F->getFunctionType();
    Type *NewRet = remapTypeRecursive(FT->getReturnType(), TypeMap, Ctx);
    SmallVector<Type *, 8> NewParams;
    bool sigChanged = (NewRet != FT->getReturnType());
    for (Type *P : FT->params()) {
      Type *NP = remapTypeRecursive(P, TypeMap, Ctx);
      NewParams.push_back(NP);
      if (NP != P) sigChanged = true;
    }
    if (!sigChanged) continue;
    FunctionType *NewFT = FunctionType::get(NewRet, NewParams, FT->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                      F->getName() + ".tyremap", &M);
    NewF->copyAttributesFrom(F);
    NewF->setDLLStorageClass(GlobalValue::DefaultStorageClass);
    if (!F->isDeclaration()) {
      NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
      SmallVector<std::pair<Value *, Value *>, 8> Seed;
      auto OA = F->arg_begin();
      auto NA = NewF->arg_begin();
      for (; OA != F->arg_end(); ++OA, ++NA) {
        NA->takeName(&*OA);
        if (OA->getType() == NA->getType())
          OA->replaceAllUsesWith(&*NA);
        else
          Seed.push_back({&*OA, &*NA});
      }
      substituteTypeInFunction(NewF, TypeMap, Seed, FieldMaps);
    }
    if (!F->use_empty())
      F->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, F->getType()));
    std::string Nm = F->getName().str();
    F->eraseFromParent();
    NewF->setName(Nm);
    Remapped.insert(NewF);
  }

  // 2. Remap remaining function bodies. Skip those ALREADY remapped in step 1:
  //    re-running the field-index shift on an instruction whose indices are
  //    already in the NEW layout would shift them a SECOND time (e.g. a GEP
  //    `i32 3` → `i32 1` in step 1 would become `i32 0` here), silently
  //    corrupting struct field access. (CombinedFM now matches the new struct
  //    key, so the old "miss → idempotent" assumption no longer holds.)
  for (Function &F : M)
    if (!F.isDeclaration() && !Remapped.count(&F))
      substituteTypeInFunction(&F, TypeMap, {}, FieldMaps);

  // Final: reconcile call ↔ callee signatures. When a caller is rebuilt in
  // step 1 BEFORE its callee, the caller's call keeps the callee's stale
  // FunctionType (and a `bitcast(newCallee, oldFnTy)` constexpr callee). Since
  // step 2 now skips already-remapped functions (to avoid double field-shift),
  // those stale calls are no longer cleaned there. Rebuild any call whose
  // stored type disagrees with its (bitcast-stripped) callee when the argument
  // types already match the callee — calling the function directly.
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    SmallVector<CallInst *, 8> Fix;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Value *Cal = CI->getCalledOperand();
          if (auto *CE = dyn_cast<ConstantExpr>(Cal))
            if (CE->getOpcode() == Instruction::BitCast)
              Cal = CE->getOperand(0);
          if (auto *F2 = dyn_cast<Function>(Cal))
            if (F2->getFunctionType() != CI->getFunctionType())
              Fix.push_back(CI);
        }
    for (CallInst *CI : Fix) {
      Value *Cal = CI->getCalledOperand();
      if (auto *CE = dyn_cast<ConstantExpr>(Cal))
        if (CE->getOpcode() == Instruction::BitCast)
          Cal = CE->getOperand(0);
      auto *F2 = cast<Function>(Cal);
      FunctionType *FT2 = F2->getFunctionType();
      if (CI->getNumArgOperands() != FT2->getNumParams() || FT2->isVarArg())
        continue;
      bool ok = true;
      for (unsigned i = 0; i < CI->getNumArgOperands(); ++i)
        if (CI->getArgOperand(i)->getType() != FT2->getParamType(i)) { ok = false; break; }
      if (!ok) continue;
      IRBuilder<> B(CI);
      SmallVector<Value *, 8> Args(CI->arg_begin(), CI->arg_end());
      CallInst *N = B.CreateCall(FT2, F2, Args, CI->getName());
      N->setCallingConv(CI->getCallingConv());
      N->setAttributes(CI->getAttributes());
      CI->replaceAllUsesWith(N);
      CI->eraseFromParent();
    }
  }

}

} } // namespace hlsrs::vxx
