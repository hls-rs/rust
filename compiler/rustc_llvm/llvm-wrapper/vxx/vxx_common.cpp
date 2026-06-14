//===----------------------------------------------------------------------===//
//
// vxx_common.cpp — cross-cutting helpers used by the Vitis lowering modules.
//
// Definitions carved out of the monolithic VXXPrep.cpp. These helpers are used
// across the lowering modules; they live in the `hlsrs::vxx` namespace
// (declared in vxx_internal.h).
//
//===----------------------------------------------------------------------===//

#include "vxx_internal.h"
#include "VXXShared.h"

#include <string>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;

namespace hlsrs { namespace vxx {

// Returns true if the struct type is the Rust-side `barista_hls::AxisDisabled`
// placeholder (a 1-byte marker that says "this AXIS sub-channel is disabled
// by the C++ template"). Such fields are skipped when emitting per-channel
// `llvm.fpga.axis.{pop,push}` intrinsics so the channel count matches what
// cosim TB expects for partial-channel `axis_user<W, ...>` / `axis_data<...>`
// variants.
bool isAxisDisabledType(Type *T) {
  auto *ST = dyn_cast<StructType>(T);
  if (!ST || !ST->hasName()) return false;
  StringRef N = ST->getName();
  // Match Rust-original names AND the C++ canonical name (post-rename).
  return N.contains("AxisDisabled") ||
         N.contains("barista_hls::axis::AxisDisabled") ||
         N == "struct.hls::axis_disabled_signal";
}

// Get or insert `llvm.fpga.axis.{pop,push}` declaration with the per-field
// arg-type signature. The intrinsic takes 2N pointer args:
// N for the AXIS stream's per-field channels (data/keep/strb/user/last/id/dest)
// and N for the local temp dest/source pointers, in that order. Type-overloaded
// by the N field types.
//
// `FieldPtrTys` must be exactly 7 pointer types (one per axis field). The
// pop variant writes into the temp pointers (args [N..2N]); push reads from
// them. Stream-side pointers (args [0..N]) are GEPs into the AXIS stream's
// inner struct.hls::axis<...>.
Function *getOrInsertAxisPopOrPush(Module &M, const char *Op,
                                   ArrayRef<Type *> FieldPtrTys) {
  std::string Name = std::string("llvm.fpga.axis.") + Op;
  for (Type *PT : FieldPtrTys)
    Name += "." + hlsrs::vxx::mangleForIntrinsic(PT);
  SmallVector<Type *, 16> ArgTys;
  for (Type *PT : FieldPtrTys) ArgTys.push_back(PT);   // stream side
  for (Type *PT : FieldPtrTys) ArgTys.push_back(PT);   // temp side
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(M.getContext()), ArgTys, /*isVarArg=*/false);
  Function *F = cast<Function>(M.getOrInsertFunction(Name, FT).getCallee());
  // LLVM auto-attaches intrinsic attributes based on the `llvm.fpga.*`
  // name pattern — but the attribute set is built for a specific
  // canonical arg count; with our overloaded variants the attribute
  // indices can overflow ("Attribute after last parameter" llvm-as
  // verifier error). Strip all attributes and re-add the minimum.
  F->setAttributes(AttributeList());
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::ArgMemOnly);
  // `__vxx_axis_packed`: emit canonical axis.pop/push attrs:
  //   `argmemonly nounwind willreturn`, every param `nocapture`, and the
  //   temp-side half (args NCh..2*NCh-1) marked `writeonly` for pop (it writes
  //   the popped value into the local temp) / `readonly` for push (reads the
  //   value to push). The earlier `noalias` on all 14 args (intended to stop a
  //   BasicAA recursion) makes the downstream CorrelatedValuePropagation /
  //   LazyValueInfo SIGSEGV on the helper, so omit noalias here.
  if (hlsrs::vxx::markerUsed(M, "__vxx_axis_packed")) {
    F->addFnAttr(Attribute::WillReturn);
    unsigned NCh = FieldPtrTys.size();
    bool IsPush = (std::string(Op) == "push");
    for (unsigned i = 0; i < F->arg_size(); ++i) {
      F->addParamAttr(i, Attribute::NoCapture);
      // noalias: the 14 channel pointers are distinct AXIS sub-channels that
      // never alias. The downstream GVN/CorrelatedValuePropagation run BasicAA
      // over them; without noalias, AA recurses through the nested struct-ptr
      // pointee types (ap_X<W>={ap_int_base={ssdm_int={iW}}}) → stack-overflow
      // SIGSEGV (HLS 200-1715). In the Rust flow rustc opt inlines read/write
      // before that AA runs, so we keep noalias to make AA terminate.
      // Keep noalias on the channel args: once the @example body is fully
      // field-wise (forwardNarrowCopyBridge) BasicAA may terminate without it,
      // and noalias is what stops CVP/LVI from SIGSEGVing on the helper.
      F->addParamAttr(i, Attribute::NoAlias);
      if (i >= NCh)
        F->addParamAttr(i, IsPush ? Attribute::ReadOnly : Attribute::WriteOnly);
    }
  }
  return F;
}

// Build the mangled name for `llvm.fpga.fifo.{pop,push}` given the element
// integer type. LLVM's overloaded intrinsic naming convention appends the
// integer suffix (e.g. `.i32.p0i32` for a pointer to i32).

// Get or insert `llvm.fpga.fifo.pop.i<N>.p0i<N>` declaration.

// Get or insert `llvm.fpga.fifo.push.i<N>.p0i<N>` declaration.

FunctionCallee getSsdmOp(Module &M, StringRef Name) {
  FunctionType *VarFnTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                                            ArrayRef<Type *>(), true);
  FunctionCallee FC = M.getOrInsertFunction(Name, VarFnTy);
  if (auto *F = dyn_cast<Function>(FC.getCallee()))
    if (!F->hasFnAttribute(Attribute::NoUnwind))
      F->addFnAttr(Attribute::NoUnwind);
  return FC;
}

// Helper: stash a NUL-terminated string in a private global and return
// the array `[N x i8]*` passed to the HLS op-bundles. We
// dedupe by initializer so the same byte sequence is shared across
// every caller, avoiding redundant identical `@.str` globals.
Constant *makeSsdmStr(Module &M, StringRef S) {
  LLVMContext &Ctx = M.getContext();
  Constant *Init = ConstantDataArray::getString(Ctx, S, /*AddNull=*/true);
  for (GlobalVariable &G : M.globals()) {
    if (!G.hasPrivateLinkage() || !G.isConstant())
      continue;
    if (G.getValueType() != Init->getType())
      continue;
    if (G.getInitializer() == Init)
      return &G;
  }
  auto *GV = new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, Init, ".str");
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

// Find the kernel function that contains a `__vxx_*` marker call.
// Returns null if the marker isn't present or has no calls.
Function *singleUserKernel(Module &M, StringRef MarkerName) {
  Function *F = M.getFunction(MarkerName);
  if (!F)
    return nullptr;
  Function *Found = nullptr;
  for (User *U : F->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    Function *P = CI->getParent()->getParent();
    if (Found && Found != P)
      return nullptr;
    Found = P;
  }
  return Found;
}

// Read a NUL-trimmed byte slice referenced by a `*const u8` ConstantExpr
// + `i32 len` pair. Used to recover the bundle name from the unified
// `__vxx_m_axi(ptr, depth, name_ptr, name_len, ...)` marker.
std::string extractByteSlice(Value *Ptr, Value *LenV) {
  while (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
    Ptr = GEP->getPointerOperand();
  while (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->getOpcode() == Instruction::GetElementPtr ||
        CE->getOpcode() == Instruction::BitCast)
      Ptr = CE->getOperand(0);
    else
      break;
  }
  while (auto *BCI = dyn_cast<BitCastInst>(Ptr))
    Ptr = BCI->getOperand(0);
  auto *GV = dyn_cast<GlobalVariable>(Ptr);
  if (!GV || !GV->hasInitializer())
    return std::string();
  Constant *Init = GV->getInitializer();
  if (auto *CS = dyn_cast<ConstantStruct>(Init)) {
    if (CS->getNumOperands() == 1)
      Init = CS->getOperand(0);
  }
  auto *CDA = dyn_cast<ConstantDataArray>(Init);
  if (!CDA || !CDA->getElementType()->isIntegerTy(8))
    return std::string();
  StringRef S = CDA->getRawDataValues();
  if (auto *LC = dyn_cast<ConstantInt>(LenV)) {
    uint64_t L = LC->getZExtValue();
    if (L <= S.size())
      S = S.substr(0, L);
  }
  std::string Out = S.str();
  while (!Out.empty() && Out.back() == '\0')
    Out.pop_back();
  return Out;
}

// Walk through casts/GEPs to find a backing AllocaInst or GlobalVariable.
Value *traceToAllocaOrGlobal(Value *V) {
  SmallPtrSet<Value *, 8> Seen;
  while (V && Seen.insert(V).second) {
    if (isa<AllocaInst>(V) || isa<GlobalVariable>(V)) return V;
    if (auto *BC = dyn_cast<BitCastInst>(V)) { V = BC->getOperand(0); continue; }
    if (auto *AS = dyn_cast<AddrSpaceCastInst>(V)) { V = AS->getOperand(0); continue; }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) { V = GEP->getPointerOperand(); continue; }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      if (CE->getOpcode() == Instruction::BitCast ||
          CE->getOpcode() == Instruction::AddrSpaceCast ||
          CE->getOpcode() == Instruction::GetElementPtr) {
        V = CE->getOperand(0); continue;
      }
    }
    break;
  }
  return V;
}

// Add a module-level Xilinx flag (used for top-level synthesis options).
bool appendXilinxAttribute(Module &M, StringRef AttrName, StringRef AttrVal) {
  NamedMDNode *N = M.getOrInsertNamedMetadata(("xlnx.attr." + AttrName).str());
  Metadata *Ops[] = { MDString::get(M.getContext(), AttrVal) };
  N->addOperand(MDNode::get(M.getContext(), Ops));
  return true;
}

bool dropMarkerDefinition(Module &M, StringRef Name) {
  Function *F = M.getFunction(Name);
  if (!F) return false;
  if (!F->use_empty()) return false;
  F->eraseFromParent();
  return true;
}

bool dropMarkerCallsAndDefinition(Module &M, StringRef Name) {
  Function *F = M.getFunction(Name);
  if (!F) return false;
  bool Changed = false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  for (CallInst *CI : Calls) { CI->eraseFromParent(); Changed = true; }
  if (F->use_empty()) { F->eraseFromParent(); Changed = true; }
  return Changed;
}

// Catch-all: drop any remaining __vxx_* calls + their declarations.
// Must run AFTER all targeted inject* passes so they can extract semantics
// before we delete the markers.
//
// For calls that return a non-void value used downstream, replace all uses
// with an undef of the return type before erasing. Without this, dropping
// `__vxx_maxi_read_i32` (or similar value-returning markers that haven't
// been lowered to a real intrinsic) leaves `<badref>` in dependent
// instructions, crashing the rustc downstream LLVM pipeline.
bool eraseUnimplementedMarkers(Module &M) {
  bool Changed = false;
  SmallVector<Function *, 16> Targets;
  for (Function &F : M)
    if (F.getName().startswith("__vxx_") || F.getName().startswith("__vxxprep_"))
      Targets.push_back(&F);
  for (Function *F : Targets) {
    SmallVector<CallInst *, 8> Calls;
    for (User *U : F->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      Type *RetTy = CI->getType();
      if (!RetTy->isVoidTy() && !CI->use_empty()) {
        CI->replaceAllUsesWith(UndefValue::get(RetTy));
      }
      CI->eraseFromParent();
      Changed = true;
    }
    if (F->use_empty()) { F->eraseFromParent(); Changed = true; }
  }
  return Changed;
}

// =====================================================================
// Generic fn-attribute injector. For marker calls with no positional ptr
// arg, attach AttrKey=AttrVal to the function containing the call.
// =====================================================================
bool injectFnAttr(Module &M, StringRef Marker, StringRef AttrKey, StringRef AttrVal) {
  Function *F = M.getFunction(Marker);
  if (!F) return false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
  bool Changed = false;
  for (CallInst *CI : Calls) {
    Function *Parent = CI->getFunction();
    if (Parent) Parent->addFnAttr(AttrKey, AttrVal);
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) F->eraseFromParent();
  return Changed;
}

std::string vxxParseRustStrSlice(Value *PtrArg, Value *LenArg) {
  std::string S;
  if (!PtrArg) return S;
  Value *V = PtrArg;
  while (auto *BC = dyn_cast<BitCastOperator>(V)) V = BC->getOperand(0);
  if (auto *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::GetElementPtr) V = CE->getOperand(0);
  auto *GV = dyn_cast<GlobalVariable>(V);
  if (!GV || !GV->hasInitializer()) return S;
  Constant *Init = GV->getInitializer();
  if (auto *CS = dyn_cast<ConstantStruct>(Init))
    if (CS->getNumOperands() == 1) Init = CS->getOperand(0);
  auto *CDA = dyn_cast<ConstantDataArray>(Init);
  if (!CDA || !CDA->getElementType()->isIntegerTy(8)) return S;
  unsigned N = CDA->getNumElements();
  if (LenArg) {
    if (auto *LenCI = dyn_cast<ConstantInt>(LenArg)) {
      uint64_t L = LenCI->getZExtValue();
      if (L > 0 && L <= N) N = L;
    }
  }
  for (unsigned i = 0; i < N; ++i)
    S.push_back((char)CDA->getElementAsInteger(i));
  // Trim trailing NUL if string literal.
  while (!S.empty() && S.back() == '\0') S.pop_back();
  return S;
}

} } // namespace hlsrs::vxx
