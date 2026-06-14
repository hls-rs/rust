//===----------------------------------------------------------------------===//
//
// dsp_fft.cpp — DSP and FFT IP passes: complex/SSR arg rename, ap_float / DSP-complex intrinsics, FFT-SSR IP wiring.
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
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <string>

using namespace llvm;
using namespace hlsrs::vxx;

namespace hlsrs { namespace vxx {

// FFT IP complex-stream arg rename.
//
// The FFT examples call a C++ proxy `barista_fft_proxy` whose xn/xk params are
// `hls::stream<std::complex<ap_fixed<W,1>>>&` (so the Vitis frontend establishes
// the stream interface for the FFT IP's get_data/set_data movers — passing a
// plain pointer makes those movers use raw load/store -> SYNCHK 200-91/214-244).
// At the prototype-check stage `hls::stream<>&` does NOT decay to a scalar
// pointer; it stays the named type `%"class.hls::stream<std::complex<...>>, 0>"*`.
// The Rust kernel passes the data as `i32*` (a u32 sample = packed 2x i16 complex),
// so the call-site prototype mismatches the proxy (HLS 214-136).
//
// This pass rewrites the FFT top's complex-sample args (the `i32*` ones passed to
// barista_fft_proxy — `i8*` status is left alone) AND the proxy declaration's
// matching params to the EXACT named stream type, so the linker merges them with
// the proxy's own (compiled from hls_stream.h) and the prototypes match. The args
// are only passed through (offset-0 GEP) to the proxy call, never dereferenced in
// the kernel, so only the pointer type needs rewriting. W = element_bits / 2
// (i32 sample = complex of two i16 = ap_fixed<16,1>).
// Narrow the pure-port complex argument arrays. The Rust source declares
// `Complex<ApFixed<W,1>>` / `Complex<f32>` (the C++ typedef mirror), but
// ApInt keeps its wide i128 storage in the type layout, so an
// `&[CmpxDataIn; N]` arg arrives as `[N x { i128, i128 }]*` (fixed) or a
// flat `[N x { float, float }]*` (float). C++ pre-reflow carries
// `class.std::complex<ap_fixed<W,1>>` / nested `struct.std::complex<float>`
// elements. Clone every function that calls the FFT proxy with a TYPE
// REMAPPER that substitutes the C++ element type everywhere — leaving no
// wide-type residue (a bitcast-placeholder variant left `{i128,i128}`
// intermediates that crashed reflow's elaborate). Sound because the FFT
// kernels only FORWARD these pointers (never load/store elements — a
// wide element access would change size under the remap). W comes from
// the proxy's mangled name.
namespace {
struct CmpxNarrowMapper : public ValueMapTypeRemapper {
  Type *Wide;
  Type *Cmpx;
  DenseMap<Type *, Type *> Cache;
  CmpxNarrowMapper(Type *W, Type *C) : Wide(W), Cmpx(C) {}
  Type *remapType(Type *T) override {
    if (T == Wide) return Cmpx;
    auto It = Cache.find(T);
    if (It != Cache.end()) return It->second;
    Type *R = T;
    if (auto *PT = dyn_cast<PointerType>(T)) {
      Type *E = remapType(PT->getElementType());
      if (E != PT->getElementType())
        R = PointerType::get(E, PT->getAddressSpace());
    } else if (auto *AT = dyn_cast<ArrayType>(T)) {
      Type *E = remapType(AT->getElementType());
      if (E != AT->getElementType()) R = ArrayType::get(E, AT->getNumElements());
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      bool Any = false;
      SmallVector<Type *, 8> Ps;
      for (Type *P : FT->params()) {
        Type *NP = remapType(P);
        Any |= (NP != P);
        Ps.push_back(NP);
      }
      Type *RT = remapType(FT->getReturnType());
      Any |= (RT != FT->getReturnType());
      if (Any) R = FunctionType::get(RT, Ps, FT->isVarArg());
    }
    Cache[T] = R;
    return R;
  }
};
} // namespace

// FIR IP glue: retype the kernel's ApFixed/ApUint pointer args to the plain
// iW form the C++ proxy prototype uses. Rust `[ApFixed<W,W>; N]` args lower
// to `[N x i128]` and `barista_fir_run(i128*, i128*, i128*)`, but the proxy
// TU (design/fir_proxy.cpp, OSS clang) is `barista_fir_run(ap_fixed<16,16>*,
// ap_fixed<24,24>*, ap_uint<8>*)` — at the pp level plain (i16*, i24*, i8*)
// (the C++ baseline's own fir_top is `([20 x i16]*, [10 x i24]*, i8*)`).
// The width triple comes from the `__vxx_fir_retype(iw, ow, cw)` marker the
// generated fir_glue.rs places next to the run call. Narrowing is also what
// Stage A needs (16/24-bit BRAM ports, not 128).
bool narrowFirApFixedArgs(Module &M) {
  Function *Marker = M.getFunction("__vxx_fir_retype");
  if (!Marker) return false;
  SmallVector<CallInst *, 2> MCalls;
  for (User *U : Marker->users())
    if (auto *CI = dyn_cast<CallInst>(U)) MCalls.push_back(CI);
  if (MCalls.empty()) {
    hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_fir_retype");
    return false;
  }
  unsigned IW = 0, OW = 0, CW = 0;
  {
    auto *A = dyn_cast<ConstantInt>(MCalls[0]->getArgOperand(0));
    auto *B = dyn_cast<ConstantInt>(MCalls[0]->getArgOperand(1));
    auto *C = dyn_cast<ConstantInt>(MCalls[0]->getArgOperand(2));
    if (!A || !B || !C) return false;
    IW = A->getZExtValue();
    OW = B->getZExtValue();
    CW = C->getZExtValue();
  }
  Function *RunDecl = M.getFunction("barista_fir_run");
  if (!RunDecl || IW == 0 || OW == 0 || CW == 0) return false;
  LLVMContext &Ctx = M.getContext();
  // The proxy TU (OSS clang) keeps the extern-"C" prototype in NAMED STRUCT
  // form: barista_fir_run(%"struct.ap_fixed<16, 16>"*, %"struct.ap_fixed<24,
  // 24>"*, %"struct.ap_uint<8>"*) — replicate the exact 3-level nesting so
  // the Vitis link unifies the types by name (the FFT retype precedent).
  auto namedTy = [&](const std::string &Nm, ArrayRef<Type *> Fl) -> StructType * {
    StructType *T = M.getTypeByName(Nm);
    if (!T) T = StructType::create(Ctx, Fl, Nm);
    return T;
  };
  auto apFixedTy = [&](unsigned W) -> StructType * {
    std::string ws = std::to_string(W);
    StructType *S =
        namedTy("struct.ssdm_int<" + ws + ", true>", {IntegerType::get(Ctx, W)});
    StructType *B = namedTy("struct.ap_fixed_base<" + ws + ", " + ws + ">", {S});
    return namedTy("struct.ap_fixed<" + ws + ", " + ws + ">", {B});
  };
  auto apUintTy = [&](unsigned W) -> StructType * {
    std::string ws = std::to_string(W);
    StructType *S = namedTy("struct.ssdm_int<" + ws + ", false>",
                            {IntegerType::get(Ctx, W)});
    StructType *B = namedTy("struct.ap_int_base<" + ws + ", false>", {S});
    return namedTy("struct.ap_uint<" + ws + ">", {B});
  };
  Type *WTys[3] = {apFixedTy(IW), apFixedTy(OW), apUintTy(CW)};
  // Map a widened type to its narrow counterpart for run-operand slot K.
  auto narrowPtrTy = [&](Type *OldPtr, unsigned K) -> PointerType * {
    auto *PT = dyn_cast<PointerType>(OldPtr);
    if (!PT) return nullptr;
    Type *E = PT->getElementType();
    if (auto *AT = dyn_cast<ArrayType>(E)) {
      if (!AT->getElementType()->isIntegerTy(128)) return nullptr;
      return ArrayType::get(WTys[K], AT->getNumElements())
          ->getPointerTo(PT->getAddressSpace());
    }
    // Wide scalar forms: i128 (the ApFixed lowering) or i8 (the cosim
    // adapter's byte-pointer marshalling) — both retype to the struct ptr.
    if (!E->isIntegerTy(128) && !E->isIntegerTy(8)) return nullptr;
    return WTys[K]->getPointerTo(PT->getAddressSpace());
  };
  // New (narrow) run declaration.
  FunctionType *NewRunFT = FunctionType::get(
      Type::getVoidTy(Ctx),
      {WTys[0]->getPointerTo(), WTys[1]->getPointerTo(),
       WTys[2]->getPointerTo()},
      false);
  // Canonicalise every caller FIRST (the cpp_proxy adapter rust_<top> is not
  // a top fn, so its args reach the run call through raw allocas/copies that
  // the argument trace below cannot follow), THEN re-find the calls.
  {
    SmallPtrSet<Function *, 2> Callers;
    for (User *U : RunDecl->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Callers.insert(CI->getFunction());
    if (Callers.empty()) return false;
    for (Function *F : Callers) {
      legacy::FunctionPassManager FPM(&M);
      FPM.add(createSROAPass());
      FPM.add(createPromoteMemoryToRegisterPass());
      FPM.add(createInstructionCombiningPass());
      FPM.doInitialization();
      FPM.run(*F);
      FPM.doFinalization();
    }
  }
  SmallVector<CallInst *, 2> RunCalls;
  for (User *U : RunDecl->users())
    if (auto *CI = dyn_cast<CallInst>(U)) RunCalls.push_back(CI);
  if (RunCalls.empty()) return false;
  std::string RunName = RunDecl->getName().str();
  RunDecl->setName(RunName + ".wide.dead");
  Function *NewRun =
      Function::Create(NewRunFT, RunDecl->getLinkage(), RunName, &M);
  NewRun->copyAttributesFrom(RunDecl);

  bool Changed = false;
  bool AnySkipped = false;
  for (CallInst *CI : RunCalls) {
    Function *F = CI->getFunction();
    // Trace each run operand back to its root Argument through GEPs/casts.
    Argument *RootArg[3] = {nullptr, nullptr, nullptr};
    for (unsigned K = 0; K < 3; ++K) {
      Value *V = CI->getArgOperand(K);
      while (true) {
        V = V->stripPointerCasts();
        if (auto *G = dyn_cast<GetElementPtrInst>(V)) {
          V = G->getPointerOperand();
          continue;
        }
        break;
      }
      RootArg[K] = dyn_cast<Argument>(V);
    }
    if (!RootArg[0] || !RootArg[1] || !RootArg[2]) {
      vxxDbg() << "vxx: narrowFir: run operand root is not an Argument in "
               << F->getName() << " — skipped\n";
      AnySkipped = true;
      continue;
    }
    // Build the retyped caller signature.
    SmallVector<Type *, 8> NewParams;
    DenseMap<Argument *, Type *> ArgNewTy;
    bool Bad = false;
    for (unsigned K = 0; K < 3; ++K) {
      PointerType *NT = narrowPtrTy(RootArg[K]->getType(), K);
      if (!NT) { Bad = true; break; }
      ArgNewTy[RootArg[K]] = NT;
    }
    if (Bad) {
      vxxDbg() << "vxx: narrowFir: unexpected arg type in " << F->getName()
               << " — skipped\n";
      AnySkipped = true;
      continue;
    }
    for (Argument &A : F->args()) {
      auto It = ArgNewTy.find(&A);
      NewParams.push_back(It != ArgNewTy.end() ? It->second : A.getType());
    }
    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParams, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(), "", &M);
    NewF->copyAttributesFrom(F);
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &MD : MDs) NewF->addMetadata(MD.first, *MD.second);
    NewF->getBasicBlockList().splice(NewF->begin(), F->getBasicBlockList());
    // Wire the args across; rebuild GEP chains for the retyped ones. The run
    // call's operands are RECORDED (NewOps) and the call rebuilt afterwards;
    // the superseded old chain is erased once the old call is gone.
    auto NI = NewF->arg_begin();
    bool Aborted = false;
    DenseMap<Value *, Value *> NewOpFor; // old CI operand -> narrow value
    SmallVector<Instruction *, 8> OldChain;
    for (auto OI = F->arg_begin(); OI != F->arg_end(); ++OI, ++NI) {
      NI->setName(OI->getName());
      if (OI->getType() == NI->getType()) {
        OI->replaceAllUsesWith(&*NI);
        continue;
      }
      std::function<void(Value *, Value *)> Rewire = [&](Value *OldV,
                                                         Value *NewV) {
        SmallVector<User *, 8> Users(OldV->user_begin(), OldV->user_end());
        for (User *U : Users) {
          if (auto *G = dyn_cast<GetElementPtrInst>(U)) {
            if (G->getPointerOperand() != OldV) { Aborted = true; return; }
            SmallVector<Value *, 4> Idx(G->idx_begin(), G->idx_end());
            IRBuilder<> B(G);
            Value *NG = B.CreateInBoundsGEP(
                cast<PointerType>(NewV->getType())->getElementType(), NewV,
                Idx, G->getName());
            OldChain.push_back(G);
            Rewire(G, NG);
            continue;
          }
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            // Collapse the wide plumbing cast (…* -> i128*): the narrow value
            // IS the final operand type, no bridging cast wanted.
            Type *DE = cast<PointerType>(BC->getDestTy())->getElementType();
            if (DE->isIntegerTy(128)) {
              OldChain.push_back(BC);
              Rewire(BC, NewV);
              continue;
            }
            IRBuilder<> B(BC);
            Value *NB = B.CreateBitCast(NewV, BC->getDestTy(), BC->getName());
            OldChain.push_back(BC);
            Rewire(BC, NB);
            continue;
          }
          if (auto *UC = dyn_cast<CallInst>(U)) {
            if (UC == CI) {
              NewOpFor[OldV] = NewV; // rebuilt below
              continue;
            }
            // Marker/spec calls tolerate any pointer type via bitcast.
            IRBuilder<> B(UC);
            Value *Cast = B.CreateBitCast(NewV, OldV->getType());
            UC->replaceUsesOfWith(OldV, Cast);
            continue;
          }
          Aborted = true;
          return;
        }
      };
      Rewire(&*OI, &*NI);
      if (Aborted) break;
    }
    if (Aborted) {
      vxxDbg() << "vxx: narrowFir: unhandled use shape in " << F->getName()
               << " — reverting this caller\n";
      F->getBasicBlockList().splice(F->begin(), NewF->getBasicBlockList());
      NewF->eraseFromParent();
      AnySkipped = true;
      continue;
    }
    // Rebuild the run call against the narrow declaration from the recorded
    // narrow operands, then retire the old call + old chain.
    {
      IRBuilder<> B(CI);
      SmallVector<Value *, 3> Ops;
      for (unsigned K = 0; K < 3; ++K) {
        Value *V = CI->getArgOperand(K);
        auto It = NewOpFor.find(V);
        if (It != NewOpFor.end()) V = It->second;
        Type *Want = NewRunFT->getParamType(K);
        if (V->getType() != Want) V = B.CreateBitCast(V, Want);
        Ops.push_back(V);
      }
      CallInst *NewCI = B.CreateCall(NewRun, Ops);
      NewCI->setCallingConv(CI->getCallingConv());
      NewCI->setDoesNotThrow();
      CI->eraseFromParent();
      for (auto It = OldChain.rbegin(); It != OldChain.rend(); ++It)
        if ((*It)->use_empty()) (*It)->eraseFromParent();
    }
    std::string Name = F->getName().str();
    F->setName(Name + ".wide.dead");
    NewF->setName(Name);
    F->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, F->getType()));
    F->eraseFromParent();
    Changed = true;
    vxxDbg() << "vxx: narrowFir: retyped " << Name << " to i" << IW << "/i"
             << OW << "/i" << CW << "\n";
  }
  if (RunDecl->use_empty()) {
    RunDecl->eraseFromParent();
  } else if (AnySkipped) {
    // A caller could not be converted: leave its calls against the ORIGINAL
    // name (no `.wide.dead` dangling into the synthesis link).
    NewRun->setName(RunName + ".narrow");
    RunDecl->setName(RunName);
  }
  hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_fir_retype");
  return Changed;
}

bool narrowFftComplexArgs(Module &M) {
  Function *Proxy = nullptr;
  for (Function &F : M)
    if (F.getName().startswith("_ZN3hls3fft")) { Proxy = &F; break; }
  if (!Proxy) return false;
  LLVMContext &Ctx = M.getContext();

  bool FloatCmpx = Proxy->getName().contains("7config1ff");
  unsigned W = 16;
  if (!FloatCmpx) {
    StringRef N = Proxy->getName();
    size_t P = N.find("ap_fixedILi");
    if (P == StringRef::npos) return false;
    unsigned V = 0;
    for (size_t k = P + 11; k < N.size() && isdigit(N[k]); ++k)
      V = V * 10 + (N[k] - '0');
    if (V) W = V;
  }
  auto namedTy = [&](const std::string &Nm, ArrayRef<Type *> Fl) -> StructType * {
    StructType *T = M.getTypeByName(Nm);
    if (!T) T = StructType::create(Ctx, Fl, Nm);
    return T;
  };
  StructType *CmpxTy;
  if (FloatCmpx) {
    StructType *T = M.getTypeByName("struct.std::complex<float>");
    if (!T) {
      StructType *Inner =
          StructType::get(Ctx, {Type::getFloatTy(Ctx), Type::getFloatTy(Ctx)});
      T = StructType::create(Ctx, {Inner}, "struct.std::complex<float>");
    }
    CmpxTy = T;
  } else {
    std::string ws = std::to_string(W);
    StructType *Ssdm =
        namedTy("struct.ssdm_int<" + ws + ", true>", {IntegerType::get(Ctx, W)});
    StructType *Base = namedTy("struct.ap_fixed_base<" + ws + ", 1>", {Ssdm});
    StructType *Apf = namedTy("struct.ap_fixed<" + ws + ", 1>", {Base});
    CmpxTy = namedTy("class.std::complex<ap_fixed<" + ws + ", 1>>", {Apf, Apf});
  }
  // The wide Rust-side element: a literal 2-field struct of i128s (fixed)
  // or floats (float kind).
  StructType *WideTy = nullptr;
  {
    Type *F0 = FloatCmpx ? (Type *)Type::getFloatTy(Ctx)
                         : (Type *)IntegerType::get(Ctx, 128);
    WideTy = StructType::get(Ctx, {F0, F0});
  }

  SmallVector<Function *, 4> Callers;
  for (User *U : Proxy->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (Function *F = CI->getFunction())
        if (std::find(Callers.begin(), Callers.end(), F) == Callers.end())
          Callers.push_back(F);

  bool Changed = false;
  for (Function *Old : Callers) {
    // Only clone when a wide complex pointer actually appears in the args.
    bool Any = false;
    for (Argument &A : Old->args()) {
      if (auto *PT = dyn_cast<PointerType>(A.getType())) {
        Type *E = PT->getElementType();
        if (auto *AT = dyn_cast<ArrayType>(E)) E = AT->getElementType();
        if (E == WideTy) { Any = true; break; }
      }
    }
    if (!Any) continue;

    CmpxNarrowMapper Mapper(WideTy, CmpxTy);
    FunctionType *NewFT =
        cast<FunctionType>(Mapper.remapType(Old->getFunctionType()));
    Function *NewF =
        Function::Create(NewFT, Old->getLinkage(), Old->getName() + ".ncx", &M);
    NewF->copyAttributesFrom(Old);

    ValueToValueMapTy VMap;
    {
      auto OI = Old->arg_begin();
      auto NI = NewF->arg_begin();
      for (; OI != Old->arg_end(); ++OI, ++NI) {
        NI->setName(OI->getName());
        VMap[&*OI] = &*NI;
      }
    }
    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewF, Old, VMap, /*ModuleLevelChanges=*/false, Returns,
                      "", nullptr, &Mapper);

    std::string Name = Old->getName().str();
    Old->setName(Name + ".wide.dead");
    NewF->setName(Name);
    Old->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, Old->getType()));
    Old->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

bool renameFftComplexArgs(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Function *Proxy = M.getFunction("barista_fft_proxy");
  // v9 direct-call form: the Rust top calls the C++ template instance
  // `hls::fft<config1, ...>` by mangled name (wrapper layers get inlined by the
  // HLS backend and trip SYNCHK 200-91 on the exposed movers). Accept that decl
  // as the proxy.
  if (!Proxy)
    for (Function &F : M)
      if (F.isDeclaration() && !F.use_empty() &&
          F.getName().startswith("_ZN3hls3fftI")) {
        Proxy = &F;
        break;
      }
  if (!Proxy) return false;

  // Build the EXACT named type chain for
  // hls::stream<std::complex<ap_fixed<W,1>>>:
  //   ssdm_int<W,true>={iW}; ap_fixed_base<W,1>={ssdm}; ap_fixed<W,1>={base};
  //   class.std::complex<ap_fixed<W,1>>={apf,apf};
  //   class.hls::stream<std::complex<ap_fixed<W,1,AP_TRN,AP_WRAP,0>>,0>={complex}
  auto getStreamTy = [&](unsigned W) -> StructType * {
    auto named = [&](const std::string &Nm, ArrayRef<Type *> F) -> StructType * {
      StructType *T = M.getTypeByName(Nm);
      if (!T) T = StructType::create(Ctx, F, Nm);
      return T;
    };
    std::string ws = std::to_string(W);
    StructType *Ssdm = named("struct.ssdm_int<" + ws + ", true>",
                             {IntegerType::get(Ctx, W)});
    StructType *Base = named("struct.ap_fixed_base<" + ws + ", 1>", {Ssdm});
    StructType *Apf = named("struct.ap_fixed<" + ws + ", 1>", {Base});
    StructType *Cx =
        named("class.std::complex<ap_fixed<" + ws + ", 1>>", {Apf, Apf});
    std::string sn = "class.hls::stream<std::complex<ap_fixed<" + ws +
                     ", 1, AP_TRN, AP_WRAP, 0>>, 0>";
    StructType *St = M.getTypeByName(sn);
    if (!St) St = StructType::create(Ctx, {Cx}, sn);
    return St;
  };

  // A complex FFT sample arrives as a pointer-to-iK (K = bits of a packed
  // complex sample = 2*W), or — after narrowFftComplexArgs (the pure-port
  // flow) — as a pointer to the NAMED complex record already. Returns W
  // (>0) for such a param, else 0. Recognizing the named form keeps this
  // bridge's ap_fifo SpecInterface emission firing on pre-narrowed args
  // (without it the adapter presents a dataflow region with un-interfaced
  // complex* sub-fn args and reflow's elaborate crashes).
  // Pure-port wide form: the Rust glue's extern declares *const CmpxDataIn,
  // whose element is the LITERAL {i128,i128} (fixed) / flat {float,float}
  // (float) — width comes from the proxy's mangled name. Without this the
  // wide params fall through and the CmpxSeen<2 window slides onto blk_exp
  // (i32*), corrupting the retyped declaration.
  unsigned MangledW = 16;
  {
    StringRef PN = Proxy->getName();
    size_t P = PN.find("ap_fixedILi");
    if (P != StringRef::npos) {
      unsigned V = 0;
      for (size_t k = P + 11; k < PN.size() && isdigit(PN[k]); ++k)
        V = V * 10 + (PN[k] - '0');
      if (V) MangledW = V;
    }
  }
  bool FloatMangled = Proxy->getName().contains("7config1ff");
  auto cmpxW = [MangledW, FloatMangled](Type *PT) -> unsigned {
    auto *Ptr = dyn_cast<PointerType>(PT);
    if (!Ptr) return 0;
    Type *E = Ptr->getElementType();
    auto isCmpxFloatStruct = [](Type *T) -> bool {
      auto *VST = dyn_cast<StructType>(T);
      if (!VST) return false;
      if (VST->hasName() && VST->getName() == "struct.std::complex<float>")
        return true;
      // Rust `Complex<f32>` arrives as a NAMED two-float struct
      // (`%"barista_hls::Complex<f32>"`); accept any 2-identical-float
      // record shape, named or literal.
      return VST->getNumElements() == 2 &&
             VST->getElementType(0) == VST->getElementType(1) &&
             VST->getElementType(0)->isFloatTy();
    };
    // The Rust `Vector<Complex<f32>, SSR>` mirror is a repr(transparent)
    // NAMED wrapper over `[SSR x Complex<f32>]` — normalize it to the
    // inner array so the vector checks below see it.
    auto unwrapVec = [&](Type *T) -> ArrayType * {
      if (auto *AT = dyn_cast<ArrayType>(T)) return AT;
      if (auto *WST = dyn_cast<StructType>(T))
        if (WST->getNumElements() == 1)
          if (auto *AT = dyn_cast<ArrayType>(WST->getElementType(0)))
            return AT;
      return nullptr;
    };
    // Source-level `Stream<Vector<Complex<f32>, SSR>>` top arg (the
    // stream-shaped kernel signature): a stream wrapper over the Vector
    // mirror — peel the stream wrapper so the vector checks below see the
    // mirror. At this point the wrapper may still carry rustc's zero-size
    // array padding fields (`{ [0 x _], value, [0 x _] }`), so the payload
    // is its single non-zero-size field. Disjoint from the fixed-point
    // stream wrapper (its payload is the 2-field complex, not a 1-field
    // array wrapper) and from the C++-named chain (whose vector field is
    // the std::array STRUCT, not an ArrayType).
    if (auto *SW = dyn_cast<StructType>(E)) {
      Type *Payload = nullptr;
      bool OnlyOne = true;
      for (Type *FT : SW->elements()) {
        if (auto *ZA = dyn_cast<ArrayType>(FT))
          if (ZA->getNumElements() == 0) continue;
        if (Payload) { OnlyOne = false; break; }
        Payload = FT;
      }
      if (OnlyOne && Payload)
        if (auto *VW = dyn_cast<StructType>(Payload))
          if (VW->getNumElements() == 1 &&
              isa<ArrayType>(VW->getElementType(0)))
            E = VW;
    }
    if (ArrayType *Vec = unwrapVec(E)) {
      // Pure-port float-stream: a SMALL array of complex<float> under the
      // pointer is the hls::vector mirror itself (`*const Vector<Complex<
      // f32>, SSR>`), NOT a data container to peel — packed bits K = SSR*64.
      if (FloatMangled && isCmpxFloatStruct(Vec->getElementType()) &&
          Vec->getNumElements() <= 8 && Vec == E)
        return (unsigned)(Vec->getNumElements() * 64 / 2);
      if (auto *WST = dyn_cast<StructType>(E)) {
        // Named wrapper directly under the pointer = the vector itself.
        (void)WST;
        if (FloatMangled && isCmpxFloatStruct(Vec->getElementType()) &&
            Vec->getNumElements() <= 8)
          return (unsigned)(Vec->getNumElements() * 64 / 2);
      }
      if (Vec == E) E = Vec->getElementType();
    }
    // Post-peel: a container of vectors ([N x vec]) leaves the vector here
    // (array form or the named Rust wrapper) — same K = SSR*64 contract.
    if (ArrayType *VecAT = unwrapVec(E)) {
      if (FloatMangled && isCmpxFloatStruct(VecAT->getElementType()) &&
          VecAT->getNumElements() <= 8)
        return (unsigned)(VecAT->getNumElements() * 64 / 2);
      return 0;
    }
    if (auto *ST = dyn_cast<StructType>(E)) {
      if (!ST->hasName()) {
        if (ST->getNumElements() == 2 &&
            ST->getElementType(0) == ST->getElementType(1)) {
          if (!FloatMangled && ST->getElementType(0)->isIntegerTy(128))
            return MangledW;
          if (FloatMangled && ST->getElementType(0)->isFloatTy())
            return 32;
        }
        return 0;
      }
      StringRef Nm = ST->getName();
      if (Nm == "struct.std::complex<float>") return 32;
      if (Nm.startswith("class.std::complex<ap_fixed<")) {
        unsigned V = 0;
        for (size_t k = 28; k < Nm.size() && isdigit(Nm[k]); ++k)
          V = V * 10 + (Nm[k] - '0');
        return V;
      }
      return 0;
    }
    auto *IT = dyn_cast<IntegerType>(E);
    if (!IT) return 0;
    unsigned K = IT->getBitWidth();
    if (K < 16 || (K & 1)) return 0;  // need even; status i8/i1 excluded
    return K / 2;
  };

  // The new POINTEE type for an FFT proxy param, or nullptr if unchanged:
  //   complex sample ptr (i32* / [N x i32]*) -> class.hls::stream<complex<..>>
  //   status ptr (i8* / [N x i8]*)           -> i1  (1-bit; get_status writes
  //     the overflow flag via IfWrite.Stream on i1* -> din/full_n/write FIFO;
  //     a byte i8* falls back to a raw store -> ap_vld port mismatch)
  auto newPointee = [&](Type *PT) -> Type * {
    if (unsigned W = cmpxW(PT)) return getStreamTy(W);
    if (auto *Ptr = dyn_cast<PointerType>(PT)) {
      Type *E = Ptr->getElementType();
      if (auto *AT = dyn_cast<ArrayType>(E)) E = AT->getElementType();
      if (auto *IT = dyn_cast<IntegerType>(E))
        if (IT->getBitWidth() == 8) {
          // status bool* -> class.hls::stream<bool> ( = { i1 } ): the proxy takes
          // hls::stream<bool>& and pushes the overflow flag, so the port is a
          // FIFO write (din/full_n/write). A bare i1*
          // would still get a raw-store -> ap_vld in get_status.
          StructType *BS = M.getTypeByName("class.hls::stream<bool>");
          if (!BS) BS = StructType::create(Ctx, {IntegerType::get(Ctx, 1)},
                                           "class.hls::stream<bool>");
          return BS;
        }
    }
    return nullptr;
  };

  // 1. Retyped proxy declaration (complex i32* -> stream*, status i8* -> i1*).
  FunctionType *POldFT = Proxy->getFunctionType();
  SmallVector<Type *, 8> PNew;
  bool ProxyChanged = false;
  for (unsigned i = 0; i < POldFT->getNumParams(); ++i) {
    Type *PT = POldFT->getParamType(i);
    Type *NE = newPointee(PT);
    if (NE) { PNew.push_back(PointerType::get(NE, 0)); ProxyChanged = true; }
    else PNew.push_back(PT);
  }
  if (!ProxyChanged) return false;
  FunctionType *PNewFT =
      FunctionType::get(POldFT->getReturnType(), PNew, POldFT->isVarArg());
  Function *NewProxy = Function::Create(PNewFT, Proxy->getLinkage(),
                                        "barista_fft_proxy.cx", &M);
  NewProxy->copyAttributesFrom(Proxy);

  // FFT no-retype bridge mode (the default; fft array-interface):
  // fft_top's xn/xk/status stay as PLAIN pointers with
  // SpecInterface("ap_fifo", -1, -1, depth) — only the hls_fft.h movers
  // (compiled inside the proxy TU) carry the stream types. The full retype
  // below diverges from that (cosim TB stream-drains the top ports → COSIM
  // 212-320). Bridge mode: keep the top args untouched, retype ONLY the
  // proxy declaration, bitcast the call args to the stream types (the
  // proxy symbol's params ARE the stream classes, so the
  // call must be stream-typed — HLS 214-136 otherwise), and emit the
  // ap_fifo SpecInterface on the top args (depth from the
  // `__vxx_stream_depth` markers lib.rs already places; the markers are
  // erased later by injectStreamDepth's Argument path).
  {  // FFT proxy kernels: emit the ap_fifo/SpecInterface form directly
    // Native-float FFT instances (e.g. hls::fft<config1, float, float> =
    // `_ZN3hls3fftI7config1ff...`): the complex element is
    // `struct.std::complex<float>` = {float, float} — the same name
    // the proxy TU uses (float_ssr2), so no sdx_kernel
    // naming shim is involved for floats.
    bool FloatCmpx = Proxy->getName().contains("7config1ff");
    // Stream-ref hls::fft overload (`hls::fft<config1>(hls::stream<..>&,
    // ...)` = mangled `RNS_6streamI...`): the top's xn/xk params are
    // `class.hls::stream<std::complex<..>>*` (nonnull align 2
    // dereferenceable(4)) with `stream_interface` + `xlx_ap_fifo` +
    // `xlx_reqd_pipe_depth` bundles (interface_stream).
    bool StreamRef = Proxy->getName().contains("6stream");
    // Float STREAM element chain (interface_stream_float_ssr2):
    //   class.hls::stream<hls::vector<std::complex<float>, S>, 0>
    //     = { class.hls::vector<std::complex<float>, S> }
    //     = { { struct.std::array<std::complex<float>, S> } }
    //     = { { { [S x struct.std::complex<float>] } } }
    // with complex<float> = { {float, float} }. SSR=1 degenerates to
    // class.hls::stream<std::complex<float>, 0> = { complex }.
    auto getFloatStreamTy = [&](unsigned ElemBits) -> StructType * {
      auto namedF = [&](const std::string &Nm,
                        ArrayRef<Type *> Fl) -> StructType * {
        StructType *T = M.getTypeByName(Nm);
        if (!T) T = StructType::create(Ctx, Fl, Nm);
        return T;
      };
      StructType *CxF = M.getTypeByName("struct.std::complex<float>");
      if (!CxF) {
        StructType *Inner = StructType::get(
            Ctx, {Type::getFloatTy(Ctx), Type::getFloatTy(Ctx)});
        CxF = StructType::create(Ctx, {Inner}, "struct.std::complex<float>");
      }
      unsigned SSR = ElemBits / 64;
      if (SSR <= 1)
        return namedF("class.hls::stream<std::complex<float>, 0>", {CxF});
      std::string ss = std::to_string(SSR);
      StructType *Arr =
          namedF("struct.std::array<std::complex<float>, " + ss + ">",
                 {ArrayType::get(CxF, SSR)});
      StructType *Vec = namedF(
          "class.hls::vector<std::complex<float>, " + ss + ">", {Arr});
      return namedF("class.hls::stream<hls::vector<std::complex<float>, " +
                        ss + ">, 0>",
                    {Vec});
    };
    // The bridge call-cast target: the std::complex<ap_fixed<W,1>> ELEMENT
    // type (the array-param proxy's params decay to this), NOT the
    // hls::stream class — a stream-typed bitcast at the call site makes
    // the backend infer a second interface on the top arg (SYN 201-504).
    auto getComplexTy = [&](unsigned W) -> StructType * {
      if (FloatCmpx) {
        // clang-16's std::complex<float> wraps a `_Complex float` member:
        // `{ { float, float } }` (literal inner struct). A flat
        // {float, float} does NOT structurally unify at llvm-link — the
        // kernel's type gets a `.0` suffix and the call goes through a
        // function bitcast -> HLS 214-136.
        StructType *T = M.getTypeByName("struct.std::complex<float>");
        if (!T) {
          StructType *Inner = StructType::get(
              Ctx, {Type::getFloatTy(Ctx), Type::getFloatTy(Ctx)});
          T = StructType::create(Ctx, {Inner}, "struct.std::complex<float>");
        }
        return T;
      }
      auto namedB = [&](const std::string &Nm,
                        ArrayRef<Type *> Fl) -> StructType * {
        StructType *T = M.getTypeByName(Nm);
        if (!T) T = StructType::create(Ctx, Fl, Nm);
        return T;
      };
      std::string ws = std::to_string(W);
      StructType *Ssdm = namedB("struct.ssdm_int<" + ws + ", true>",
                                {IntegerType::get(Ctx, W)});
      StructType *Base = namedB("struct.ap_fixed_base<" + ws + ", 1>", {Ssdm});
      StructType *Apf = namedB("struct.ap_fixed<" + ws + ", 1>", {Base});
      // `class.std::complex<...>`: must match the proxy TU's record name
      // exactly (214-136 prototype check) AND the FFT mover lowering
      // string-matches the class.-name to emit BitConcatenate+IfWrite.Stream
      // (struct.-named complex falls back to byte-RMW volatile → SYNCHK
      // 200-91). The proxy TU is class.-named via its sdx_kernel naming
      // shim (fft_proxy.cpp barista_fft_unused, first fn in TU).
      return namedB("class.std::complex<ap_fixed<" + ws + ", 1>>",
                    {Apf, Apf});
    };
    // Bridge param target per proxy param (v7): the proxy .cpp is now
    // ARRAY-param (`cmpxDataIn xn[N]`, `bool* status`) — the
    // top's hls::fft call shape — so the call casts to the complex ELEMENT
    // type / i8 (memory bool), NOT the hls::stream classes. With no
    // stream typing on the top args, the explicit ap_fifo SpecInterface
    // emitted below is the ONLY interface source, so the
    // 201-504 double-interface conflict cannot arise.
    // Per-param bridge types, positional: only the FIRST TWO complex-sample
    // pointer params (xn/xk resp. in/out) retype to the complex element
    // (any later i32* — hls::fft's blk_exp — stays); bool-ish pointers
    // (status/ovflo) retype to i1* (the backend types the def's bool*
    // as i1* — must match exactly for the 214-136 prototype check).
    SmallVector<Type *, 8> BNew;
    bool BChanged = false;
    {
      unsigned CmpxSeen = 0;
      for (unsigned i = 0; i < POldFT->getNumParams(); ++i) {
        Type *PT = POldFT->getParamType(i);
        Type *BT = nullptr;
        if (auto *Ptr = dyn_cast<PointerType>(PT)) {
          Type *E = Ptr->getElementType();
          if (auto *AT = dyn_cast<ArrayType>(E)) E = AT->getElementType();
          if (unsigned W = cmpxW(PT)) {
            if (CmpxSeen < 2) {
              BT = PointerType::get(
                  StreamRef ? (FloatCmpx ? (Type *)getFloatStreamTy(W * 2)
                                         : (Type *)getStreamTy(W))
                            : (Type *)getComplexTy(W),
                  0);
              ++CmpxSeen;
            }
          } else if (E->isIntegerTy(8) || E->isIntegerTy(1)) {
            BT = PointerType::get(IntegerType::get(Ctx, 1), 0);
          }
        }
        if (BT) { BNew.push_back(BT); BChanged = true; }
        else BNew.push_back(PT);
      }
    }
    FunctionType *BProxyFT =
        FunctionType::get(POldFT->getReturnType(), BNew, POldFT->isVarArg());
    std::string ProxyName = Proxy->getName().str();
    Function *BProxy = Function::Create(BProxyFT, Proxy->getLinkage(),
                                        ProxyName + ".arr", &M);
    BProxy->copyAttributesFrom(Proxy);
    (void)BChanged;

    // Collect __vxx_stream_depth markers (depth per OLD top Argument).
    DenseMap<Value *, uint64_t> DepthOf;
    if (Function *DM = M.getFunction("__vxx_stream_depth"))
      for (User *U : DM->users())
        if (auto *DCI = dyn_cast<CallInst>(U))
          if (DCI->arg_size() >= 2)
            if (auto *D = dyn_cast<ConstantInt>(DCI->getArgOperand(1)))
              DepthOf[DCI->getArgOperand(0)->stripPointerCasts()] =
                  D->getZExtValue();

    FunctionType *SpecTy = FunctionType::get(Type::getVoidTy(Ctx), true);
    FunctionCallee SpecFn =
        M.getOrInsertFunction("_ssdm_op_SpecInterface", SpecTy);
    if (auto *FF = dyn_cast<Function>(SpecFn.getCallee()))
      FF->addFnAttr(Attribute::NoUnwind);
    GlobalVariable *FifoStr = getOrCreateCStrGlobal(M, "ap_fifo");
    GlobalVariable *EmptyStr = getOrCreateCStrGlobal(M, "");
    Type *I32 = Type::getInt32Ty(Ctx);
    Type *I64b = Type::getInt64Ty(Ctx);
    (void)I64b;

    // The param type for a proxy-fed arg. The pre-restructure
    // top has COMPLEX-typed params (`class.std::complex<..>*`
    // with fpga.decayed.dim.hint) — the backend converts them to i32*
    // ports and MERGES the movers' {re,im} i16 field accesses into single
    // 32-bit IfRead/IfWrite + BitConcatenate. Decaying to i32* ourselves
    // loses that: the movers become TWO 16-bit FIFO accesses per element
    // (II=2 "resource conflict xn_read/xn_read_1"), the RTL pops 2048 beats
    // from a 1024-beat TV → cosim deadlock at the first frame.
    // status: it is `i1*` (bool* lowers to i1*) — the FIFO
    // din port must be 1 bit wide to match the exported component.xml.
    auto decayedTy = [&](Type *PT) -> Type * {
      if (!newPointee(PT)) return nullptr;  // not a proxy-fed pointer
      auto *Ptr = cast<PointerType>(PT);
      if (unsigned W = cmpxW(PT))
        return PointerType::get(
            StreamRef ? (FloatCmpx ? (Type *)getFloatStreamTy(W * 2)
                                   : (Type *)getStreamTy(W))
                      : (Type *)getComplexTy(W),
            Ptr->getAddressSpace());
      return PointerType::get(IntegerType::get(Ctx, 1),
                              Ptr->getAddressSpace());
    };

    SmallPtrSet<Function *, 4> BTops;
    for (User *U : Proxy->users())
      if (auto *PCI = dyn_cast<CallInst>(U))
        if (Function *F = PCI->getFunction()) BTops.insert(F);

    for (Function *F : BTops) {
      FunctionType *OldFT = F->getFunctionType();
      unsigned N = OldFT->getNumParams();
      SmallVector<Type *, 8> NewParams;
      SmallVector<bool, 8> Decayed(N, false);
      SmallVector<uint64_t, 8> ArgDepth(N, 0);
      {
        auto AIt = F->arg_begin();
        for (unsigned i = 0; i < N; ++i, ++AIt) {
          Type *PT = OldFT->getParamType(i);
          if (Type *DT = decayedTy(PT)) {
            Decayed[i] = true;
            ArgDepth[i] = DepthOf.lookup(&*AIt);
            NewParams.push_back(DT);
          } else {
            NewParams.push_back(PT);
          }
        }
      }
      FunctionType *NewFT =
          FunctionType::get(F->getReturnType(), NewParams, F->isVarArg());
      Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                        F->getName() + ".fftarr", &M);
      NewF->copyAttributesFrom(F);
      {
        AttributeList AL = NewF->getAttributes();
        for (unsigned i = 0; i < N; ++i)
          if (Decayed[i]) AL = AL.removeParamAttributes(Ctx, i);
        NewF->setAttributes(AL);
      }
      NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
      SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
      F->getAllMetadata(MDs);
      for (auto &PMD : MDs) NewF->setMetadata(PMD.first, PMD.second);

      auto OldA = F->arg_begin();
      auto NewA = NewF->arg_begin();
      for (unsigned i = 0; i < N; ++i, ++OldA, ++NewA) {
        NewA->takeName(&*OldA);
        if (!Decayed[i]) {
          for (Attribute A : F->getAttributes().getParamAttributes(i))
            NewA->addAttr(A);
          OldA->replaceAllUsesWith(&*NewA);
          continue;
        }
        NewA->addAttr(Attribute::NoAlias);
        // Stream-ref overload: the C++ top's stream params carry
        // `nonnull align A dereferenceable(EltBytes)` — ap_fixed streams:
        // align 2 deref 4 (interface_stream); float SSR vector streams:
        // align 16 deref 16 (interface_stream_float_ssr2; hls::vector is
        // over-aligned to its size).
        if (StreamRef && cmpxW(OldFT->getParamType(i))) {
          unsigned EltBytes = cmpxW(OldFT->getParamType(i)) * 2 / 8;
          unsigned AlignB = FloatCmpx ? (EltBytes > 8 ? EltBytes : 4) : 2;
          NewA->addAttr(Attribute::get(Ctx, Attribute::NonNull));
          NewA->addAttr(Attribute::getWithAlignment(Ctx, Align(AlignB)));
          NewA->addAttr(
              Attribute::getWithDereferenceableBytes(Ctx, EltBytes));
        }
        // Body uses are `gep [N x T]* arg, 0, 0` (as_ptr) plus marker
        // bitcasts. The decayed arg IS &arr[0], so all-zero GEPs collapse
        // to the new arg; everything else gets a bitcast.
        SmallVector<Use *, 8> Uses;
        for (Use &U : OldA->uses()) Uses.push_back(&U);
        for (Use *U : Uses) {
          User *Usr = U->getUser();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
            if (GEP->getPointerOperand() == &*OldA &&
                GEP->hasAllZeroIndices()) {
              if (GEP->getType() == NewA->getType()) {
                GEP->replaceAllUsesWith(&*NewA);
                GEP->eraseFromParent();
              } else {
                // status: decayed i1* vs the body's i8* GEP — bitcast.
                IRBuilder<> GB(GEP);
                GEP->replaceAllUsesWith(GB.CreateBitCast(&*NewA, GEP->getType()));
                GEP->eraseFromParent();
              }
              continue;
            }
          }
          IRBuilder<> UB(cast<Instruction>(Usr));
          U->set(UB.CreateBitCast(&*NewA, OldA->getType()));
        }
      }

      // Erase the __vxx_stream_depth markers on the decayed args (their depth
      // is consumed into ArgDepth above) so injectStreamDepth doesn't emit a
      // second, i8*-bitcast SpecStream later.
      if (Function *DM = M.getFunction("__vxx_stream_depth")) {
        SmallVector<CallInst *, 4> DCalls;
        for (User *U : DM->users())
          if (auto *DCI = dyn_cast<CallInst>(U))
            if (DCI->getFunction() == NewF) {
              Value *R = DCI->getArgOperand(0)->stripPointerCasts();
              if (auto *AV = dyn_cast<Argument>(R))
                if (AV->getParent() == NewF && Decayed[AV->getArgNo()])
                  DCalls.push_back(DCI);
            }
        for (CallInst *DCI : DCalls) {
          Value *P = DCI->getArgOperand(0);
          DCI->eraseFromParent();
          if (auto *PI = dyn_cast<Instruction>(P))
            if (PI->use_empty()) PI->eraseFromParent();
        }
      }

      // Interface pragmas on each decayed arg, in the CANONICAL
      // pre-restructure op-bundle form (NOT the _ssdm
      // calls — those are the LOWERED form, and emitting them directly
      // is too late: the hls_fft.h movers' complex stores are lowered
      // during the link/restructure, which reads these op-bundles. Without
      // them the store becomes a byte-RMW volatile whose read of xk trips
      // SYNCHK 200-91):
      //   call void @llvm.sideeffect() #A [ "xlx_ap_fifo"(T* arg, i32 -1,
      //        [0 x i8] zeroinitializer, i64 depth) ]
      //   call void @llvm.sideeffect() #A [ "xlx_reqd_pipe_depth"(T* arg,
      //        i32 0) ]                       ; only stream-pragma'd args
      // with call-site attrs: inaccessiblememonly nounwind
      //   "xlx.port.bitwidth"="<total bits>" "xlx.source"="user".
      // The bundle pointer for xn/xk is the COMPLEX-typed ptr (the arg
      // type pre-decay); status stays i1*.
      {
        (void)SpecFn;
        (void)FifoStr;
        (void)EmptyStr;
        Function *SideEff =
            Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
        IRBuilder<> EB(&*NewF->getEntryBlock().getFirstInsertionPt());
        Type *I64sp = Type::getInt64Ty(Ctx);
        Constant *EmptyArr = ConstantAggregateZero::get(
            ArrayType::get(IntegerType::get(Ctx, 8), 0));
        auto SA = NewF->arg_begin();
        for (unsigned i = 0; i < N; ++i, ++SA) {
          if (!Decayed[i]) continue;
          bool IsStatus = cast<PointerType>(SA->getType())
                              ->getElementType()
                              ->isIntegerTy(1);
          uint64_t ArrLen = 0;
          unsigned ElemBits = 0;
          if (auto *OPT = dyn_cast<PointerType>(OldFT->getParamType(i)))
            if (auto *AT = dyn_cast<ArrayType>(OPT->getElementType())) {
              ArrLen = AT->getNumElements();
              if (auto *IT = dyn_cast<IntegerType>(AT->getElementType()))
                ElemBits = IT->getBitWidth();
            }
          unsigned CmpxWBits = cmpxW(OldFT->getParamType(i));
          Value *BundlePtr = &*SA;
          if (!IsStatus) {
            // StreamRef: SA already IS the stream-typed ptr (no cast); the
            // C++ stream args carry no array dim hint either.
            if (!StreamRef && CmpxWBits)
              BundlePtr = EB.CreateBitCast(
                  &*SA, PointerType::get(getComplexTy(CmpxWBits), 0));
            if (ArrLen && !StreamRef)
              SA->addAttr(Attribute::get(Ctx, "fpga.decayed.dim.hint",
                                         std::to_string(ArrLen)));
          }
          // C++ bitwidth attr: total array bits for array args
          // (interface_array: 1024*32=32768), ELEMENT bits for stream args
          // (interface_stream: 32), 0 for status.
          uint64_t BW =
              IsStatus ? 0
                       : (StreamRef ? (uint64_t)CmpxWBits * 2
                                    : (uint64_t)ArrLen * ElemBits);
          auto addBundleCall = [&](StringRef Tag, ArrayRef<Value *> Vals,
                                   bool WithBW) {
            OperandBundleDef OB(
                Tag.str(), SmallVector<Value *, 4>(Vals.begin(), Vals.end()));
            CallInst *CI = EB.CreateCall(SideEff, {}, {OB});
            CI->addAttribute(AttributeList::FunctionIndex,
                             Attribute::get(Ctx, Attribute::InaccessibleMemOnly));
            CI->addAttribute(AttributeList::FunctionIndex,
                             Attribute::get(Ctx, Attribute::NoUnwind));
            if (WithBW)
              CI->addAttribute(AttributeList::FunctionIndex,
                               Attribute::get(Ctx, "xlx.port.bitwidth",
                                              std::to_string(BW)));
            CI->addAttribute(AttributeList::FunctionIndex,
                             Attribute::get(Ctx, "xlx.source", "user"));
          };
          // hls::stream-typed args always carry a `stream_interface` bundle
          // (from the TYPE, pragma-independent; no bitwidth attr).
          if (StreamRef && !IsStatus)
            addBundleCall("stream_interface", {&*SA}, /*WithBW=*/false);
          // Interface bundles only for PRAGMA'd args (= depth marker
          // present). float_ssr2 has NO interface pragma on in/out —
          // they fall to the backend's default (BRAM-side) handling.
          uint64_t D = ArgDepth[i];
          if (!D) continue;
          addBundleCall("xlx_ap_fifo",
                        {BundlePtr, ConstantInt::get(I32, -1), EmptyArr,
                         ConstantInt::get(I64sp, D)},
                        /*WithBW=*/true);
          if (!IsStatus)
            addBundleCall("xlx_reqd_pipe_depth",
                          {BundlePtr, ConstantInt::get(I32, 0)},
                          /*WithBW=*/true);
        }
      }

      // Rewrite the proxy calls: bitcast each mismatched pointer arg to the
      // stream-typed NewProxy param (the proxy symbol's params are
      // the hls::stream classes; HLS 214-136 otherwise).
      SmallVector<CallInst *, 2> PCalls;
      for (BasicBlock &BB : *NewF)
        for (Instruction &I : BB)
          if (auto *PCI = dyn_cast<CallInst>(&I))
            if (PCI->getCalledFunction() == Proxy) PCalls.push_back(PCI);
      for (CallInst *PCI : PCalls) {
        IRBuilder<> CB(PCI);
        SmallVector<Value *, 8> Args;
        for (unsigned i = 0;
             i < PCI->arg_size() && i < BProxyFT->getNumParams(); ++i) {
          Value *A = PCI->getArgOperand(i);
          Type *Want = BProxyFT->getParamType(i);
          if (A->getType() != Want) {
            // Prefer the cast-free root (the decayed top Argument itself):
            // a bitcast ROUND-TRIP (complex* -> i64* -> complex*) makes
            // the backend's array_reshape-through-call give up ("HLS 214-184
            // reshape pragma on function argument, in 'call' is
            // unsupported") and the un-reshaped call then mismatches the
            // reshaped callee -> HLS 214-136 (float_ssr2).
            Value *Root = A->stripPointerCasts();
            if (Root->getType() == Want) {
              A = Root;
            } else if (A->getType()->isPointerTy() && Want->isPointerTy()) {
              A = CB.CreateBitCast(A, Want);
            } else if (A->getType()->isIntegerTy() && Want->isIntegerTy()) {
              A = CB.CreateZExtOrTrunc(A, Want);  // e.g. i8 bool -> i1
            }
          }
          // Use a spill-and-reload argument shape for complex-array pointers:
          // spill to an alloca and reload (`%in.addr = alloca complex*;
          // store; %0 = load; call(..., %0, ...)`). The backend's
          // array_reshape-through-call (hls_fft.h SSR overload carries
          // `#pragma HLS array_reshape` on its params) only follows that
          // form — a direct function-Argument argument trips
          // "HLS 214-184 reshape pragma on function argument, in 'call' is
          // unsupported" and the unreshaped call then mismatches the
          // reshaped callee (HLS 214-136, float_ssr2).
          if (A->getType()->isPointerTy() &&
              isa<StructType>(
                  cast<PointerType>(A->getType())->getElementType())) {
            IRBuilder<> AB(&NewF->getEntryBlock(),
                           NewF->getEntryBlock().getFirstInsertionPt());
            AllocaInst *Addr = AB.CreateAlloca(A->getType());
            CB.CreateStore(A, Addr);
            A = CB.CreateLoad(A->getType(), Addr);
          }
          Args.push_back(A);
        }
        CallInst *NC = CB.CreateCall(BProxy, Args);
        NC->setCallingConv(PCI->getCallingConv());
        PCI->replaceAllUsesWith(NC);
        PCI->eraseFromParent();
      }

      hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
    }
    if (NewProxy->use_empty()) NewProxy->eraseFromParent();
    // BProxy is a DECLARATION that llvm-link must resolve against the
    // anchor TU's definition — it needs the REAL symbol name. Swap
    // unconditionally: a transient non-call use of the original (seen in
    // the pure-port float-stream flow) must not leave the redirected
    // calls dangling on the ".arr" name (HLS 214-194).
    Proxy->setName(ProxyName + ".orig");
    BProxy->setName(ProxyName);
    if (Proxy->use_empty()) Proxy->eraseFromParent();
    vxxDbg() << "vxx: FFT bridge mode v9 (direct hls::fft call + decayed "
              "top args [status i1*] + C++-exact SpecStream/SpecInterface "
              "ap_fifo)\n";
    return true;
  }

  // 2. Clone each caller (FFT top) with the complex args retyped.
  SmallPtrSet<Function *, 4> Tops;
  for (User *U : Proxy->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (Function *F = CI->getFunction()) Tops.insert(F);

  Type *I64 = Type::getInt64Ty(Ctx);
  for (Function *F : Tops) {
    FunctionType *OldFT = F->getFunctionType();
    unsigned N = OldFT->getNumParams();
    SmallVector<Type *, 8> NewParams;
    SmallVector<Type *, 8> NewElem(N, nullptr);
    for (unsigned i = 0; i < N; ++i) {
      Type *PT = OldFT->getParamType(i);
      Type *NE = newPointee(PT);
      if (NE) { NewElem[i] = NE; NewParams.push_back(PointerType::get(NE, 0)); }
      else NewParams.push_back(PT);
    }
    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParams, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                      F->getName() + ".fftcx", &M);
    NewF->copyAttributesFrom(F);
    { AttributeList AL = NewF->getAttributes();
      for (unsigned i = 0; i < N; ++i) if (NewElem[i]) AL = AL.removeParamAttributes(Ctx, i);
      NewF->setAttributes(AL); }
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);

    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < N; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (!NewElem[i]) {
        for (Attribute A : F->getAttributes().getParamAttributes(i)) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Strip the explicit `_ssdm_op_SpecInterface(arg, "ap_fifo")` marker ONLY
      // on the complex *stream* args (class.hls::stream pointee): the stream type
      // carries its own interface, so the stream top emits SpecStream
      // only — keeping ap_fifo SpecInterface causes SYN 201-504. The narrowed i1*
      // status, by contrast, KEEPS its SpecStream + SpecInterface(ap_fifo)
      // (i1* status needs both -> din/full_n/write).
      bool IsStreamArg = isa<StructType>(NewElem[i]);
      SmallVector<CallInst *, 2> EraseCalls;
      SmallVector<Use *, 8> Uses;
      for (Use &U : OldA->uses()) Uses.push_back(&U);
      for (Use *U : Uses) {
        User *Usr = U->getUser();
        if (IsStreamArg) {
          if (auto *CI = dyn_cast<CallInst>(Usr)) {
            if (Function *Cal = CI->getCalledFunction()) {
              if (Cal->getName() == "_ssdm_op_SpecInterface") {
                EraseCalls.push_back(CI);
                continue;
              }
            }
          }
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
          if (GEP->getPointerOperand() == &*OldA) {
            bool allZero = true;
            for (unsigned k = 1; k < GEP->getNumOperands(); ++k) {
              auto *C = dyn_cast<ConstantInt>(GEP->getOperand(k));
              if (!C || !C->isZero()) { allZero = false; break; }
            }
            if (allZero) {
              IRBuilder<> GB(GEP);
              // Array retype (BRAM, e.g. [N x complex<float>]): preserve the
              // original all-zero index list so the GEP still yields an element
              // pointer. Stream/struct retype: a single 0 index (pass-through to
              // the stream pointer).
              SmallVector<Value *, 4> Idx;
              if (isa<ArrayType>(NewElem[i]))
                for (unsigned k = 1; k < GEP->getNumOperands(); ++k)
                  Idx.push_back(ConstantInt::get(I64, 0));
              else
                Idx.push_back(ConstantInt::get(I64, 0));
              auto *NG = GB.CreateInBoundsGEP(NewElem[i], &*NewA, Idx, GEP->getName());
              GEP->replaceAllUsesWith(NG);
              GEP->eraseFromParent();
              continue;
            }
          }
        }
        U->set(&*NewA);
      }
      for (CallInst *CI : EraseCalls) CI->eraseFromParent();
    }

    SmallVector<CallInst *, 2> ProxyCalls;
    for (BasicBlock &BB : *NewF)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() == Proxy) ProxyCalls.push_back(CI);
    for (CallInst *CI : ProxyCalls) {
      SmallVector<Value *, 8> Args(CI->arg_begin(), CI->arg_end());
      IRBuilder<> CB(CI);
      CallInst *NC = CB.CreateCall(NewProxy, Args);
      NC->setCallingConv(CI->getCallingConv());
      CI->replaceAllUsesWith(NC);
      CI->eraseFromParent();
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
  }

  if (Proxy->use_empty()) {
    Proxy->eraseFromParent();
    NewProxy->setName("barista_fft_proxy");
  }
  vxxDbg() << "vxx: renamed FFT complex stream kernel arg(s) to "
            "class.hls::stream<std::complex<ap_fixed>>\n";
  return true;
}



// SSR / native-float FFT variant (interface_stream_float_ssr2,
// logicore_fft_float_ssr). The proxy is `barista_fft_proxy_ssr`:
//   void barista_fft_proxy_ssr(i8 direction,
//       hls::stream<hls::vector<std::complex<float>, 2>>& in,
//       hls::stream<hls::vector<std::complex<float>, 2>>& out, i8* ovflo)
// The stream element is a 128-bit vector<complex<float>,2>, so the Rust kernel
// passes the streams as `[N x i128]*` (u128 = one packed vector transaction).
// We retype ONLY those i128 pointer args to the matching named stream type;
// the i8 `direction` scalar and the i8* `ovflo` (an unused ap_fifo -> read
// port, kept as a plain pointer) are left untouched. The ap_fixed pass
// `renameFftComplexArgs` only fires for `barista_fft_proxy`, so the two are
// disjoint.
bool renameFftSsrArgsImpl(Module &M, StringRef ProxyName) {
  LLVMContext &Ctx = M.getContext();
  Function *Proxy = M.getFunction(ProxyName);
  if (!Proxy) return false;

  // Build the EXACT named chain the proxy's hls_stream.h/hls_vector.h compile
  // emits, so the linker merges the kernel's retyped
  // arg with the proxy's param and the prototype matches (else HLS 214-136):
  //   class.hls::stream<hls::vector<std::complex<float>, 2>, 0> = { vector }
  //   class.hls::vector<std::complex<float>, 2>   = { array }
  //   struct.std::array<std::complex<float>, 2>   = { [2 x complex] }
  //   struct.std::complex<float>                  = { { float, float } }
  auto getSsrStreamTy = [&]() -> StructType * {
    auto named = [&](const std::string &Nm, ArrayRef<Type *> F) -> StructType * {
      StructType *T = M.getTypeByName(Nm);
      if (!T) T = StructType::create(Ctx, F, Nm);
      return T;
    };
    Type *F32 = Type::getFloatTy(Ctx);
    StructType *Inner = StructType::get(Ctx, {F32, F32});  // anon { float, float }
    StructType *Cx = named("struct.std::complex<float>", {Inner});
    StructType *Arr = named("struct.std::array<std::complex<float>, 2>",
                            {ArrayType::get(Cx, 2)});
    StructType *Vec = named("class.hls::vector<std::complex<float>, 2>", {Arr});
    return named("class.hls::stream<hls::vector<std::complex<float>, 2>, 0>",
                 {Vec});
  };

  // class.hls::stream<bool> = { i1 } — the proxy reads `hls::stream<bool>& ovflo`
  // once (a side-effecting FIFO pop), so ovflo becomes a stream channel with a
  // dout/empty_n/read port (the unused-ap_fifo ovflo).
  // Modelling it as a stream (not i1*+ap_fifo) avoids the SpecStream/SpecInterface
  // and array-decay ap_memory conflicts (SYN 201-504).
  auto getBoolStreamTy = [&]() -> StructType * {
    StructType *BS = M.getTypeByName("class.hls::stream<bool>");
    if (!BS) BS = StructType::create(Ctx, {IntegerType::get(Ctx, 1)},
                                     "class.hls::stream<bool>");
    return BS;
  };
  // struct.std::complex<float> = { { float, float } } — the array-variant FFT
  // (interface_array_float_ssr2) uses complex<float> BRAM arrays. The Rust BRAM
  // arg is `[N x i64]*` (each u64 = one complex<float>); we retype it to
  // `[N x struct.std::complex<float>]*` so the proxy's cmpxDataIn[] prototype
  // matches without a cast (a cast breaks the FFT IP dataflow form, HLS 214-113).
  auto getComplexFloatTy = [&]() -> StructType * {
    StructType *Cx = M.getTypeByName("struct.std::complex<float>");
    if (!Cx) {
      Type *F32 = Type::getFloatTy(Ctx);
      StructType *Inner = StructType::get(Ctx, {F32, F32});
      Cx = StructType::create(Ctx, {Inner}, "struct.std::complex<float>");
    }
    return Cx;
  };

  // The new pointee for an SSR proxy param, or nullptr if unchanged:
  //   pointer-to-i128 (the [N x i128] stream) -> vector<complex<float>,2> stream
  //   pointer-to-i8   (the [1 x i8] ovflo)    -> class.hls::stream<bool>
  //   pointer-to-i64  (the [N x i64] BRAM arr) -> [N x complex<float>] / complex<float>
  // The i8 scalar `direction` is not a pointer, so it is left alone.
  auto newPointee = [&](Type *PT) -> Type * {
    auto *Ptr = dyn_cast<PointerType>(PT);
    if (!Ptr) return nullptr;
    Type *E = Ptr->getElementType();
    ArrayType *AT = dyn_cast<ArrayType>(E);
    Type *Elem = AT ? AT->getElementType() : E;
    if (auto *IT = dyn_cast<IntegerType>(Elem)) {
      if (IT->getBitWidth() == 128) return getSsrStreamTy();
      if (IT->getBitWidth() == 8) return getBoolStreamTy();
      if (IT->getBitWidth() == 64) {
        // BRAM array element: complex<float>. Preserve the array wrapper so the
        // arg stays a BRAM array (collapsing to a bare pointer would drop BRAM).
        StructType *Cx = getComplexFloatTy();
        return AT ? (Type *)ArrayType::get(Cx, AT->getNumElements())
                  : (Type *)Cx;
      }
    }
    return nullptr;
  };

  // 1. Retyped proxy declaration.
  FunctionType *POldFT = Proxy->getFunctionType();
  SmallVector<Type *, 8> PNew;
  bool ProxyChanged = false;
  for (unsigned i = 0; i < POldFT->getNumParams(); ++i) {
    Type *PT = POldFT->getParamType(i);
    Type *NE = newPointee(PT);
    if (NE) { PNew.push_back(PointerType::get(NE, 0)); ProxyChanged = true; }
    else PNew.push_back(PT);
  }
  if (!ProxyChanged) return false;
  FunctionType *PNewFT =
      FunctionType::get(POldFT->getReturnType(), PNew, POldFT->isVarArg());
  Function *NewProxy = Function::Create(PNewFT, Proxy->getLinkage(),
                                        (ProxyName + ".cx").str(), &M);
  NewProxy->copyAttributesFrom(Proxy);

  // 2. Clone each caller (FFT top) with the i128 stream args retyped.
  SmallPtrSet<Function *, 4> Tops;
  for (User *U : Proxy->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (Function *F = CI->getFunction()) Tops.insert(F);

  Type *I64 = Type::getInt64Ty(Ctx);
  for (Function *F : Tops) {
    FunctionType *OldFT = F->getFunctionType();
    unsigned N = OldFT->getNumParams();
    SmallVector<Type *, 8> NewParams;
    SmallVector<Type *, 8> NewElem(N, nullptr);
    for (unsigned i = 0; i < N; ++i) {
      Type *PT = OldFT->getParamType(i);
      Type *NE = newPointee(PT);
      if (NE) { NewElem[i] = NE; NewParams.push_back(PointerType::get(NE, 0)); }
      else NewParams.push_back(PT);
    }
    FunctionType *NewFT =
        FunctionType::get(F->getReturnType(), NewParams, F->isVarArg());
    Function *NewF = Function::Create(NewFT, F->getLinkage(),
                                      F->getName() + ".ssrcx", &M);
    NewF->copyAttributesFrom(F);
    { AttributeList AL = NewF->getAttributes();
      for (unsigned i = 0; i < N; ++i) if (NewElem[i]) AL = AL.removeParamAttributes(Ctx, i);
      NewF->setAttributes(AL); }
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F->getAllMetadata(MDs);
    for (auto &P : MDs) NewF->setMetadata(P.first, P.second);

    auto OldA = F->arg_begin();
    auto NewA = NewF->arg_begin();
    for (unsigned i = 0; i < N; ++i, ++OldA, ++NewA) {
      NewA->takeName(&*OldA);
      if (!NewElem[i]) {
        for (Attribute A : F->getAttributes().getParamAttributes(i)) NewA->addAttr(A);
        OldA->replaceAllUsesWith(&*NewA);
        continue;
      }
      // Stream args: strip the explicit ap_fifo SpecInterface (the stream type
      // carries its own interface; keeping it -> SYN 201-504) and rewrite the
      // offset-0 pass-through GEP to the new stream pointee.
      SmallVector<CallInst *, 2> EraseCalls;
      SmallVector<Use *, 8> Uses;
      for (Use &U : OldA->uses()) Uses.push_back(&U);
      for (Use *U : Uses) {
        User *Usr = U->getUser();
        if (auto *CI = dyn_cast<CallInst>(Usr)) {
          if (Function *Cal = CI->getCalledFunction())
            if (Cal->getName() == "_ssdm_op_SpecInterface") {
              EraseCalls.push_back(CI);
              continue;
            }
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
          if (GEP->getPointerOperand() == &*OldA) {
            bool allZero = true;
            for (unsigned k = 1; k < GEP->getNumOperands(); ++k) {
              auto *C = dyn_cast<ConstantInt>(GEP->getOperand(k));
              if (!C || !C->isZero()) { allZero = false; break; }
            }
            if (allZero) {
              IRBuilder<> GB(GEP);
              // Array retype (BRAM, e.g. [N x complex<float>]): preserve the
              // original all-zero index list so the GEP still yields an element
              // pointer. Stream/struct retype: a single 0 index (pass-through to
              // the stream pointer).
              SmallVector<Value *, 4> Idx;
              if (isa<ArrayType>(NewElem[i]))
                for (unsigned k = 1; k < GEP->getNumOperands(); ++k)
                  Idx.push_back(ConstantInt::get(I64, 0));
              else
                Idx.push_back(ConstantInt::get(I64, 0));
              auto *NG = GB.CreateInBoundsGEP(NewElem[i], &*NewA, Idx, GEP->getName());
              GEP->replaceAllUsesWith(NG);
              GEP->eraseFromParent();
              continue;
            }
          }
        }
        U->set(&*NewA);
      }
      for (CallInst *CI : EraseCalls) CI->eraseFromParent();
    }

    SmallVector<CallInst *, 2> ProxyCalls;
    for (BasicBlock &BB : *NewF)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() == Proxy) ProxyCalls.push_back(CI);
    for (CallInst *CI : ProxyCalls) {
      SmallVector<Value *, 8> Args(CI->arg_begin(), CI->arg_end());
      IRBuilder<> CB(CI);
      CallInst *NC = CB.CreateCall(NewProxy, Args);
      NC->setCallingConv(CI->getCallingConv());
      CI->replaceAllUsesWith(NC);
      CI->eraseFromParent();
    }

    hlsrs::vxx::replaceFunctionKeepingName(F, NewF);
  }

  if (Proxy->use_empty()) {
    Proxy->eraseFromParent();
    NewProxy->setName(ProxyName);
  }
  vxxDbg() << "vxx: renamed FFT SSR stream kernel arg(s) to "
            "class.hls::stream<hls::vector<std::complex<float>, 2>>\n";
  return true;
}



// Run the SSR rename for both SSR proxies: the stream variant
// (barista_fft_proxy_ssr: i128* in/out -> vector streams) and the array variant
// (barista_fft_proxy_ssr_arr: i64* in/out arrays stay BRAM — newPointee only
// retypes i128 and i8, so the i64 arrays are untouched; only the i8* ovflo ->
// stream<bool>). Both share newPointee, so the array proxy's BRAM args are
// preserved while ovflo gets the stream<bool> read port.
bool renameFftSsrArgs(Module &M) {
  bool C = renameFftSsrArgsImpl(M, "barista_fft_proxy_ssr");
  C |= renameFftSsrArgsImpl(M, "barista_fft_proxy_ssr_arr");
  return C;
}



// FFT-SSR ovflo/status fixup. The ap_fifo ovflo (a dead `bool*`
// top arg) must lower to `SpecStream(arg, 0, 0)` (depth 0) + `SpecInterface(arg,
// "ap_fifo", ..., depth 1)`. Rust's ap_fifo(arg, 1) marker, however, lowers
// (injectApFifo) to SpecStream with depth 1 — and a non-zero SpecStream depth
// alongside the ap_fifo SpecInterface conflicts (SYN 201-504). This pass runs
// after injectApFifo and, for the i1* args of FFT-SSR tops (callers of
// barista_fft_proxy_ssr / _ssr_arr), forces their `_ssdm_SpecStream` depth
// operand (index 2) back to 0. Targeted to
// FFT-SSR tops so no other ap_fifo example is touched.
bool fixFftSsrOvfloSpecStreamDepth(Module &M) {
  SmallPtrSet<Function *, 4> Tops;
  for (const char *PName : {"barista_fft_proxy_ssr", "barista_fft_proxy_ssr_arr"})
    if (Function *PF = M.getFunction(PName))
      for (User *U : PF->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (Function *F = CI->getFunction()) Tops.insert(F);
  if (Tops.empty()) return false;

  bool Changed = false;
  for (Function *F : Tops) {
    // (a) Erase the `_ssdm_SpecStream` on the i1* (ovflo/status) arg. The
    // csynth INPUT should have only SpecInterface(ap_fifo) on ovflo; the backend
    // ADDS SpecStream(0,0) itself. Pre-emitting SpecStream here (from the Rust
    // ap_fifo marker) makes the backend double-spec the port -> SYN 201-504. Keep
    // only SpecInterface(ap_fifo) so the backend recreates the right shape.
    SmallVector<CallInst *, 2> DeadSS;
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Cal = CI->getCalledFunction())
            if (Cal->getName() == "_ssdm_SpecStream" && CI->arg_size() >= 1) {
              auto *PT = dyn_cast<PointerType>(CI->getArgOperand(0)->getType());
              if (PT && PT->getElementType()->isIntegerTy(1)) DeadSS.push_back(CI);
            }
    for (CallInst *CI : DeadSS) { CI->eraseFromParent(); Changed = true; }
    // (b) Erase dead `bitcast i1* %ovflo to [N x i8]*` bridges left by
    // decayKernelArrayParams. A live [N x i8]* view makes HLS infer an
    // ap_memory interface on ovflo, conflicting with ap_fifo (SYN 201-504).
    for (Argument &A : F->args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT || !PT->getElementType()->isIntegerTy(1)) continue;
      SmallVector<BitCastInst *, 2> DeadCasts;
      for (User *U : A.users())
        if (auto *BC = dyn_cast<BitCastInst>(U))
          if (BC->use_empty()) DeadCasts.push_back(BC);
      for (BitCastInst *BC : DeadCasts) { BC->eraseFromParent(); Changed = true; }
    }
  }
  if (Changed)
    vxxDbg() << "vxx: fixed FFT-SSR ovflo/status (erased pre-emitted "
              "SpecStream + dead array-decay bitcast; reflow re-adds SpecStream)\n";
  return Changed;
}



// __vxx_fft_ssr(in, out, nfft_max, ssr, len) → SpecIPCore(Vivado_FFT, ...).
// For now, drop the marker — the Rust ssr_fft wrapper does the data-mover
// work, and the actual FFT IP wiring is handled separately.
bool injectFftSsrIp(Module &M) {
  return hlsrs::vxx::dropMarkerCallsAndDefinition(M, "__vxx_fft_ssr");
}



// __vxx_apf_{from_float,to_float,acc} → _ssdm_op_FloatingPoint_* intrinsics,
// matching the <ap_float.h> lowering:
//   from_f32: %r = call iW _ssdm_op_FloatingPoint_FloatToFloat.iW.i32(i32 bits, i32 8, i32 E)
//   to_f32:   %r = call i32 _ssdm_op_FloatingPoint_FloatToFloat.i32.iW(iW v,  i32 E, i32 8)
//   acc:      %r = call iW _ssdm_op_FloatingPoint_Acc.iW.p0iOW.iW(iOW* sum, iW v, i1 last, i32 E, i32 OI, i32 II)
// Without this, the accumulate op is deleted → output reg is undef → RTL all-X
// (using_ap_float_accumulator cosim 212-361). The Rust markers carry the
// values via pointers (ApFloat<W,E> = iW alloca; BitFixed<OW,OI> = iOW alloca).
bool injectApFloatIntrinsics(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I1 = Type::getInt1Ty(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);

  auto constU = [&](Value *V) -> uint64_t {
    return cast<ConstantInt>(V)->getZExtValue();
  };
  // Bitcast a pointer arg (often i8*) to a pointer-to-Ty in the arg's addrspace.
  auto asPtrTo = [&](IRBuilder<> &B, Value *Ptr, Type *Ty) -> Value * {
    unsigned AS = cast<PointerType>(Ptr->getType())->getAddressSpace();
    Type *Want = PointerType::get(Ty, AS);
    return Ptr->getType() == Want ? Ptr : B.CreateBitCast(Ptr, Want);
  };

  bool Changed = false;

  // ---- from_float(out:*ptr(iW), val:f32, w, e) ----
  if (Function *MF = M.getFunction("__vxx_apf_from_float")) {
    SmallVector<CallInst *, 8> Calls;
    for (User *U : MF->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      if (CI->arg_size() != 4) continue;
      IRBuilder<> B(CI);
      Value *Out = CI->getArgOperand(0);
      Value *Val = CI->getArgOperand(1);
      uint64_t W = constU(CI->getArgOperand(2));
      uint64_t E = constU(CI->getArgOperand(3));
      Type *IW = IntegerType::get(Ctx, (unsigned)W);
      Value *Bits = Val->getType()->isFloatTy() ? B.CreateBitCast(Val, I32)
                                                : B.CreateZExtOrTrunc(Val, I32);
      std::string Nm = "_ssdm_op_FloatingPoint_FloatToFloat.i" +
                       std::to_string(W) + ".i32";
      FunctionCallee Fn = M.getOrInsertFunction(
          Nm, FunctionType::get(IW, {I32, I32, I32}, false));
      if (auto *F = dyn_cast<Function>(Fn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
      Value *R = B.CreateCall(Fn, {Bits, ConstantInt::get(I32, 8),
                                   ConstantInt::get(I32, E)});
      B.CreateStore(R, asPtrTo(B, Out, IW));
      CI->eraseFromParent();
      Changed = true;
    }
    if (MF->use_empty()) MF->eraseFromParent();
  }

  // ---- to_float(out:*f32, val:*ptr(iW), w, e) ----
  if (Function *MF = M.getFunction("__vxx_apf_to_float")) {
    SmallVector<CallInst *, 8> Calls;
    for (User *U : MF->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      if (CI->arg_size() != 4) continue;
      IRBuilder<> B(CI);
      Value *Out = CI->getArgOperand(0);
      Value *ValP = CI->getArgOperand(1);
      uint64_t W = constU(CI->getArgOperand(2));
      uint64_t E = constU(CI->getArgOperand(3));
      Type *IW = IntegerType::get(Ctx, (unsigned)W);
      Value *V = B.CreateLoad(IW, asPtrTo(B, ValP, IW));
      std::string Nm = "_ssdm_op_FloatingPoint_FloatToFloat.i32.i" +
                       std::to_string(W);
      FunctionCallee Fn = M.getOrInsertFunction(
          Nm, FunctionType::get(I32, {IW, I32, I32}, false));
      if (auto *F = dyn_cast<Function>(Fn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
      Value *R = B.CreateCall(Fn, {V, ConstantInt::get(I32, E),
                                   ConstantInt::get(I32, 8)});
      Value *F = B.CreateBitCast(R, FloatTy);
      B.CreateStore(F, asPtrTo(B, Out, FloatTy));
      CI->eraseFromParent();
      Changed = true;
    }
    if (MF->use_empty()) MF->eraseFromParent();
  }

  // ---- acc(out:*ptr(iW), sum:*ptr(iOW), val:*ptr(iW), last, w,e,ow,oi,ii) ----
  if (Function *MF = M.getFunction("__vxx_apf_acc")) {
    SmallVector<CallInst *, 8> Calls;
    for (User *U : MF->users())
      if (auto *CI = dyn_cast<CallInst>(U)) Calls.push_back(CI);
    // The Rust `self.sum: BitFixed<OW,OI>` alloca is i128-backed; HLS rejects a
    // bitcast i128* → iOW* ("unsupported pointer reinterpretation"). Instead use
    // a *dedicated* `alloca iOW, align 512` register that
    // the FloatingPoint_Acc IP reads/writes (align 512 is required by the IP).
    // Map each distinct sum alloca → one such register, shared across its calls.
    DenseMap<Value *, Value *> AccReg;
    for (CallInst *CI : Calls) {
      if (CI->arg_size() != 9) continue;
      IRBuilder<> B(CI);
      Value *Out = CI->getArgOperand(0);
      Value *SumP = CI->getArgOperand(1);
      Value *ValP = CI->getArgOperand(2);
      Value *Last = CI->getArgOperand(3);
      uint64_t W = constU(CI->getArgOperand(4));
      uint64_t E = constU(CI->getArgOperand(5));
      uint64_t OW = constU(CI->getArgOperand(6));
      uint64_t OI = constU(CI->getArgOperand(7));
      uint64_t IIv = constU(CI->getArgOperand(8));
      Type *IW = IntegerType::get(Ctx, (unsigned)W);
      Type *IOW = IntegerType::get(Ctx, (unsigned)OW);
      Value *V = B.CreateLoad(IW, asPtrTo(B, ValP, IW));
      // Dedicated accumulator register (align 512), one per source sum alloca.
      Value *Root = SumP->stripPointerCasts();
      Value *SumPtr = AccReg.lookup(Root);
      if (!SumPtr) {
        Function *Fn = CI->getFunction();
        IRBuilder<> EB(&Fn->getEntryBlock(), Fn->getEntryBlock().getFirstInsertionPt());
        AllocaInst *Reg = EB.CreateAlloca(IOW, nullptr, "apf_acc");
        Reg->setAlignment(Align(512));
        SumPtr = Reg;
        AccReg[Root] = Reg;
      }
      Value *L = Last->getType()->isIntegerTy(1)
                     ? Last
                     : B.CreateICmpNE(Last, ConstantInt::get(Last->getType(), 0));
      std::string Nm = "_ssdm_op_FloatingPoint_Acc.i" + std::to_string(W) +
                       ".p0i" + std::to_string(OW) + ".i" + std::to_string(W);
      FunctionCallee Fn = M.getOrInsertFunction(
          Nm, FunctionType::get(
                  IW, {PointerType::get(IOW, 0), IW, I1, I32, I32, I32}, false));
      if (auto *F = dyn_cast<Function>(Fn.getCallee())) F->addFnAttr(Attribute::NoUnwind);
      Value *R = B.CreateCall(Fn, {SumPtr, V, L, ConstantInt::get(I32, E),
                                   ConstantInt::get(I32, OI),
                                   ConstantInt::get(I32, IIv)});
      B.CreateStore(R, asPtrTo(B, Out, IW));
      CI->eraseFromParent();
      Changed = true;
    }
    if (MF->use_empty()) MF->eraseFromParent();
  }

  return Changed;
}



// C++'s clang applies `#pragma HLS unroll` on the DSP-cascade tap loop BEFORE
// reflow: a.pp.bc arrives FULLY UNROLLED — one _ssdm_op_DSP call per tap, each
// with its own i58* state (systolic_fir: %my_fir_dsp0 + %my_fir_dsp_full_0/1/2).
// The Rust rolled `for j in 1..TAPS { unroll(); dsp_full[j-1].mul_add(..) }`
// reaches VXXPrep with ONE marker call site; the marker lowering would create
// ONE state alloca which every backend-unrolled copy then shares — the DSP
// cascade mis-wires (a cosim-only failure; the C-model is per-value). Align
// the input with C++: fully unroll any loop containing a DSP-cascade marker
// BEFORE lowering the markers, so each unrolled copy is a distinct call site
// with its own state alloca. Gated to DSP-marker loops (the systolic examples).
bool unrollDspCascadeLoops(Module &M) {
  SmallPtrSet<Function *, 4> Fns;
  for (const char *Name :
       {"__vxx_dsp58_cascade_mul_add", "__vxx_dspcplx_cascade_mul_add"})
    if (Function *Marker = M.getFunction(Name))
      for (User *U : Marker->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          Fns.insert(CI->getFunction());
  if (Fns.empty()) return false;
  LLVMContext &Ctx = M.getContext();
  bool Changed = false;
  for (Function *F : Fns) {
    if (F->isDeclaration()) continue;
    // Cheap pre-gate: only touch functions that also carry an `unroll()`
    // marker at all (the hand-unrolled forms have DSP markers but no rolled
    // tap loop — leave them byte-identical).
    bool AnyUnrollMarker = false;
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() &&
              CI->getCalledFunction()->getName() == "__vxx_loop_unroll") {
            AnyUnrollMarker = true;
            break;
          }
    vxxDbg() << "[dsp-unroll] fn=" << F->getName() << " anyUnroll=" << AnyUnrollMarker << "\n";
    if (!AnyUnrollMarker) continue;

    // Phase A — canonicalise FIRST, exactly as simplifyKernelLoops does for
    // TOP functions: the cpp_proxy adapter (rust_<top>) is NOT a top fn, so
    // its loops arrive in raw rustc form (trip count behind un-promoted
    // allocas, un-rotated). Marking metadata BEFORE this loses it when
    // LoopRotate rebuilds the latch branch (the kbcp/cosim kernel silently
    // kept its rolled tap loop while the kbc/Stage-A one unrolled →
    // cosim-only mis-wiring). SROA is safe pre-lowering: the DSP marker's
    // &out pointers are escaping call args, so their allocas are left alone.
    {
      legacy::FunctionPassManager FPM(&M);
      FPM.add(createSROAPass());
      FPM.add(createPromoteMemoryToRegisterPass());
      // Fold the iterator's speculated `select` increments (SimplifyCFG's
      // spec.select) into plain adds — SCEV cannot form an add-rec through a
      // select, which left the cpp_proxy adapter's tap loop with TC=0.
      FPM.add(createInstructionCombiningPass());
      FPM.add(createLoopSimplifyPass());
      FPM.add(createLoopRotatePass());
      FPM.add(createIndVarSimplifyPass());
      // JumpThreading + CFGSimplify resolve the Range iterator's loop-carried
      // continue-flag phi (`j += zext(flag)` with flag always-true on the
      // backedge) into a plain `j += 1` — without this SCEV sees a variable
      // step and cannot compute the trip count (the exact canonicalisation
      // simplifyKernelLoops applies to TOP functions).
      FPM.add(createJumpThreadingPass());
      // CVP (LazyValueInfo) proves the Range iterator's continue-flag phi is
      // TRUE on the backedge (the latch only re-enters when its condition
      // held), folding `j += zext(flag)` into `j += 1` — JumpThreading cannot
      // (the phi feeds a zext, not a branch).
      FPM.add(createCorrelatedValuePropagationPass());
      FPM.add(createCFGSimplificationPass());
      FPM.add(createInstructionCombiningPass());
      FPM.add(createLoopSimplifyPass());
      FPM.add(createIndVarSimplifyPass());
      FPM.doInitialization();
      FPM.run(*F);
      FPM.doFinalization();
    }

    // Phase B — on the CANONICAL form, find the tap loops and mark them.
    // Target = the innermost loop containing BOTH a DSP-cascade marker AND
    // the user's `unroll()` marker — exactly the loop C++ tags with
    // `#pragma HLS unroll`. A DSP call alone must NOT widen the gate: the
    // cascade HEAD (dsp0) sits directly in the outer streaming loop
    // (LOOP_FIR), which stays rolled in the C++ baseline.
    bool Marked = false;
    {
      DominatorTree DT(*F);
      LoopInfo LI;
      LI.analyze(DT);
      SmallPtrSet<Loop *, 4> HasDsp, HasUnroll;
      SmallVector<CallInst *, 8> UnrollMarkers;
      for (BasicBlock &BB : *F) {
        Loop *L = LI.getLoopFor(&BB);
        if (!L) continue;
        for (Instruction &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction()) continue;
          StringRef N = CI->getCalledFunction()->getName();
          if (N == "__vxx_dsp58_cascade_mul_add" ||
              N == "__vxx_dspcplx_cascade_mul_add")
            HasDsp.insert(L);
          else if (N == "__vxx_loop_unroll") {
            HasUnroll.insert(L);
            UnrollMarkers.push_back(CI);
          }
        }
      }
      SmallPtrSet<Loop *, 4> Targets;
      for (Loop *L : HasDsp)
        if (HasUnroll.count(L))
          Targets.insert(L);
      vxxDbg() << "[dsp-unroll]   hasDsp=" << HasDsp.size() << " hasUnroll="
             << HasUnroll.size() << " targets=" << Targets.size() << "\n";
      if (Targets.empty()) continue;
      for (Loop *L : Targets) {
        BasicBlock *Latch = L->getLoopLatch();
        if (!Latch) continue;
        Instruction *Term = Latch->getTerminator();
        // Fresh loop-ID: keep existing non-unroll operands, add unroll.full.
        SmallVector<Metadata *, 4> Ops;
        Ops.push_back(nullptr); // self-ref placeholder
        if (MDNode *Existing = Term->getMetadata("llvm.loop"))
          for (unsigned i = 1; i < Existing->getNumOperands(); ++i) {
            bool IsUnrollOp = false;
            if (auto *N = dyn_cast<MDNode>(Existing->getOperand(i)))
              if (N->getNumOperands() > 0)
                if (auto *S = dyn_cast<MDString>(N->getOperand(0)))
                  IsUnrollOp = S->getString().startswith("llvm.loop.unroll");
            if (!IsUnrollOp)
              Ops.push_back(Existing->getOperand(i));
          }
        Ops.push_back(MDNode::get(
            Ctx, {MDString::get(Ctx, "llvm.loop.unroll.full")}));
        MDNode *NewID = MDNode::getDistinct(Ctx, Ops);
        NewID->replaceOperandWith(0, NewID);
        Term->setMetadata("llvm.loop", NewID);
        Marked = true;
      }
      // The user's `unroll()` marker inside a target loop is superseded by
      // the eager unroll (it must not survive as N cloned straight-line calls).
      for (CallInst *CI : UnrollMarkers)
        if (Loop *L = LI.getLoopFor(CI->getParent()))
          if (Targets.count(L))
            CI->eraseFromParent();
    }
    if (!Marked) continue;

    // Phase C — force-unroll the marked loops and merge the copies into one
    // straight-line block: Vitis's DSP-cascade fusion only recognises the tap
    // chain STRAIGHT-LINE in one block (the C++ a.pp.bc and the hand-unrolled
    // form both have all taps in a single block).
    // Direct UnrollLoop utility (SimpleLoopUnrollPass silently bailed on the
    // cpp_proxy adapter): Force + exact trip count, one marked loop at a time
    // (analyses are invalidated by each unroll).
    for (int Round = 0; Round < 4; ++Round) {
      DominatorTree DT(*F);
      LoopInfo LI;
      LI.analyze(DT);
      AssumptionCache AC(*F);
      TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
      TargetLibraryInfo TLI(TLII);
      ScalarEvolution SE(*F, TLI, AC, DT, LI);
      OptimizationRemarkEmitter ORE(F);
      TargetTransformInfo TTI(M.getDataLayout());
      Loop *Target = nullptr;
      for (Loop *L : LI.getLoopsInPreorder()) {
        if (MDNode *ID = L->getLoopID())
          for (unsigned i = 1; i < ID->getNumOperands(); ++i)
            if (auto *N = dyn_cast<MDNode>(ID->getOperand(i)))
              if (N->getNumOperands() > 0)
                if (auto *S = dyn_cast<MDString>(N->getOperand(0)))
                  if (S->getString() == "llvm.loop.unroll.full") {
                    Target = L;
                    break;
                  }
        if (Target) break;
      }
      if (!Target) break;
      unsigned TC = SE.getSmallConstantTripCount(Target);
      SmallVector<BasicBlock *, 4> Ex;
      Target->getExitingBlocks(Ex);
      vxxDbg() << "[dsp-unroll]   loop=" << Target->getHeader()->getName()
             << " TC=" << TC << " exiting=" << Ex.size()
             << " latch=" << (Target->getLoopLatch() != nullptr)
             << " preheader=" << (Target->getLoopPreheader() != nullptr) << "\n";
      if (!TC) {
        // Exact count unavailable: with a single exit the MAX backedge count
        // is exact enough for a full unroll of these tiny constant loops.
        unsigned MaxTC = SE.getSmallConstantMaxTripCount(Target);
        vxxDbg() << "[dsp-unroll]   maxTC=" << MaxTC << "\n";
        if (Ex.size() == 1 && MaxTC && MaxTC <= 64) {
          TC = MaxTC;
        } else {
          for (BasicBlock *BB : Target->getBlocks())
            BB->print(vxxDbg());
          break;
        }
      }
      UnrollLoopOptions ULO;
      ULO.Count = TC;
      ULO.TripCount = TC;
      ULO.Force = true;
      ULO.PeelCount = 0;
      ULO.AllowRuntime = false;
      ULO.AllowExpensiveTripCount = false;
      ULO.PreserveCondBr = false;
      ULO.PreserveOnlyFirst = false;
      ULO.TripMultiple = SE.getSmallConstantTripMultiple(Target);
      ULO.UnrollRemainder = false;
      ULO.ForgetAllSCEV = false;
      LoopUnrollResult R = UnrollLoop(Target, ULO, &LI, &SE, &DT, &AC, &TTI,
                                      &ORE, /*PreserveLCSSA=*/false, nullptr);
      vxxDbg() << "[dsp-unroll]   UnrollLoop result="
             << (R == LoopUnrollResult::FullyUnrolled ? "FULL"
                 : R == LoopUnrollResult::PartiallyUnrolled ? "PARTIAL"
                                                            : "UNMODIFIED")
             << "\n";
      if (R == LoopUnrollResult::Unmodified) break;
      Changed = true;
    }
    {
      legacy::FunctionPassManager FPM(&M);
      FPM.add(createCFGSimplificationPass());
      // Restore canonical loop form (preheader/latch) for the remaining
      // outer loops — later passes (injectAutoLoopName etc.) rely on it.
      FPM.add(createLoopSimplifyPass());
      FPM.doInitialization();
      FPM.run(*F);
      FPM.doFinalization();
    }
    // Mark for cleanupDspCascadeAllocas (below): the unrolled copies still
    // share the loop body's single R out-alloca, which must be dissolved
    // after the marker lowering.
    {
      unsigned NDsp = 0;
      for (BasicBlock &BB : *F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (CI->getCalledFunction() &&
                CI->getCalledFunction()->getName().startswith("__vxx_dsp"))
              ++NDsp;
      vxxDbg() << "[dsp-unroll]   post-unroll dsp-marker calls=" << NDsp << "\n";
    }
    F->addFnAttr("vxx.dsp.cascade.unrolled");
    vxxDbg() << "vxx: unrolled DSP-cascade loop(s) in " << F->getName() << "\n";
  }
  return Changed;
}

// Post-lowering cleanup for unrollDspCascadeLoops functions. The unroller
// clones the tap body but NOT its entry-block R out-alloca, so all taps share
// ONE temp: three stores to the same slots create WAW memory dependencies
// between the _ssdm_op_DSP ops, the scheduler cannot fuse them into the DSP58
// cascade, and the un-fused b-cascade skew mis-computes from the second
// output on (cosim-only). After injectDsp58Intrinsics the marker's out
// pointers are plain store/load slots (no more escaping call uses), so SROA
// dissolves the round trips into the pure SSA extract chain — exactly the
// C++ a.pp.bc form, which carries NO allocas at all.
bool cleanupDspCascadeAllocas(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("vxx.dsp.cascade.unrolled")) continue;
    F.removeFnAttr("vxx.dsp.cascade.unrolled");
    legacy::FunctionPassManager FPM(&M);
    FPM.add(createSROAPass());
    FPM.doInitialization();
    if (FPM.run(F)) Changed = true;
    FPM.doFinalization();
    vxxDbg() << "vxx: dissolved shared DSP-cascade temps in " << F.getName()
             << "\n";
  }
  return Changed;
}

// Retype DSPCPLX complex kernel ports from the Rust field-pair form to the
// C++ pp packed-integer form. clang lowers `std::complex<ap_int<18>>` /
// `std::complex<ap_int<58>>` ports to single packed integers before reflow
// (a.pp.bc: `i36* %coeff_0`, `[10 x i36]* %b`, `[10 x i116]* %hw`; field 0
// (re) in the LOW half, im in the high half), so the BRAM data ports come out
// 36/116 bits wide. The Rust `CmpxA {re: i32, im: i32}` / `CmpxC {re: i64,
// im: i64}` args stay `{i32,i32}` / `{i64,i64}` pairs, so the same ports come
// out 64/128 bits wide — a Stage A width mismatch (systolic_fir_using_complex
// b_q0/coeff 63:0 vs 35:0, hw_d0 127:0 vs 115:0).
//
// The DSPCPLX operand widths are fixed by the intrinsic itself (18/58 per
// channel), so no user-side width transport is needed: any top-function
// pointer arg to `{i32,i32}` packs to i36 and `[N x {i64,i64}]` to i116.
// A BARE `{i64,i64}*` arg is left alone — that is the by-reference C-operand
// scalar (`bias`), which the C++ side passes by value as i128 and whose port
// already matches.
//
// Conservative by construction: every use of a retyped arg must be a GEP
// whose last index is a constant field 0/1 feeding exactly one load or one
// store, and stores must come in complete re+im pairs on the same element;
// any other shape aborts the whole rewrite (leaving the wide-but-working
// form). Runs in Phase 7 right after the DSPCPLX marker lowering, i.e. after
// disaggCompletePartitionKernelSig split `coeff` into per-element args.
bool narrowDspCplxPortTypes(Module &M) {
  if (!M.getFunction("_ssdm_op_DSP.i36.i36.i116.i116")) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  StructType *PairA = StructType::get(Ctx, {I32, I32}); // CmpxA (18-bit lanes)
  StructType *PairC = StructType::get(Ctx, {I64, I64}); // CmpxC (58-bit lanes)

  struct Retype {
    Argument *Arg;
    Type *NewPtee;     // i36 / [N x i36] / [N x i116]
    IntegerType *Wide; // i36 / i116
    IntegerType *Half; // i18 / i58
    Type *LaneTy;      // i32 / i64 (original field type)
    bool IsArray;
  };

  bool Changed = false;
  SmallVector<Function *, 2> Tops;
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    if (!F.hasFnAttribute("fpga.top.func")) continue;
    if (!F.use_empty()) continue; // called tops: adapter ABIs must not shift
    Tops.push_back(&F);
  }

  for (Function *Old : Tops) {
    SmallVector<Retype, 8> Plan;
    for (Argument &A : Old->args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT) continue;
      Type *E = PT->getElementType();
      StructType *Pair = nullptr;
      Type *NewPtee = nullptr;
      bool IsArray = false;
      if (auto *AT = dyn_cast<ArrayType>(E)) {
        Pair = dyn_cast<StructType>(AT->getElementType());
        if (Pair == PairA)
          NewPtee = ArrayType::get(IntegerType::get(Ctx, 36), AT->getNumElements());
        else if (Pair == PairC)
          NewPtee = ArrayType::get(IntegerType::get(Ctx, 116), AT->getNumElements());
        else
          continue;
        IsArray = true;
      } else if (E == PairA) {
        // Per-element product of the complete-partition disagg.
        Pair = PairA;
        NewPtee = IntegerType::get(Ctx, 36);
      } else {
        continue; // bare {i64,i64}* (bias by-ref) and everything else: keep
      }
      unsigned HalfBits = (Pair == PairA) ? 18 : 58;
      Plan.push_back({&A, NewPtee,
                      IntegerType::get(Ctx, HalfBits * 2),
                      IntegerType::get(Ctx, HalfBits),
                      Pair->getElementType(0), IsArray});
    }
    if (Plan.empty()) continue;

    // Validate every use: GEP (last index = const field 0/1) -> 1 load/store.
    struct Access {
      GetElementPtrInst *GEP;
      Instruction *LS; // LoadInst or StoreInst
      uint64_t Field;  // 0 = re, 1 = im
      Value *ElemIdx;  // array element index (null for bare per-element args)
    };
    bool Bail = false;
    SmallVector<std::pair<const Retype *, SmallVector<Access, 8>>, 8> Work;
    for (const Retype &R : Plan) {
      SmallVector<Access, 8> Accs;
      for (User *U : R.Arg->users()) {
        auto *GEP = dyn_cast<GetElementPtrInst>(U);
        if (!GEP || !GEP->hasOneUse()) { Bail = true; break; }
        auto *FieldC = dyn_cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1));
        unsigned WantIdx = R.IsArray ? 3u : 2u; // (0, elem, f) / (0, f)
        if (!FieldC || GEP->getNumOperands() != WantIdx + 1 ||
            FieldC->getZExtValue() > 1) { Bail = true; break; }
        Instruction *LS = cast<Instruction>(*GEP->user_begin());
        if (!isa<LoadInst>(LS) &&
            !(isa<StoreInst>(LS) &&
              cast<StoreInst>(LS)->getPointerOperand() == GEP)) {
          Bail = true;
          break;
        }
        Accs.push_back({GEP, LS, FieldC->getZExtValue(),
                        R.IsArray ? GEP->getOperand(2) : nullptr});
      }
      if (Bail) break;
      Work.push_back({&R, std::move(Accs)});
    }
    // Stores must pair up re+im on the same element (single packed store).
    if (!Bail) {
      for (auto &W : Work) {
        SmallVector<Access *, 4> Stores;
        for (Access &Ac : W.second)
          if (isa<StoreInst>(Ac.LS)) Stores.push_back(&Ac);
        if (Stores.empty()) continue;
        if (Stores.size() % 2) { Bail = true; break; }
        for (Access *S : Stores) {
          unsigned Mates = 0;
          for (Access *T : Stores)
            if (T != S && T->ElemIdx == S->ElemIdx && T->Field != S->Field &&
                T->LS->getParent() == S->LS->getParent())
              ++Mates;
          if (Mates != 1) { Bail = true; break; }
        }
        if (Bail) break;
      }
    }
    if (Bail) continue;

    // Build the retyped clone: same body (spliced), new arg types.
    SmallVector<Type *, 8> NewParams;
    for (Argument &A : Old->args()) NewParams.push_back(A.getType());
    for (const Retype &R : Plan)
      NewParams[R.Arg->getArgNo()] =
          PointerType::get(R.NewPtee, R.Arg->getType()->getPointerAddressSpace());
    FunctionType *NewFT =
        FunctionType::get(Old->getReturnType(), NewParams, false);
    Function *NewF = Function::Create(NewFT, Old->getLinkage(), "", &M);
    NewF->copyAttributesFrom(Old);
    NewF->copyMetadata(Old, 0);
    NewF->getBasicBlockList().splice(NewF->begin(), Old->getBasicBlockList());
    {
      auto OI = Old->arg_begin();
      auto NI = NewF->arg_begin();
      for (; OI != Old->arg_end(); ++OI, ++NI) {
        NI->setName(OI->getName());
        bool Retyped = false;
        for (const Retype &R : Plan)
          if (R.Arg == &*OI) { Retyped = true; break; }
        if (!Retyped) OI->replaceAllUsesWith(&*NI);
      }
    }

    // Rewrite the retyped args' accesses on the spliced body.
    for (auto &W : Work) {
      const Retype &R = *W.first;
      Argument *NewArg = NewF->arg_begin() + R.Arg->getArgNo();
      // The packed width no longer matches the old byte-exact hints.
      NewF->removeParamAttr(R.Arg->getArgNo(), Attribute::Dereferenceable);
      NewF->removeParamAttr(R.Arg->getArgNo(), Attribute::Alignment);
      unsigned H = R.Half->getBitWidth();
      DenseMap<Instruction *, bool> Erased;
      for (Access &Ac : W.second) {
        IRBuilder<> B(Ac.LS);
        Value *ElemPtr;
        if (R.IsArray)
          ElemPtr = B.CreateInBoundsGEP(
              R.NewPtee, NewArg,
              {ConstantInt::get(Type::getInt64Ty(Ctx), 0), Ac.ElemIdx});
        else
          ElemPtr = NewArg;
        if (auto *LD = dyn_cast<LoadInst>(Ac.LS)) {
          Value *Packed = B.CreateLoad(R.Wide, ElemPtr);
          Value *Lane = Ac.Field == 0
                            ? B.CreateTrunc(Packed, R.Half)
                            : B.CreateTrunc(B.CreateLShr(Packed, H), R.Half);
          LD->replaceAllUsesWith(B.CreateSExt(Lane, R.LaneTy));
          LD->eraseFromParent();
        } else if (!Erased.count(Ac.LS)) {
          // Find the mate (validated above) and fuse into one packed store.
          Access *Mate = nullptr;
          for (Access &T : W.second)
            if (&T != &Ac && isa<StoreInst>(T.LS) && T.ElemIdx == Ac.ElemIdx &&
                T.Field != Ac.Field && !Erased.count(T.LS)) {
              Mate = &T;
              break;
            }
          Access *Re = Ac.Field == 0 ? &Ac : Mate;
          Access *Im = Ac.Field == 0 ? Mate : &Ac;
          Value *ReV = cast<StoreInst>(Re->LS)->getValueOperand();
          Value *ImV = cast<StoreInst>(Im->LS)->getValueOperand();
          // Insert at the later store so both lane values dominate the pack.
          Instruction *Later =
              Ac.LS->comesBefore(Mate->LS) ? Mate->LS : Ac.LS;
          IRBuilder<> BS(Later);
          Value *Lo = BS.CreateZExt(BS.CreateTrunc(ReV, R.Half), R.Wide);
          Value *Hi = BS.CreateShl(
              BS.CreateZExt(BS.CreateTrunc(ImV, R.Half), R.Wide), H);
          BS.CreateStore(BS.CreateOr(Lo, Hi), ElemPtr);
          Erased[Ac.LS] = Erased[Mate->LS] = true;
          Ac.LS->eraseFromParent();
          Mate->LS->eraseFromParent();
        }
        if (Ac.GEP->use_empty()) Ac.GEP->eraseFromParent();
      }
      vxxDbg() << "vxx: packed DSPCPLX port " << NewArg->getName() << " to i"
               << R.Wide->getBitWidth() << " in " << Old->getName() << "\n";
    }

    std::string Name = Old->getName().str();
    Old->eraseFromParent();
    NewF->setName(Name);
    Changed = true;
  }
  return Changed;
}

// Rewrite `__vxx_dsp58_cascade_mul_add(flags, a, b, c, &acout, &bcout,
// &pcout)` markers to the integer DSP58 intrinsic — the C++ baseline shape
// (systolic_fir a.pp.bc):
//   %res = call { i27, i24, i58 } @_ssdm_op_DSP.i27.i24.i58.i27(
//     i32 10,            ; op = integer DSP58 mul-add
//     i64 %flags,        ; REG_* pipeline flags (325 head / 333 tail)
//     i27 0,             ; D port (unused)
//     i27 %a, i24 %b, i58 %c,
//     i1 false, i58* %state)   ; per-instance P-cascade state alloca
// with sign-extended extracts stored to the out pointers. Cascade stage
// fusion feeds a previous stage's extracted bcout/pcout DIRECTLY into the
// next call (the C++ clang chains extractvalue into the next _ssdm_op_DSP;
// an unpack->store->load->trunc round trip breaks Vitis's DSP cascade
// recognition, as established for DSPCPLX).
bool injectDsp58Intrinsics(Module &M) {
  Function *Marker = M.getFunction("__vxx_dsp58_cascade_mul_add");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I1 = Type::getInt1Ty(Ctx);
  Type *I24 = IntegerType::get(Ctx, 24);
  Type *I27 = IntegerType::get(Ctx, 27);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I58 = IntegerType::get(Ctx, 58);
  Type *I64 = Type::getInt64Ty(Ctx);

  StructType *DspRetTy = StructType::get(Ctx, {I27, I24, I58});
  FunctionType *DspTy = FunctionType::get(
      DspRetTy, {I32, I64, I27, I27, I24, I58, I1, PointerType::get(I58, 0)},
      false);
  FunctionCallee DspFn =
      M.getOrInsertFunction("_ssdm_op_DSP.i27.i24.i58.i27", DspTy);
  if (auto *F = dyn_cast<Function>(DspFn.getCallee()))
    F->addFnAttr(Attribute::NoUnwind);

  // Program order matters for the cascade stage fusion below.
  SmallVector<CallInst *, 8> Calls;
  {
    SmallPtrSet<CallInst *, 8> CallSet;
    for (User *U : Marker->users())
      if (auto *CI = dyn_cast<CallInst>(U)) CallSet.insert(CI);
    SmallPtrSet<Function *, 4> Fns;
    for (CallInst *CI : CallSet) Fns.insert(CI->getFunction());
    for (Function &F : M) {
      if (!Fns.count(&F)) continue;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (CallSet.count(CI)) Calls.push_back(CI);
    }
  }
  if (Calls.empty()) {
    if (Marker->use_empty()) Marker->eraseFromParent();
    return false;
  }

  const DataLayout &DLf = M.getDataLayout();
  auto resolvePtr = [&](Value *P) -> std::pair<Value *, int64_t> {
    int64_t Off = 0;
    while (true) {
      if (auto *BC = dyn_cast<BitCastInst>(P)) { P = BC->getOperand(0); continue; }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(P)) {
        APInt GOff(DLf.getPointerSizeInBits(0), 0);
        if (!GEP->accumulateConstantOffset(DLf, GOff))
          return {nullptr, 0};
        Off += GOff.getSExtValue();
        P = GEP->getPointerOperand();
        continue;
      }
      break;
    }
    return {P, Off};
  };
  // (base, byte offset) an earlier stage stored bcout/pcout into -> the
  // stage's extracted i24/i58 value, fed directly to the next call.
  DenseMap<std::pair<Value *, int64_t>, Value *> FieldSrc;
  auto lookupFused = [&](Value *Arg) -> Value * {
    auto *L = dyn_cast<LoadInst>(Arg);
    if (!L) return nullptr;
    auto K = resolvePtr(L->getPointerOperand());
    if (!K.first) return nullptr;
    auto It = FieldSrc.find(K);
    return It == FieldSrc.end() ? nullptr : It->second;
  };

  for (CallInst *CI : Calls) {
    if (CI->arg_size() != 7) continue;
    IRBuilder<> B(CI);
    Value *Flags = CI->getArgOperand(0);
    Value *A = CI->getArgOperand(1);
    Value *Bv = CI->getArgOperand(2);
    Value *Cv = CI->getArgOperand(3);
    Value *OutAc = CI->getArgOperand(4);
    Value *OutBc = CI->getArgOperand(5);
    Value *OutPc = CI->getArgOperand(6);

    // Per-call-site P-cascade state alloca (the C++ shape passes each
    // cascade object's member; one call site = one cascade instance here).
    Function *F = CI->getFunction();
    IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
    Value *State = EB.CreateAlloca(I58, nullptr, "dsp58_state");

    Value *A27 = B.CreateTrunc(A, I27);
    Value *B24 = lookupFused(Bv);
    if (!B24)
      B24 = B.CreateTrunc(Bv, I24);
    Value *C58 = lookupFused(Cv);
    if (!C58)
      C58 = B.CreateTrunc(Cv, I58);

    Value *DspArgs[] = {
        ConstantInt::get(I32, 10), // op = integer DSP58 mul-add
        Flags,
        ConstantInt::get(I27, 0), // D port unused
        A27, B24, C58,
        ConstantInt::get(I1, 0),
        State,
    };
    Value *Res = B.CreateCall(DspFn, DspArgs);
    Value *Acout = B.CreateExtractValue(Res, 0); // i27
    Value *Bcout = B.CreateExtractValue(Res, 1); // i24
    Value *Pcout = B.CreateExtractValue(Res, 2); // i58

    B.CreateStore(B.CreateSExt(Acout, I64), OutAc);
    B.CreateStore(B.CreateSExt(Bcout, I64), OutBc);
    B.CreateStore(B.CreateSExt(Pcout, I64), OutPc);

    auto KB = resolvePtr(OutBc);
    if (KB.first) FieldSrc[KB] = Bcout;
    auto KP = resolvePtr(OutPc);
    if (KP.first) FieldSrc[KP] = Pcout;

    CI->eraseFromParent();
  }
  if (Marker->use_empty()) Marker->eraseFromParent();
  return true;
}


// Rewrite `__vxx_dspcplx_cascade_mul_add(flags, a_re, a_im, b_re, b_im,
//   c_re, c_im, &out_acout_re, &out_acout_im, &out_bcout_re, &out_bcout_im,
//   &out_pcout_re, &out_pcout_im)` markers to C++-equivalent DSP intrinsic:
//
//   %ar18 = trunc i32 %a_re to i18
//   %ai18 = trunc i32 %a_im to i18
//   %a_packed = call i36 @_ssdm_op_BitConcatenate.i36.i18.i18(i18 %ar18, i18 %ai18)
//   %b_packed = same for b
//   %c_packed = same for c (i64 → i58 truncs, then BitConcat to i116)
//   %res = call { i36, i36, i116 } @_ssdm_op_DSP.i36.i36.i116.i116(
//     i32 16,            ; op = DSPCPLX mul-add
//     i64 %flags,
//     i116 0,            ; (initial?)
//     i36 %a_packed, i36 %b_packed, i116 %c_packed,
//     i1 false, i116* null)
//   %acout = extractvalue %res, 0  ; i36
//   %bcout = extractvalue %res, 1  ; i36
//   %pcout = extractvalue %res, 2  ; i116
//   ; unpack and store
//   %ar_lo = trunc i36 %acout to i18; %ar_zext = zext i18 %ar_lo to i32; store
//   %ar_hi = lshr i36 %acout, 18; trunc; zext; store (acout_im)
//   ... same for bcout/pcout
//
// Without this, library Cascade::mul_add inlines to raw multiply/add (no DSP
// intrinsic), so HLS can't synthesize it as DSPCPLX and the port shape is wrong.
bool injectDspCplxIntrinsics(Module &M) {
  Function *Marker = M.getFunction("__vxx_dspcplx_cascade_mul_add");
  if (!Marker) return false;
  LLVMContext &Ctx = M.getContext();
  Type *I1 = Type::getInt1Ty(Ctx);
  Type *I18 = IntegerType::get(Ctx, 18);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I36 = IntegerType::get(Ctx, 36);
  Type *I58 = IntegerType::get(Ctx, 58);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I116 = IntegerType::get(Ctx, 116);

  // Declare helper intrinsics + DSP intrinsic
  FunctionType *BC36Ty = FunctionType::get(I36, {I18, I18}, false);
  FunctionCallee BC36 = M.getOrInsertFunction("_ssdm_op_BitConcatenate.i36.i18.i18", BC36Ty);
  FunctionType *BC116Ty = FunctionType::get(I116, {I58, I58}, false);
  FunctionCallee BC116 = M.getOrInsertFunction("_ssdm_op_BitConcatenate.i116.i58.i58", BC116Ty);
  if (auto *F = dyn_cast<Function>(BC36.getCallee())) F->addFnAttr(Attribute::NoUnwind);
  if (auto *F = dyn_cast<Function>(BC116.getCallee())) F->addFnAttr(Attribute::NoUnwind);

  StructType *DspRetTy = StructType::get(Ctx, {I36, I36, I116});
  FunctionType *DspTy = FunctionType::get(DspRetTy,
      {I32, I64, I116, I36, I36, I116, I1, PointerType::get(I116, 0)}, false);
  FunctionCallee DspFn = M.getOrInsertFunction("_ssdm_op_DSP.i36.i36.i116.i116", DspTy);
  if (auto *F = dyn_cast<Function>(DspFn.getCallee())) F->addFnAttr(Attribute::NoUnwind);

  SmallVector<CallInst *, 8> Calls;
  {
    // Program order matters for the cascade stage fusion below (stage N's
    // outputs feed stage N+1); Marker->users() order is arbitrary.
    SmallPtrSet<CallInst *, 8> CallSet;
    for (User *U : Marker->users())
      if (auto *CI = dyn_cast<CallInst>(U)) CallSet.insert(CI);
    SmallPtrSet<Function *, 4> Fns;
    for (CallInst *CI : CallSet) Fns.insert(CI->getFunction());
    for (Function &F : M) {
      if (!Fns.count(&F)) continue;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (CallSet.count(CI)) Calls.push_back(CI);
    }
  }
  if (Calls.empty()) {
    if (Marker->use_empty()) Marker->eraseFromParent();
    return false;
  }

  // Cascade stage fusion: maps a (base pointer, byte offset) an earlier stage
  // stored an unpacked bcout/pcout lane into -> (that stage's PACKED result,
  // lane 0=lo/1=hi). When the next stage's b/c operands are loads of both
  // lanes of the same packed value, feed the packed i36/i116 DIRECTLY —
  // exactly the C++ clang shape (`extractvalue` chained into the next
  // `_ssdm_op_DSP` call). The unpack->store->load->repack round trip breaks
  // Vitis's DSPCPLX cascade recognition and the RTL mis-wires
  // (systolic_fir_using_complex: every output wrong vs the C++ baseline's
  // delayed-FIR golden, while the same TB passes on the C++ IR).
  const DataLayout &DLf = M.getDataLayout();
  auto resolvePtr = [&](Value *P) -> std::pair<Value *, int64_t> {
    int64_t Off = 0;
    while (true) {
      if (auto *BC = dyn_cast<BitCastInst>(P)) { P = BC->getOperand(0); continue; }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(P)) {
        APInt GOff(DLf.getPointerSizeInBits(0), 0);
        if (!GEP->accumulateConstantOffset(DLf, GOff))
          return {nullptr, 0};
        Off += GOff.getSExtValue();
        P = GEP->getPointerOperand();
        continue;
      }
      break;
    }
    return {P, Off};
  };
  DenseMap<std::pair<Value *, int64_t>, std::pair<Value *, int>> FieldSrc;
  auto lookupPacked2 = [&](Value *ReArg, Value *ImArg) -> Value * {
    auto *ReL = dyn_cast<LoadInst>(ReArg);
    auto *ImL = dyn_cast<LoadInst>(ImArg);
    if (!ReL || !ImL) return nullptr;
    auto RK = resolvePtr(ReL->getPointerOperand());
    auto IK = resolvePtr(ImL->getPointerOperand());
    if (!RK.first || !IK.first) return nullptr;
    auto RIt = FieldSrc.find(RK);
    auto IIt = FieldSrc.find(IK);
    if (RIt == FieldSrc.end() || IIt == FieldSrc.end()) return nullptr;
    if (RIt->second.first != IIt->second.first) return nullptr;
    if (RIt->second.second != 0 || IIt->second.second != 1) return nullptr;
    return RIt->second.first;
  };

  for (CallInst *CI : Calls) {
    if (CI->arg_size() != 13) continue;
    IRBuilder<> B(CI);
    Value *Flags = CI->getArgOperand(0);
    Value *ARe   = CI->getArgOperand(1);
    Value *AIm   = CI->getArgOperand(2);
    Value *BRe   = CI->getArgOperand(3);
    Value *BIm   = CI->getArgOperand(4);
    Value *CRe   = CI->getArgOperand(5);
    Value *CIm   = CI->getArgOperand(6);
    Value *OutAcRe = CI->getArgOperand(7);
    Value *OutAcIm = CI->getArgOperand(8);
    Value *OutBcRe = CI->getArgOperand(9);
    Value *OutBcIm = CI->getArgOperand(10);
    Value *OutPcRe = CI->getArgOperand(11);
    Value *OutPcIm = CI->getArgOperand(12);

    // Pack a_re/a_im (i32) → i36
    Value *ARe18 = B.CreateTrunc(ARe, I18);
    Value *AIm18 = B.CreateTrunc(AIm, I18);
    Value *APacked = B.CreateCall(BC36, {ARe18, AIm18});
    // Pack b — or feed the PREVIOUS stage's packed bcout directly (cascade
    // stage fusion, the C++ clang shape; see FieldSrc above).
    Value *BPacked = lookupPacked2(BRe, BIm);
    if (!BPacked) {
      Value *BRe18 = B.CreateTrunc(BRe, I18);
      Value *BIm18 = B.CreateTrunc(BIm, I18);
      BPacked = B.CreateCall(BC36, {BRe18, BIm18});
    }
    // Pack c — or feed the previous stage's packed pcout directly.
    Value *CPacked = lookupPacked2(CRe, CIm);
    if (!CPacked) {
      Value *CRe58 = B.CreateTrunc(CRe, I58);
      Value *CIm58 = B.CreateTrunc(CIm, I58);
      CPacked = B.CreateCall(BC116, {CRe58, CIm58});
    }

    // Call DSP intrinsic
    Value *DspArgs[] = {
        ConstantInt::get(I32, 16),               // op = DSPCPLX mul-add
        Flags,                                    // flags (REG_M/REG_P/PIPELINE)
        ConstantInt::get(I116, 0),                // (initial?)
        APacked, BPacked, CPacked,
        ConstantInt::get(I1, 0),                  // last = false
        ConstantPointerNull::get(PointerType::get(I116, 0))  // pcout_alloca = null
    };
    Value *Res = B.CreateCall(DspFn, DspArgs);

    // Extract outputs
    Value *Acout = B.CreateExtractValue(Res, 0);  // i36
    Value *Bcout = B.CreateExtractValue(Res, 1);  // i36
    Value *Pcout = B.CreateExtractValue(Res, 2);  // i116

    // Unpack acout (i36) into re (low i18) + im (high i18), SIGN-extend to
    // i32, store. The 18-bit lanes are signed (ap_int<18>): zext turned a
    // negative lane into a large positive (e.g. -5 -> 2^18-5), which the
    // faithful systolic_fir_using_complex TB exposed as corrupted RTL
    // outputs on any negative product (2^58-5-style hw values).
    auto unpackAndStore36 = [&](Value *Packed, Value *OutRe, Value *OutIm) {
      Value *ReLo = B.CreateTrunc(Packed, I18);
      Value *ReZ = B.CreateSExt(ReLo, I32);
      B.CreateStore(ReZ, OutRe);
      Value *Sh = B.CreateLShr(Packed, ConstantInt::get(I36, 18));
      Value *ImLo = B.CreateTrunc(Sh, I18);
      Value *ImZ = B.CreateSExt(ImLo, I32);
      B.CreateStore(ImZ, OutIm);
    };
    unpackAndStore36(Acout, OutAcRe, OutAcIm);
    unpackAndStore36(Bcout, OutBcRe, OutBcIm);
    // Register this stage's outputs for the next stage's fusion lookup.
    {
      auto RB = resolvePtr(OutBcRe);
      auto IB = resolvePtr(OutBcIm);
      if (RB.first && IB.first) {
        FieldSrc[RB] = {Bcout, 0};
        FieldSrc[IB] = {Bcout, 1};
      }
      auto RP = resolvePtr(OutPcRe);
      auto IP = resolvePtr(OutPcIm);
      if (RP.first && IP.first) {
        FieldSrc[RP] = {Pcout, 0};
        FieldSrc[IP] = {Pcout, 1};
      }
    }

    // Unpack pcout (i116): the DSPCPLX P accumulator's lanes are REVERSED
    // relative to the a/b operand packing — LOW 58 bits carry the IMAGINARY
    // part, HIGH 58 the REAL part. (Established empirically: with the
    // operand packs matching the C++ clang shape byte-for-byte, the RTL's
    // outputs came back with re/im exactly swapped on every transaction of
    // the C++-faithful systolic_fir_using_complex golden until this unpack
    // was flipped.) SIGN-extend both 58-bit lanes to i64.
    Value *PcImLo = B.CreateTrunc(Pcout, I58);
    Value *PcImZ = B.CreateSExt(PcImLo, I64);
    B.CreateStore(PcImZ, OutPcIm);
    Value *PcSh = B.CreateLShr(Pcout, ConstantInt::get(I116, 58));
    Value *PcReHi = B.CreateTrunc(PcSh, I58);
    Value *PcReZ = B.CreateSExt(PcReHi, I64);
    B.CreateStore(PcReZ, OutPcRe);

    CI->eraseFromParent();
  }
  if (Marker->use_empty()) Marker->eraseFromParent();
  return true;
}



} } // namespace hlsrs::vxx
