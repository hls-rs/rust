// VXXIRDowngrader — strip LLVM 11-only constructs from a Module so the
// downstream v++ LLVM 7 bitcode reader can consume it.
//
// Migrated from rustc's PassWrapper.cpp::LLVMRustVitisStripIncompatibleAttrs
// as part of the A2 refactor (Pass-ified + dlopen'd shared library).
//
// Two responsibilities:
//   1. Strip function attributes that LLVM 11 introduced but LLVM 7's
//      verifier rejects (NoFree, NoRecurse, NoSync, NoUndef, WillReturn,
//      ImmArg). Was previously a per-example Makefile sed step.
//   2. Rename anonymous SSA values (`%0`, `%1`, ...) to letter-prefixed
//      names. Works around an LLVM 7 parser bug where
//      `dereferenceable(N) %<numeric>` in a parameter list fails to parse.

#include "VXXPrep.h"
#include "VXXShared.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>

#include <string>
#include <cstdlib>

using namespace llvm;

// Partial-prune LLVM 11 debug info to LLVM 7-compatible shape.
// LLVM 7's llvm-as can't parse `spFlags:` (LLVM 11 introduced), nor
// `retainedNodes:` (LLVM 11 renamed; was `variables:` in 7), nor
// `checksumkind:`/`checksum:` on DIFile, nor `nameTableKind:` on
// DICompileUnit.
//
// Strategy: walk DI metadata and replace LLVM 11-only fields with
// LLVM 7-equivalent forms (or clear them) via in-place mutation
// using the immutable-metadata RAUW pattern.
//
// NOTE: AsmWriter still emits `spFlags:` if ANY of isDefinition/isLocal/
// isOptimized is set on DISubprogram (the textual format is hardcoded
// in LLVM 11). We work around by setting spFlags to 0 — this loses
// the isDefinition flag (semantic side-effect), but the cosim TB generator only
// cares about the metadata STRUCTURE (DILocalVariable per arg), not
// the spFlags themselves.
static void downgradeDebugInfoLLVM11to7(Module &M) {
  unsigned StrippedSPFlags = 0, StrippedRetained = 0;
  unsigned StrippedChecksum = 0, StrippedNameTable = 0;

  // DISubprogram: clear spFlags + retainedNodes via direct mutation
  // (DI nodes are uniqued; we replace operand pointers in-place).
  for (Function &F : M) {
    DISubprogram *SP = F.getSubprogram();
    if (!SP) continue;
    bool Changed = false;
    // Clear spFlags-derived bool flags by reconstructing with explicit
    // false values. DISubprogram::cloneWithReplacedNode doesn't exist
    // pre-LLVM 13, so we directly replace via TempDISubprogram.
    if (SP->getSPFlags() != 0) {
      // LLVM 11's DISubprogram has setIsDistinct etc. but no setSPFlags.
      // Workaround: rebuild the node with spFlags=0.
      auto *NewSP = DISubprogram::getDistinct(
          M.getContext(),
          SP->getScope(), SP->getName(), SP->getLinkageName(),
          SP->getFile(), SP->getLine(), SP->getType(), SP->getScopeLine(),
          SP->getContainingType(), SP->getVirtualIndex(),
          SP->getThisAdjustment(), SP->getFlags(),
          /*SPFlags=*/(DISubprogram::DISPFlags)0,
          SP->getUnit(), SP->getTemplateParams(),
          SP->getDeclaration(),
          /*RetainedNodes=*/nullptr,
          SP->getThrownTypes());
      F.setSubprogram(NewSP);
      StrippedSPFlags++;
      Changed = true;
    } else if (SP->getRetainedNodes() && SP->getRetainedNodes()->getNumOperands() > 0) {
      // retainedNodes-only case (no spFlags): clear retainedNodes by rebuild
      auto *NewSP = DISubprogram::getDistinct(
          M.getContext(),
          SP->getScope(), SP->getName(), SP->getLinkageName(),
          SP->getFile(), SP->getLine(), SP->getType(), SP->getScopeLine(),
          SP->getContainingType(), SP->getVirtualIndex(),
          SP->getThisAdjustment(), SP->getFlags(),
          SP->getSPFlags(),
          SP->getUnit(), SP->getTemplateParams(),
          SP->getDeclaration(),
          /*RetainedNodes=*/nullptr,
          SP->getThrownTypes());
      F.setSubprogram(NewSP);
      StrippedRetained++;
    }
    (void)Changed;
  }

  // DIFile: strip checksumkind + checksum (LLVM 11 only). DICompileUnit:
  // strip nameTableKind. We can't easily mutate uniqued nodes; instead
  // iterate over Named metadata + traverse.
  // Note: this catch-all uses module's named "llvm.dbg.cu" + walks each CU.
  if (NamedMDNode *CUs = M.getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0; i < CUs->getNumOperands(); ++i) {
      auto *CU = dyn_cast<DICompileUnit>(CUs->getOperand(i));
      if (!CU) continue;
      // nameTableKind: only DICompileUnit. Default == Default == 0 → no emit.
      if (CU->getNameTableKind() != DICompileUnit::DebugNameTableKind::Default) {
        CU->replaceOperandWith(
            /*idx unknown — LLVM 11 doesn't expose nameTableKind setter*/ 0,
            CU->getOperand(0).get());  // no-op safety
        // Direct API: setNameTableKind doesn't exist in LLVM 11.
        // Workaround: skip — this field defaults to "Default" rarely.
        StrippedNameTable++;
      }
    }
  }
  // checksumkind / checksum on DIFile: similar limitation. Most rustc-emitted
  // DIFiles have these set. We accept the LLVM 11 syntax leak for now and
  // strip them via a future improvement (full rebuild via DIFile::get).

  if (StrippedSPFlags || StrippedRetained)
    hlsrs::vxx::vxxDbg()
        << "VXXPrep: downgraded DI — cleared spFlags on " << StrippedSPFlags
        << " DISubprogram, retainedNodes on " << StrippedRetained << "\n";
}

extern "C" void LLVMRustVitisStripIncompatibleAttrs(LLVMModuleRef M) {
  Module *Mod = unwrap(M);

  // Full DI strip — partial-prune (downgradeDebugInfoLLVM11to7) was
  // leaving dangling parent links in the DI scope graph (rebuilt
  // DISubprogram orphaned the OLD one which was still referenced by
  // CU + lexical blocks). LLVM 7's ADCE pass walks DI scopes and
  // crashed on null parents when encountering certain instruction
  // patterns (e.g. injected `_ssdm_op_ReadReq.m_axi.p1i32` calls).
  // Full strip eliminates the broken graph entirely.
  //
  // Tradeoff: HLS uses DILocalVariable names to name BRAM/URAM ports
  // (e.g. uram_ecc → `ap_ecc_res1_U`). Without DI, ports inherit the
  // SSA name + inline suffix (`res1_i7`) instead of the debug names.
  // The LLVM-11 debug metadata is fully stripped here (LLVM-7 can't parse its
  // DI syntax), so ports fall back to the SSA + inline-suffix names.
  llvm::StripDebugInfo(*Mod);

  static const Attribute::AttrKind StripKinds[] = {
      Attribute::NoFree,
      Attribute::NoRecurse,
      Attribute::NoSync,
      Attribute::NoUndef,
      Attribute::WillReturn,
      Attribute::ImmArg,
      // NoRedZone is x86_64 calling-convention specific (disables 128-byte
      // stack red zone). On Rust IR rustc emits it by default. Vitis HLS
      // appears to interpret it as needing an FSM clock for top kernels
      // even when the body is pure combinational — emitting `ap_clk`
      // for a combinational kernel. Strip to keep such kernels clockless.
      Attribute::NoRedZone,
  };

  auto stripFromFunction = [](Function &F) {
    for (auto K : StripKinds) {
      if (F.hasFnAttribute(K))
        F.removeFnAttr(K);
      if (F.hasAttribute(AttributeList::ReturnIndex, K))
        F.removeAttribute(AttributeList::ReturnIndex, K);
      for (unsigned i = 0; i < F.arg_size(); ++i) {
        if (F.hasParamAttribute(i, K))
          F.removeParamAttr(i, K);
      }
    }
  };

  auto stripFromCallBase = [](CallBase &CB) {
    for (auto K : StripKinds) {
      CB.removeAttribute(AttributeList::FunctionIndex, K);
      CB.removeAttribute(AttributeList::ReturnIndex, K);
      for (unsigned i = 0; i < CB.arg_size(); ++i) {
        CB.removeParamAttr(i, K);
      }
    }
  };

  for (Function &F : *Mod) {
    stripFromFunction(F);
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I))
          stripFromCallBase(*CB);
      }
    }
  }

  // Rename anonymous parameter / BB / instruction names so every SSA value
  // starts with a letter. Works around an LLVM 7 parser bug that rejects
  // `dereferenceable(N) %<numeric>` in parameter lists.
  for (Function &F : *Mod) {
    if (F.isDeclaration())
      continue;
    unsigned argIdx = 0;
    for (Argument &Arg : F.args()) {
      if (!Arg.hasName()) {
        Arg.setName("a" + std::to_string(argIdx));
      }
      ++argIdx;
    }
    unsigned bbIdx = 0;
    for (BasicBlock &BB : F) {
      if (!BB.hasName())
        BB.setName("b" + std::to_string(bbIdx));
      ++bbIdx;
      unsigned valIdx = 0;
      for (Instruction &I : BB) {
        if (!I.hasName() && !I.getType()->isVoidTy())
          I.setName("v" + std::to_string(valIdx));
        ++valIdx;
      }
    }
  }
}
