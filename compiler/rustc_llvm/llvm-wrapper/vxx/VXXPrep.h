// VXXPrep — the Vitis-HLS preprocessing passes, compiled statically into rustc.
//
// Every pass here exists to turn rustc-emitted Rust IR into LLVM IR that the
// Vitis HLS back-end accepts and synthesizes: BitInt narrowing, loop
// canonicalization, static/struct layout, and the Xilinx pragma/interface
// lowering (_ssdm_op_*, xlx_* operand bundles, SpecInterface, AXIS/m_axi/DSP).
// None of it is vendor-neutral today — it all targets Vitis. If a second
// back-end (e.g. Altera) is ever added, whatever genuinely generalizes would
// move up to the empty `hlsrs` umbrella namespace; for now everything lives in
// `hlsrs::vxx`.
//
// Naming: VXX = transliteration of `v++` (the AMD Vitis CLI driver).
// Same convention as Fortran HLS's VXXIRDowngrader.

#ifndef VXXPREP_H
#define VXXPREP_H

#include <llvm-c/Core.h>

#ifdef __cplusplus
extern "C" {
#endif

// === Public C ABI (called directly by rustc codegen: back/write.rs) ===

// EarlyPrep: shape rustc -O0 IR before LLVM mid-end optimization, so the
// standard opts (inlining, GVN, IndVarSimplify) can simplify aggressively.
void LLVMRustVitisEarlyPrep(LLVMModuleRef M);

// Full prep pipeline, run after LLVM optimization: IR normalization
// (vxxShapeIR) followed by Vitis pragma/interface lowering (vxxLowerVitis).
void LLVMRustVitisPrep(LLVMModuleRef M);

// LLVM 11 -> 7 downgrade: strip LLVM 11-only attributes and rename anonymous
// SSA values so the v++ LLVM 7 reader can parse the bitcode.
void LLVMRustVitisStripIncompatibleAttrs(LLVMModuleRef M);

// === Internal stage entries ===

// Pre-opt rustc -O0 cleanup: synthesize Range/slice trampolines, drop
// llvm.expect, strip panic chains. Backs VXXEarlyPrep. (vxx_prep.cpp)
void vxxEarlyShape(LLVMModuleRef M);

// Post-opt IR normalization: Rust idioms -> synthesizable fpga64 IR
// (loop canonicalization, BitInt narrowing, static globals, struct layout),
// with no Xilinx intrinsics yet. First half of VXXPrep. (vxx_prep.cpp)
void vxxShapeIR(LLVMModuleRef M);

// Vitis lowering: all HLS interface/pragma semantics + signature shaping
// (_ssdm_op_*, xlx_* operand bundles, SpecInterface, AXIS/m_axi/DSP/dataflow).
// Second half of VXXPrep. (VXXPrep.cpp)
void vxxLowerVitis(LLVMModuleRef M);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VXXPREP_H
