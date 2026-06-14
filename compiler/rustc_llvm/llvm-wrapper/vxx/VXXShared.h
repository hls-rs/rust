// VXXShared.h — low-level helpers shared across the prep stages.
//
// Carved out of the monolithic VXXPrep.cpp so the IR-normalization stage
// (vxx_prep.cpp) and the Vitis lowering modules can both call them. Because
// they cross translation units they cannot be `static`/anonymous-namespace;
// they live in the `hlsrs::vxx` namespace.

#ifndef VXXSHARED_H
#define VXXSHARED_H

#include <string>
#include <vector>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace hlsrs { namespace vxx {

// Debug/trace sink. Returns a discarding stream — the prep stages keep their
// progress traces in the source but stay quiet.
llvm::raw_ostream &vxxDbg();

// True when a build-mode marker (e.g. `__vxx_cpp_proxy`, `__vxx_axis_packed`,
// `__vxx_aphs_stream`) is present as an EXTERNAL DECLARATION in this module — the
// case for a kernel crate that emitted a call to the marker (whose body lives in
// barista_hls). The marker DEFINITIONS live in barista_hls, so its own
// translation unit sees a definition, not a declaration, and this returns false
// there — stopping the marker paths from mis-firing on the library IR.
bool markerUsed(llvm::Module &M, llvm::StringRef Name);

// Mangle a type for a Vitis FPGA intrinsic name.
std::string mangleForIntrinsic(llvm::Type *T);

// Finalize a function-clone rewrite: erase `Old` and rename `New` to the name
// `Old` had. The standard tail of every "clone F with a new signature, then
// swap it in" pass.
void replaceFunctionKeepingName(llvm::Function *Old, llvm::Function *New);

// Get or insert `llvm.fpga.fifo.{pop,push}` for arbitrary element types.
llvm::Function *getOrInsertFifoPopAny(llvm::Module &M, llvm::Type *ElemTy);
llvm::Function *getOrInsertFifoPushAny(llvm::Module &M, llvm::Type *ElemTy);

// Recursively remap a type by substituting any occurrences of types in TypeMap.
llvm::Type *remapTypeRecursive(llvm::Type *T,
                               const llvm::DenseMap<llvm::Type *, llvm::Type *> &TypeMap,
                               llvm::LLVMContext &Ctx);

// Walk a function's body and substitute OldT -> NewT (and wrappers thereof).
void substituteTypeInFunction(
    llvm::Function *F,
    const llvm::DenseMap<llvm::Type *, llvm::Type *> &TypeMap,
    llvm::ArrayRef<std::pair<llvm::Value *, llvm::Value *>> SeedVMap = {},
    const llvm::DenseMap<llvm::StructType *, std::vector<int>> *FieldMaps = nullptr);

// Rebuild a constant so it references remapped types.
llvm::Constant *remapConstantType(
    llvm::Constant *C,
    const llvm::DenseMap<llvm::Type *, llvm::Type *> &TypeMap,
    llvm::LLVMContext &Ctx,
    const llvm::DenseMap<llvm::StructType *, std::vector<int>> *FieldMaps = nullptr);

// LLVM-rule-compliant whole-module struct replacement.
void remapStructsInModule(
    llvm::Module &M,
    const llvm::DenseMap<llvm::Type *, llvm::Type *> &TypeMap,
    const llvm::DenseMap<llvm::StructType *, std::vector<int>> *FieldMaps);

} } // namespace hlsrs::vxx

#endif // VXXSHARED_H
