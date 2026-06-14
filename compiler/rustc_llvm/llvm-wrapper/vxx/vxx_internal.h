// vxx_internal.h — cross-cutting helpers used by the Vitis lowering modules.
//
// These helpers are used across VXXPrep.cpp and the per-domain modules
// (maxi.cpp, axis.cpp, ...). Their definitions live in vxx_common.cpp.
// Because they cross translation units they cannot be `static`/anonymous-
// namespace; they live in the `hlsrs::vxx` namespace.

#ifndef VXX_INTERNAL_H
#define VXX_INTERNAL_H

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace hlsrs { namespace vxx {

// _ssdm_op_* get-or-insert + private string-constant primitives.
llvm::FunctionCallee getSsdmOp(llvm::Module &M, llvm::StringRef Name);
llvm::Constant *makeSsdmStr(llvm::Module &M, llvm::StringRef S);

// Generic marker-erasure helpers.
bool dropMarkerDefinition(llvm::Module &M, llvm::StringRef Name);
bool dropMarkerCallsAndDefinition(llvm::Module &M, llvm::StringRef Name);
bool eraseUnimplementedMarkers(llvm::Module &M);

// AXIS / FIFO declaration primitives.
bool isAxisDisabledType(llvm::Type *T);
llvm::Function *getOrInsertAxisPopOrPush(llvm::Module &M, const char *Op,
                                         llvm::ArrayRef<llvm::Type *> FieldPtrTys);

// Value-tracing / attribute / string-slice helpers.
llvm::Value *traceToAllocaOrGlobal(llvm::Value *V);
bool appendXilinxAttribute(llvm::Module &M, llvm::StringRef AttrName,
                           llvm::StringRef AttrVal);
bool injectFnAttr(llvm::Module &M, llvm::StringRef Marker, llvm::StringRef AttrKey,
                  llvm::StringRef AttrVal);
llvm::Function *singleUserKernel(llvm::Module &M, llvm::StringRef MarkerName);
std::string extractByteSlice(llvm::Value *Ptr, llvm::Value *LenV);
std::string vxxParseRustStrSlice(llvm::Value *PtrArg, llvm::Value *LenArg);

} } // namespace hlsrs::vxx

#endif // VXX_INTERNAL_H
