// Xilinx HLS FPGA backend: IR-only target.
//
// Keep argument/return layouts as-is — no integer width extension, no forced
// indirect passing. The downstream HLS toolchain performs its own ABI
// flattening and narrow-integer handling, and we want rustc's IR to preserve
// the natural widths the user wrote (e.g. `u13` → `i13`).

use crate::abi::call::FnAbi;

pub fn compute_abi_info<Ty>(_fn_abi: &mut FnAbi<'_, Ty>) {}
