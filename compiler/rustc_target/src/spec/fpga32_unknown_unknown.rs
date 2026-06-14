// Xilinx HLS 32-bit FPGA target.
//
// IR-only target: no assembly, no linker is actually invoked. Intended to be
// driven with `--emit=llvm-ir` and then fed to Vitis HLS for synthesis.
//
// data_layout mirrors FPGATargetMachine::computeDataLayout for 32-bit mode.
// CodeModel::Large mirrors FPGATargetMachine (CodeModel::Large is hard-coded there).

use crate::spec::{CodeModel, LinkerFlavor, LldFlavor, PanicStrategy, RelocModel};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "fpga32-unknown-unknown".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-\
                      i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-\
                      n8:16:32:64-S128-\
                      v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
            .to_string(),
        arch: "fpga32".to_string(),

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
            linker: Some("rust-lld".to_owned()),
            executables: false,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            code_model: Some(CodeModel::Large),
            max_atomic_width: Some(0),
            atomic_cas: false,
            features: String::new(),
            dynamic_linking: false,
            disable_redzone: true,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            obj_is_bitcode: true,
            ..Default::default()
        },
    }
}
