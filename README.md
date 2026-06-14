# hls-rs/rust — HLS-extended Rust toolchain

Rust compiler extended with support for HLS (High-Level Synthesis) toolchains.

Each branch targets a specific Rust version × HLS toolchain combination,
named `rust-<rust-version>-<vendor>-<vendor-version>`. The bundled LLVM under
`src/llvm-project` is patched with HLS backend support corresponding to the
target branch — see [hls-rs/llvm-project](https://github.com/hls-rs/llvm-project)
on the matching branch.

Original Rust README: see [README.origin.md](README.origin.md).

## License
Dual MIT / Apache 2.0, same as upstream Rust.
