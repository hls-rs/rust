#!/usr/bin/env bash
# Strict HLS rustc toolchain build + install.
#
# WHY THIS EXISTS
# ---------------
# `cargo +hls2` does NOT use build/x86_64-*/stage1/bin/rustc directly — the
# `hls2` rustup toolchain is a symlink to build/hls-install. So running only
# `x.py build --stage 1` updates stage1 but leaves the *installed* toolchain
# (hls-install) stale, and your VXXPrep / compiler changes silently do NOT take
# effect. This script closes that gap: it builds versioned dist tarballs and
# installs them from scratch, so the toolchain in use is EXACTLY what was built.
#
# WHAT IT INSTALLS (matches the working hls-install component set)
#   1. rustc        (the compiler — contains VXXPrep, statically linked)
#   2. cargo
#   3. rust-std     for x86_64-unknown-linux-gnu (host: csim / testbench)
#   4. rust-std     for fpga64-unknown-unknown   (kernel synthesis target)
#
# Conceptually "2 tarballs that matter": the host toolchain (rustc+cargo+host
# std) and the fpga64 target std. For a compiler-only edit only `rustc` actually
# changes, but this script always does a clean full install to stay foolproof.
#
# USAGE:  bash hls-rs/install-hls.sh
set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUST_DIR"

HOST=x86_64-unknown-linux-gnu
FPGA=fpga64-unknown-unknown
PREFIX="$RUST_DIR/build/hls-install"
DIST="$RUST_DIR/build/dist"

export RUSTUP_HOME="${RUSTUP_HOME:-$TMPDIR/rustup-hls}"
export CARGO_HOME="${CARGO_HOME:-$TMPDIR/cargo-home}"

echo "== [1/4] x.py dist (stage1) — builds all component tarballs =="
# This bootstrap (1.51) does not accept per-component dist paths, so build the
# full dist set. The install step below extracts ONLY the 4 components we need
# (rustc, cargo, rust-std host & fpga64), so clippy/miri/docs are built-but-
# -not-installed. `--exclude src/doc` skips the slow docs build.
python3 x.py dist --stage 1 --exclude src/doc

echo "== [2/4] clean install prefix: $PREFIX =="
rm -rf "$PREFIX"

echo "== [3/4] extract + install each component tarball =="
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
install_tarball() {
  local glob="$1"
  local tb
  tb="$(ls -1 "$DIST"/$glob 2>/dev/null | head -1)"
  if [ -z "$tb" ]; then echo "FATAL: no dist tarball matching $glob" >&2; exit 1; fi
  echo "   installing $(basename "$tb")"
  tar xf "$tb" -C "$STAGE"
  # The tarball's top dir is its basename without the .tar.xz suffix — derive it
  # directly (NOT via `tar tf | head`, which SIGPIPEs under `set -o pipefail`).
  local dir
  dir="$(basename "$tb" .tar.xz)"
  "$STAGE/$dir/install.sh" --prefix="$PREFIX" --disable-ldconfig >/dev/null
}
install_tarball "rustc-*-${HOST}.tar.xz"
install_tarball "cargo-*-${HOST}.tar.xz"
install_tarball "rust-std-*-${HOST}.tar.xz"
install_tarball "rust-std-*-${FPGA}.tar.xz"

echo "== [4/4] verify =="
RUSTC="$PREFIX/bin/rustc"
"$RUSTC" -Vv | sed 's/^/   /'
echo "   installed components:"; sed 's/^/     /' "$PREFIX/lib/rustlib/components"
echo "DONE: hls toolchain installed at $PREFIX (used by 'cargo +hls2')."
