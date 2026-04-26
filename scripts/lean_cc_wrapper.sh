#!/usr/bin/env bash
set -euo pipefail

sysroot="${LEAN_SYSROOT:-}"
if [[ -z "$sysroot" ]]; then
  if command -v lean >/dev/null 2>&1; then
    sysroot="$(lean --print-prefix)"
  fi
fi

extra=()
is_compile=0
is_macos=0
if [[ "${OSTYPE:-}" == darwin* ]]; then
  is_macos=1
fi
for arg in "$@"; do
  if [[ "$arg" == "-c" ]]; then
    is_compile=1
    break
  fi
done

if [[ "${LEAN_CC_FAST:-0}" == "1" && "$is_compile" -eq 1 ]]; then
  # Speed up local iteration on giant Lean-generated C files.
  # Lean passes -O3 by default; later flags win, so add -O0 here.
  extra+=("-O0")
fi

# Pick the gcc to invoke. Default behavior:
#   1. LEAN_CC_GCC env (explicit override)
#   2. `gcc` from PATH (so HPC modules / nix shells / etc. just work)
#   3. /usr/bin/gcc fallback
# Critical for HPC: the system gcc is often older than the gcc that
# compiled the project's C++ sources. Using a mismatched gcc to *link*
# means the linker's hardcoded SPEC paths point at the system's older
# libstdc++, which won't have C++11-ABI symbols (e.g. `std::__cxx11::*`)
# that the .o files reference. PATH-discovered `gcc` typically matches
# the loaded toolchain module on cluster setups.
gcc_bin="${LEAN_CC_GCC:-}"
if [[ -z "$gcc_bin" ]]; then
  gcc_bin="$(command -v gcc 2>/dev/null || true)"
fi
if [[ -z "$gcc_bin" || ! -x "$gcc_bin" ]]; then
  gcc_bin="/usr/bin/gcc"
fi

mapped=()
need_uv=0
# When LEAN_CC_LIBTORCH_DIR is set (e.g. to a local /tmp cache), substitute
# -L<original_libtorch_dir> with -L<cache_dir> at link time. This is huge on
# clusters where the libtorch tree lives on networked storage that's slow for
# the linker's random-access mmap pattern. Only the link-time path is
# rewritten; -Wl,-rpath flags are left alone, so the resulting binary still
# loads dylibs from their original location at runtime.
libtorch_cache="${LEAN_CC_LIBTORCH_DIR:-}"

# Detect if we're linking a binary that doesn't actually use libtorch / arrow
# / parquet / soxr / libTyrC.a (e.g. pure-Lean codegen tools). The binary
# name is in `-o <path>`. If it matches LEAN_CC_STRIP_HEAVY_LINK_PATTERN,
# strip the heavy flags before invoking gcc. This avoids a multi-minute
# link against ~1.7 GB of dylibs for a binary that only needs Lean stdlib.
strip_heavy_pattern="${LEAN_CC_STRIP_HEAVY_LINK_PATTERN:-}"
strip_heavy=0
if [[ -n "$strip_heavy_pattern" ]]; then
  prev=""
  for arg in "$@"; do
    if [[ "$prev" == "-o" && "$arg" == *${strip_heavy_pattern}* ]]; then
      strip_heavy=1
      break
    fi
    prev="$arg"
  done
fi
prev=""
for arg in "$@"; do
  case "$arg" in
    -lc++)
      if [[ -n "$sysroot" && -f "$sysroot/lib/libc++.a" ]]; then
        mapped+=("$sysroot/lib/libc++.a")
      else
        mapped+=("$arg")
      fi
      ;;
    -lc++abi)
      if [[ -n "$sysroot" && -f "$sysroot/lib/libc++abi.a" ]]; then
        mapped+=("$sysroot/lib/libc++abi.a")
      else
        mapped+=("$arg")
      fi
      ;;
    -lgmp)
      if [[ -n "$sysroot" && -f "$sysroot/lib/libgmp.a" ]]; then
        mapped+=("$sysroot/lib/libgmp.a")
      else
        mapped+=("$arg")
      fi
      ;;
    -luv)
      need_uv=1
      ;;
    -L*)
      cand="${arg#-L}"
      # Only substitute when the path part is non-empty (Lake sometimes
      # emits `-L` and the path as two separate args; the next iteration
      # carries the path as a positional arg).
      if [[ -z "$cand" ]]; then
        mapped+=("$arg")
      elif [[ -n "$libtorch_cache" && -f "${cand}/libtorch.so" ]]; then
        # Looks like the libtorch dir; redirect to the fast local cache.
        mapped+=("-L${libtorch_cache}")
      elif [[ -n "${LEAN_CC_LUSTRE_CACHE:-}" && -d "${LEAN_CC_LUSTRE_CACHE}${cand}" ]]; then
        # Generic Lustre-cache substitution: e.g.
        # `-L/grid/it/.../Arrow/.../lib` → `-L/tmp/tyr_lustre_cache/grid/it/.../Arrow/.../lib`
        # so the linker reads from local SSD instead of the slow networked FS.
        mapped+=("-L${LEAN_CC_LUSTRE_CACHE}${cand}")
      else
        mapped+=("$arg")
      fi
      ;;
    -ltorch|-ltorch_cpu|-ltorch_cuda|-ltorch_cuda_linalg|-lc10|-lc10_cuda|-lcudart|-larrow|-lparquet|-lsoxr)
      if [[ "$strip_heavy" -eq 1 ]]; then
        # Skip — not needed for this binary.
        :
      else
        mapped+=("$arg")
      fi
      ;;
    *.a)
      # Strip libTyrC.a (positional path arg) for stripped binaries.
      if [[ "$strip_heavy" -eq 1 && "$arg" == *libTyrC.a ]]; then
        :
      else
        mapped+=("$arg")
      fi
      ;;
    *)
      mapped+=("$arg")
      ;;
  esac
  prev="$arg"
done

if [[ "$need_uv" -eq 1 ]]; then
  if [[ -n "$sysroot" && -f "$sysroot/lib/libuv.a" ]]; then
    if [[ "$is_macos" -eq 1 ]]; then
      mapped+=("-Wl,-force_load,$sysroot/lib/libuv.a")
    else
      mapped+=("-Wl,--whole-archive" "$sysroot/lib/libuv.a" "-Wl,--no-whole-archive")
    fi
  else
    mapped+=("-luv")
  fi
fi

declare -a cuda_link_args=()
if [[ "$is_compile" -eq 0 && "$strip_heavy" -eq 0 ]]; then
  has_ltorch=0
  has_ltorch_cuda=0
  torch_lib_dir=""

  for arg in "${mapped[@]}"; do
    case "$arg" in
      -ltorch)
        has_ltorch=1
        ;;
      -ltorch_cuda)
        has_ltorch_cuda=1
        ;;
      -L*)
        cand_dir="${arg#-L}"
        if [[ -f "${cand_dir}/libtorch_cuda.so" ]]; then
          torch_lib_dir="${cand_dir}"
        fi
        ;;
    esac
  done

  if [[ "$has_ltorch" -eq 1 && "$has_ltorch_cuda" -eq 0 ]]; then
    if [[ -z "$torch_lib_dir" && -f "external/libtorch/lib/libtorch_cuda.so" ]]; then
      torch_lib_dir="external/libtorch/lib"
    fi
    if [[ -n "$torch_lib_dir" ]]; then
      cuda_link_args+=("-ltorch_cuda")
      if [[ -f "${torch_lib_dir}/libtorch_cuda_linalg.so" ]]; then
        cuda_link_args+=("-ltorch_cuda_linalg")
      fi
      if [[ -f "${torch_lib_dir}/libc10_cuda.so" ]]; then
        cuda_link_args+=("-lc10_cuda")
      fi
      cudart_dir=""
      if compgen -G "${torch_lib_dir}/libcudart*.so*" >/dev/null 2>&1; then
        cudart_dir="${torch_lib_dir}"
      else
        cand_dirs=()
        while IFS= read -r cand; do
          cand_dirs+=("$cand")
        done < <(compgen -G "${torch_lib_dir}/../../nvidia/cu*/lib" || true)
        cand_dirs+=("${torch_lib_dir}/../../nvidia/cuda_runtime/lib")
        for cand in "${cand_dirs[@]}"; do
          if [[ -d "$cand" ]] && compgen -G "${cand}/libcudart*.so*" >/dev/null 2>&1; then
            cudart_dir="$cand"
            break
          fi
        done
      fi
      if [[ -n "$cudart_dir" ]]; then
        if [[ "$cudart_dir" != "$torch_lib_dir" ]]; then
          cuda_link_args+=("-L${cudart_dir}" "-Wl,-rpath,${cudart_dir}")
        fi
        cuda_link_args+=("-lcudart")
      fi
    fi
  fi
fi

cmd=("$gcc_bin" "${mapped[@]}")
if ((${#cuda_link_args[@]} > 0)); then
  cmd+=("${cuda_link_args[@]}")
fi
if ((${#extra[@]} > 0)); then
  cmd+=("${extra[@]}")
fi

# Link-time rpath for the cached lib paths. Without this, when ld pulls in
# a .so from one of the substituted `-L${cache}` paths it'd resolve its
# DT_NEEDED dylibs via the original `-Wl,-rpath` paths OR default search —
# both of which still point at the slow networked filesystem. -rpath-link
# only affects link-time resolution; the binary's runtime rpath is
# unaffected. Walk the substituted -L paths and add corresponding -rpath-link.
if [[ "$is_compile" -eq 0 && -n "$libtorch_cache" ]]; then
  cmd+=("-Wl,-rpath-link=${libtorch_cache}")
fi
if [[ "$is_compile" -eq 0 && -n "${LEAN_CC_LUSTRE_CACHE:-}" ]]; then
  # Add -rpath-link for every cached directory that holds at least one .so.
  # gcc's specs file injects implicit -L paths from the loaded toolchain
  # module that point at the slow networked filesystem; without rpath-link
  # entries pointing at the cache, ld resolves DT_NEEDED dylibs (e.g.
  # libabsl_*.so transitively pulled in by libarrow.so) via those
  # implicit -L paths -> still slow.
  while IFS= read -r d; do
    cmd+=("-Wl,-rpath-link=${d}")
  done < <(find "${LEAN_CC_LUSTRE_CACHE}" -name '*.so*' -type f 2>/dev/null \
            | xargs -r dirname 2>/dev/null | sort -u)
fi

# Pick a linker for link steps. Decision:
#  - For shared-library builds (-shared in args), use gold by default if
#    available. Per-module Lean precompileModules .so files dominate the
#    build and don't include libTyrC.a, so they're free of the
#    hidden-symbol C++ visibility issue gold trips on.
#  - For executable builds (no -shared), use bfd. Tyr's lean_exe link line
#    includes libTyrC.a, which has tyr.o compiled with -fvisibility=hidden
#    referencing C++14 sized operator delete (_ZdlPvm). Gold rejects this
#    as a hidden-symbol-not-defined-locally error; bfd silently lets it
#    resolve at runtime via libstdc++.so.6.
# Override either default with LEAN_CC_LINKER (set explicitly to gold/mold/
# lld/bfd, or empty for system default).
if [[ "$is_compile" -eq 0 && "$is_macos" -eq 0 ]]; then
  is_shared=0
  for arg in "$@"; do
    if [[ "$arg" == "-shared" ]]; then
      is_shared=1
      break
    fi
  done
  if [[ -n "${LEAN_CC_LINKER+x}" ]]; then
    linker_choice="${LEAN_CC_LINKER}"
  elif [[ "$is_shared" -eq 1 ]]; then
    linker_choice="gold"
  else
    # Executables: default to lld via the elan toolchain. bfd is correct but
    # 100x slower than lld on big LTO links on networked filesystems (the
    # GenerateGpuKernels exe pulls in libtorch/arrow + LTO bitcode from every
    # Tyr.* module). lld handles the libstdc++ hidden-symbol case correctly,
    # unlike gold which rejects `_ZdlPvm` from tyr.o.
    linker_choice="lld"
  fi
  if [[ -n "$linker_choice" ]]; then
    # gcc's `-fuse-ld=<name>` form requires `ld.<name>` to live in PATH or in
    # a `-B<dir>`-supplied search path. Probe candidate locations for a
    # working binary (the elan toolchain ships an `ld.lld` whose libLLVM.so
    # needs GLIBC newer than the host, so a naive PATH-based selection is not
    # enough; verify with `--version` before picking).
    chosen_dir=""
    candidates=""
    if [[ -n "$sysroot" ]]; then
      candidates+="$sysroot/bin/ld.${linker_choice}"$'\n'
    fi
    candidates+="/grid/zador/home/pehle/.elan/toolchains/leanprover--lean4---v4.12.0/bin/ld.${linker_choice}"$'\n'
    cmd_lld_path="$(command -v "ld.${linker_choice}" 2>/dev/null || true)"
    if [[ -n "$cmd_lld_path" ]]; then
      candidates+="$cmd_lld_path"$'\n'
    fi
    cmd_alt_path="$(command -v "${linker_choice}" 2>/dev/null || true)"
    if [[ -n "$cmd_alt_path" ]]; then
      candidates+="$cmd_alt_path"$'\n'
    fi
    while IFS= read -r c; do
      [[ -z "$c" ]] && continue
      [[ ! -x "$c" ]] && continue
      if "$c" --version >/dev/null 2>&1; then
        chosen_dir="$(dirname "$c")"
        break
      fi
    done <<< "$candidates"
    if [[ -n "$chosen_dir" ]]; then
      cmd+=("-B${chosen_dir}" "-fuse-ld=${linker_choice}")
      # Ensure 64-bit ELF target — lld 15 (the host-glibc-compatible build we
      # ship) defaults to elf32-i386 if not told otherwise, and Lake's link
      # command line does not include `-m elf_x86_64` explicitly.
      if [[ "${linker_choice}" == "lld" ]]; then
        cmd+=("-Wl,-m,elf_x86_64")
      fi
    fi
  fi
fi

exec "${cmd[@]}"
