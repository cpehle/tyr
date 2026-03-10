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

gcc_bin="${LEAN_CC_GCC:-/grid/it/data/elzar/easybuild/software/GCCcore/12.3.0/bin/gcc}"
if [[ ! -x "$gcc_bin" ]]; then
  gcc_bin="/usr/bin/gcc"
fi

mapped=()
need_uv=0
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
    *)
      mapped+=("$arg")
      ;;
  esac
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
if [[ "$is_compile" -eq 0 ]]; then
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
      if compgen -G "${torch_lib_dir}/libcudart*.so*" >/dev/null 2>&1; then
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

exec "${cmd[@]}"
