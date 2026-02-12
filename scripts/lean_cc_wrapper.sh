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
    mapped+=("-Wl,--whole-archive" "$sysroot/lib/libuv.a" "-Wl,--no-whole-archive")
  else
    mapped+=("-luv")
  fi
fi

exec "$gcc_bin" "${mapped[@]}" "${extra[@]}"
