#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

print_nonempty_lines() {
  if [[ -n "${1:-}" ]]; then
    printf '%s\n' "$1"
  fi
}

lake_exes="$(
  if command -v rg >/dev/null 2>&1; then
    rg '^\s*lean_exe\s+([A-Za-z0-9_]+)\s+where' -or '$1' lakefile.lean
  else
    awk '
      match($0, /^[[:space:]]*lean_exe[[:space:]]+([A-Za-z0-9_]+)[[:space:]]+where/, m) { print m[1] }
    ' lakefile.lean
  fi | sort -u
)"

bazel_bins="$(
  sed -n '/lean_binary(/,/)/p' BUILD.bazel |
    if command -v rg >/dev/null 2>&1; then
      rg 'name = "([^"]+)"' -or '$1'
    else
      awk '
        match($0, /name = "([^"]+)"/, m) { print m[1] }
      '
    fi |
    sort -u
)"

missing_in_bazel="$(
  comm -23 <(print_nonempty_lines "$lake_exes") <(print_nonempty_lines "$bazel_bins")
)"
missing_in_lake="$(
  comm -13 <(print_nonempty_lines "$lake_exes") <(print_nonempty_lines "$bazel_bins")
)"

has_drift=0
if [[ -n "$missing_in_bazel" ]]; then
  echo "Missing Bazel lean_binary targets for these Lake executables:"
  printf '%s\n' "$missing_in_bazel"
  has_drift=1
fi

if [[ -n "$missing_in_lake" ]]; then
  if [[ "$has_drift" -eq 1 ]]; then
    echo
  fi
  echo "Missing Lake executables for these Bazel lean_binary targets:"
  printf '%s\n' "$missing_in_lake"
  has_drift=1
fi

if [[ "$has_drift" -eq 1 ]]; then
  exit 1
fi

echo "Lake/Bazel executable target parity check passed."
