#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

lake_exes="$(
  rg '^\s*lean_exe\s+([A-Za-z0-9_]+)\s+where' -or '$1' lakefile.lean | sort -u
)"

bazel_bins="$(
  sed -n '/lean_binary(/,/)/p' BUILD.bazel |
    rg 'name = "([^"]+)"' -or '$1' |
    sort -u
)"

missing="$(comm -23 <(printf "%s\n" "$lake_exes") <(printf "%s\n" "$bazel_bins"))"

if [[ -n "$missing" ]]; then
  echo "Missing Bazel lean_binary targets for these Lake executables:"
  printf '%s\n' "$missing"
  exit 1
fi

echo "Lake/Bazel executable target parity check passed."
