#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
candidate_repo="${1:-$repo_root/../doc-gen4}"
fallback_url="${DOC_GEN4_DEFAULT_REPO_URL:-https://github.com/leanprover/doc-gen4.git}"
remote_name="${DOC_GEN4_REMOTE:-origin}"

normalize_github_url() {
  local url="$1"
  case "$url" in
    git@github.com:*)
      printf 'https://github.com/%s\n' "${url#git@github.com:}"
      ;;
    ssh://git@github.com/*)
      printf 'https://github.com/%s\n' "${url#ssh://git@github.com/}"
      ;;
    *)
      printf '%s\n' "$url"
      ;;
  esac
}

if [ -d "$candidate_repo/.git" ]; then
  if upstream_url="$(git -C "$candidate_repo" remote get-url "$remote_name" 2>/dev/null)"; then
    normalize_github_url "$upstream_url"
    exit 0
  fi
fi

printf '%s\n' "$fallback_url"
