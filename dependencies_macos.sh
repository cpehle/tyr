#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.1}"
ARCH="$(uname -m)"

case "${ARCH}" in
  arm64|aarch64)
    LIBTORCH_ARCHIVE="libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
    ;;
  x86_64)
    LIBTORCH_ARCHIVE="libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip"
    ;;
  *)
    echo "Unsupported macOS architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/${LIBTORCH_ARCHIVE}"

mkdir -p "${ROOT_DIR}/external"
cd "${ROOT_DIR}/external"
rm -rf libtorch

echo "Downloading ${LIBTORCH_URL}"
curl --fail --location --retry 5 --retry-all-errors --show-error -o libtorch.zip "${LIBTORCH_URL}"
unzip -tq libtorch.zip
unzip -q libtorch.zip
rm -f libtorch.zip

echo "Installed LibTorch ${LIBTORCH_VERSION} to ${ROOT_DIR}/external/libtorch"
