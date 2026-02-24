#!/usr/bin/env bash
set -euo pipefail

LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.7.1}"

ARCH="$(uname -m)"
if [[ "${ARCH}" == "arm64" ]]; then
  LIBTORCH_PKG="libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
else
  LIBTORCH_PKG="libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip"
fi

mkdir -p external
cd external
curl --fail --location --retry 5 --retry-all-errors --show-error -O \
  "https://download.pytorch.org/libtorch/cpu/${LIBTORCH_PKG}"
unzip -q "${LIBTORCH_PKG}"
rm -f "${LIBTORCH_PKG}"
