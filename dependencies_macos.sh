#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.1}"
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "${OS}" != "Darwin" ]]; then
  echo "dependencies_macos.sh supports macOS only (detected: ${OS})" >&2
  exit 1
fi

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
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

echo "Downloading ${LIBTORCH_URL}"
curl --fail --location --retry 5 --retry-all-errors --show-error -o "${tmp_dir}/libtorch.zip" "${LIBTORCH_URL}"
unzip -tq "${tmp_dir}/libtorch.zip"
unzip -q "${tmp_dir}/libtorch.zip" -d "${tmp_dir}"
if [[ ! -d "${tmp_dir}/libtorch" ]]; then
  echo "Downloaded archive did not contain expected libtorch directory" >&2
  exit 1
fi
rm -rf libtorch
mv "${tmp_dir}/libtorch" ./libtorch

echo "Installed LibTorch ${LIBTORCH_VERSION} to ${ROOT_DIR}/external/libtorch"
