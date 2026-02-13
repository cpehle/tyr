#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SIZES="${SIZES:-1 2 4}"
RUN_ARGS="${RUN_ARGS:---debug --iterations 2 --data data/nanochat --val data/nanochat}"
LOG_DIR="${LOG_DIR:-/tmp}"

mkdir -p "${LOG_DIR}"

declare -a rows=()
build_done=0

for nproc in ${SIZES}; do
  if ! [[ "${nproc}" =~ ^[0-9]+$ ]] || [[ "${nproc}" -lt 1 ]]; then
    echo "error: invalid process count '${nproc}' in SIZES='${SIZES}'" >&2
    exit 2
  fi

  log_file="${LOG_DIR}/tyr_nanochat_${nproc}gpu.log"
  echo "=== ${nproc} GPU run ==="
  echo "log: ${log_file}"

  if [[ "${build_done}" -eq 0 ]]; then
    SKIP_BUILD=0 NPROC_PER_NODE="${nproc}" ./scripts/nanochat/run_train_torchrun.sh ${RUN_ARGS} \
      2>&1 | tee "${log_file}"
    build_done=1
  else
    SKIP_BUILD=1 NPROC_PER_NODE="${nproc}" ./scripts/nanochat/run_train_torchrun.sh ${RUN_ARGS} \
      2>&1 | tee "${log_file}"
  fi

  if ! grep -q "Training complete!" "${log_file}"; then
    echo "error: run did not reach training completion for ${nproc} GPU(s)" >&2
    exit 1
  fi

  step40_toks="$(awk -F'tok/s=' '/Step 40:/{split($2,a," "); v=a[1]} END{print v}' "${log_file}")"
  total_tokens="$(awk -F'Total tokens: ' '/Training complete! Total tokens:/{v=$2} END{print v}' "${log_file}")"

  if [[ -z "${step40_toks}" || -z "${total_tokens}" ]]; then
    echo "error: failed to extract throughput/token metrics for ${nproc} GPU(s)" >&2
    exit 1
  fi

  rows+=("${nproc}|${step40_toks}|${total_tokens}|${log_file}")
done

base_toks="$(printf '%s\n' "${rows[@]}" | head -n1 | cut -d'|' -f2)"

echo
echo "Distributed NanoChat throughput summary"
printf "%-8s %-16s %-12s %-10s %s\n" "gpus" "step40_tok_per_s" "speedup" "tokens" "log"
for row in "${rows[@]}"; do
  gpus="$(cut -d'|' -f1 <<<"${row}")"
  toks="$(cut -d'|' -f2 <<<"${row}")"
  tokens="$(cut -d'|' -f3 <<<"${row}")"
  log_file="$(cut -d'|' -f4 <<<"${row}")"
  speedup="$(awk -v cur="${toks}" -v base="${base_toks}" 'BEGIN { if (base == 0) print "n/a"; else printf "%.3f", cur / base }')"
  printf "%-8s %-16s %-12s %-10s %s\n" "${gpus}" "${toks}" "${speedup}" "${tokens}" "${log_file}"
done
