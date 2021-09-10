#!/usr/bin/env bash
source ${BASH_SOURCE%/*}/common.sh

exec_check lean --plugin ${TYR_PLUGIN} -j 0 "$f"
