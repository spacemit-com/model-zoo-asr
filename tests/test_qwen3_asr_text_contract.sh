#!/usr/bin/env bash
set -euo pipefail

module_dir="components/model_zoo/asr"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-qwen3-text-test.XXXXXX")"

cleanup() {
  rm -rf "${build_dir}"
}
trap cleanup EXIT

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  "${module_dir}/tests/qwen3_asr_text_contract_test.cpp" \
  "${module_dir}/src/backends/qwen3_asr/qwen3_asr_text.cpp" \
  -I"${module_dir}/src" \
  -o "${build_dir}/qwen3_asr_text_contract_test"

"${build_dir}/qwen3_asr_text_contract_test"
