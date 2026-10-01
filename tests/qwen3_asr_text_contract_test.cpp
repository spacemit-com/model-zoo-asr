/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include "backends/qwen3_asr/qwen3_asr_text.hpp"

namespace {

void requireEqual(
        const std::string& actual,
        const std::string& expected,
        const std::string& name) {
    if (actual == expected) {
        return;
    }
    std::cerr << "ASSERTION FAILED: " << name << "\n"
        << "  expected: [" << expected << "]\n"
        << "  actual:   [" << actual << "]\n";
    std::exit(1);
}

}  // namespace

int main() {
    using asr::qwen3::normalizeTranscript;

    requireEqual(
        normalizeTranscript("language Chinese<asr_text>\xE5\xB0\x8F\xE9\x9D\x99\xEF\xBC\x8C\xE5\xB0\x8F\xE9\x9D\x99\xE3\x80\x82"),
        "\xE5\xB0\x8F\xE9\x9D\x99\xEF\xBC\x8C\xE5\xB0\x8F\xE9\x9D\x99\xE3\x80\x82",
        "language tag with Chinese text");
    requireEqual(
        normalizeTranscript("language None<asr_text>"), "", "no speech");
    requireEqual(
        normalizeTranscript("language English<asr_text>turn on the light<|im_end|>"),
        "turn on the light",
        "end-of-turn token");
    requireEqual(
        normalizeTranscript(
            "  <|im_start|> language Chinese<asr_text>\xE6\x89\x93\xE5\xBC\x80\xE7\xA9\xBA\xE8\xB0\x83 <|im_end|>  "),
        "\xE6\x89\x93\xE5\xBC\x80\xE7\xA9\xBA\xE8\xB0\x83",
        "edge control tokens");
    requireEqual(
        normalizeTranscript("language Chinese<asr_text>\xE2\x80\x9C\xE6\x89\x93\xE5\xBC\x80\xE7\x81\xAF\xE2\x80\x9D"),
        "\xE2\x80\x9C\xE6\x89\x93\xE5\xBC\x80\xE7\x81\xAF\xE2\x80\x9D",
        "quotes in the transcript are kept");
    requireEqual(
        normalizeTranscript("plain transcript"),
        "plain transcript",
        "text without the tag is unchanged");

    std::cout << "PASS --qwen3-asr-text-contract" << std::endl;
    return 0;
}
