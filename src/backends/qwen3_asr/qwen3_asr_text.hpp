/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QWEN3_ASR_TEXT_HPP
#define QWEN3_ASR_TEXT_HPP

#include <string>

namespace asr::qwen3 {

// Extracts the transcript from Qwen3-ASR's "language <Name><asr_text><transcript>" output and
// drops edge <|...|> control tokens. No speech ("language None<asr_text>") gives "".
std::string normalizeTranscript(const std::string& text);

}  // namespace asr::qwen3

#endif  // QWEN3_ASR_TEXT_HPP
