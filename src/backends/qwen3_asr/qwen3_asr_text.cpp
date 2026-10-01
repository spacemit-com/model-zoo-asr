/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3_asr_text.hpp"

#include <cctype>
#include <string>

namespace asr::qwen3 {
namespace {

void trimAsciiWhitespace(std::string* text) {
    size_t begin = 0;
    while (begin < text->size() &&
            std::isspace(static_cast<unsigned char>((*text)[begin]))) {
        ++begin;
    }

    size_t end = text->size();
    while (end > begin &&
            std::isspace(static_cast<unsigned char>((*text)[end - 1]))) {
        --end;
    }

    *text = text->substr(begin, end - begin);
}

bool stripLeadingControlToken(std::string* text) {
    if (text->rfind("<|", 0) != 0) {
        return false;
    }
    const size_t end = text->find("|>", 2);
    if (end == std::string::npos) {
        return false;
    }
    text->erase(0, end + 2);
    trimAsciiWhitespace(text);
    return true;
}

bool stripTrailingControlToken(std::string* text) {
    if (text->size() < 4 || text->compare(text->size() - 2, 2, "|>") != 0) {
        return false;
    }
    const size_t begin = text->rfind("<|");
    if (begin == std::string::npos) {
        return false;
    }
    text->erase(begin);
    trimAsciiWhitespace(text);
    return true;
}

}  // namespace

std::string normalizeTranscript(const std::string& text) {
    // Qwen3-ASR answers "language <Name><asr_text><transcript>" ("language None<asr_text>"
    // when there is no speech); llama-server passes it through unparsed.
    static const std::string kAsrTextMarker = "<asr_text>";

    std::string normalized = text;
    trimAsciiWhitespace(&normalized);
    while (stripLeadingControlToken(&normalized)) {
    }

    const size_t marker = normalized.rfind(kAsrTextMarker);
    if (marker != std::string::npos) {
        normalized.erase(0, marker + kAsrTextMarker.size());
        trimAsciiWhitespace(&normalized);
    }

    while (stripTrailingControlToken(&normalized)) {
    }
    return normalized;
}

}  // namespace asr::qwen3
