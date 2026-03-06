/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpaceAudioSDK 静态文件识别示例
 *
 * Usage:
 *   ./asr_file_demo [audio_file] [model_dir]
 *
 * Examples:
 *   ./asr_file_demo                              # 使用默认测试文件
 *   ./asr_file_demo ~/test.wav
 *   ./asr_file_demo ~/test.wav ~/.cache/models/asr/sensevoice
 */

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "asr_service.h"

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [audio_file] [model_dir]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  audio_file  Path to WAV audio file (default: ~/ringbuffer.wav)" << std::endl;
    std::cout << "  model_dir   Path to SenseVoice model directory" << std::endl;
    std::cout << "              Default: ~/.cache/models/asr/sensevoice" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << std::endl;
    std::cout << "  " << program << " ~/test.wav" << std::endl;
    std::cout << "  " << program << " ~/test.wav /path/to/models" << std::endl;
}

void TestFileRecognition(const std::string& audio_file, const std::string& model_dir) {
    std::cout << "========================================" << std::endl;
    std::cout << "    SpaceAudioSDK 文件识别测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // 1. 使用 AsrConfig 创建引擎
    std::cout << ">>> 创建 ASR 引擎 (使用 AsrConfig)..." << std::endl;

    SpacemiT::AsrConfig config = SpacemiT::AsrConfig::Preset("sensevoice");
    config.model_dir = model_dir;
    config.language = "zh";       // 设置语言
    config.punctuation = true;    // 启用自动标点

    auto asrEngine = std::make_shared<SpacemiT::AsrEngine>(config);

    if (!asrEngine->IsInitialized()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return;
    }

    // 显示当前配置
    auto cfg = asrEngine->GetConfig();
    std::cout << "引擎类型: " << cfg.engine << std::endl;
    std::cout << "语言: " << cfg.language << std::endl;
    std::cout << "标点: " << (cfg.punctuation ? "启用" : "禁用") << std::endl;
    std::cout << "采样率: " << cfg.sample_rate << " Hz" << std::endl;
    std::cout << "音频文件: " << audio_file << std::endl;
    std::cout << std::endl;

    // 2. 调用 Call 方法 (阻塞直到完成)
    std::cout << ">>> 开始识别文件..." << std::endl;
    auto result = asrEngine->Call(audio_file);

    // 3. 处理结果
    if (result && !result->IsEmpty()) {
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "           识别结果" << std::endl;
        std::cout << "========================================" << std::endl;

        // 使用 GetText() 获取完整文本
        std::cout << "完整文本: " << result->GetText() << std::endl;
        std::cout << "是否最终: " << (result->IsSentenceEnd() ? "是" : "否") << std::endl;
        std::cout << std::endl;

        // 使用 GetSentence() 获取主句子
        SpacemiT::Sentence sentence = result->GetSentence();
        std::cout << "--- 主句子 (GetSentence) ---" << std::endl;
        std::cout << "文本: " << sentence.text << std::endl;
        std::cout << "开始时间: " << sentence.begin_time << " ms" << std::endl;
        std::cout << "结束时间: " << sentence.end_time << " ms" << std::endl;
        std::cout << "置信度: " << std::fixed << std::setprecision(2) << sentence.confidence << std::endl;
        std::cout << std::endl;

        // 使用 GetSentences() 获取所有句子
        auto sentences = result->GetSentences();
        std::cout << "--- 所有句子 (GetSentences) ---" << std::endl;
        std::cout << "句子数量: " << sentences.size() << std::endl;
        for (size_t i = 0; i < sentences.size(); ++i) {
            const auto& s = sentences[i];
            std::cout << "[" << i << "] " << s.text
                << " (" << s.begin_time << "-" << s.end_time << " ms, "
                << "conf=" << std::fixed << std::setprecision(2) << s.confidence << ")"
                << std::endl;
        }
        std::cout << std::endl;

        std::cout << "--- 性能指标 ---" << std::endl;
        std::cout << "Request ID: " << result->GetRequestId() << std::endl;
        std::cout << "音频时长: " << result->GetAudioDuration() << " ms" << std::endl;
        std::cout << "处理时间: " << result->GetProcessingTime() << " ms" << std::endl;
        std::cout << "RTF: " << std::fixed << std::setprecision(3) << result->GetRTF() << std::endl;
        std::cout << std::endl;

        std::cout << "--- 延迟信息 ---" << std::endl;
        std::cout << "Last Request ID: " << asrEngine->GetLastRequestId() << std::endl;
        std::cout << "首包延迟: " << asrEngine->GetFirstPackageDelay() << " ms" << std::endl;
        std::cout << "尾包延迟: " << asrEngine->GetLastPackageDelay() << " ms" << std::endl;
        std::cout << "========================================" << std::endl;

        // 4. 获取 JSON 响应 (可选)
        std::cout << std::endl;
        std::cout << ">>> JSON 响应 (GetResponse):" << std::endl;
        std::cout << asrEngine->GetResponse() << std::endl;

    } else {
        std::cerr << "识别失败或未检测到语音" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // 检查帮助选项
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    // 解析参数
    std::string audio_file = "~/ringbuffer.wav";  // 默认测试文件
    std::string model_dir = "~/.cache/models/asr/sensevoice";  // 空则使用默认路径

    if (argc >= 2) {
        audio_file = argv[1];
    }
    if (argc >= 3) {
        model_dir = argv[2];
    }

    // 展开 ~ 为 HOME 目录
    if (!audio_file.empty() && audio_file[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            audio_file = std::string(home) + audio_file.substr(1);
        }
    }

    TestFileRecognition(audio_file, model_dir);

    std::cout << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}
