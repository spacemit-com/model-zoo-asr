/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backends/sensevoice/sensevoice_backend.hpp"

#include <sndfile.h>

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "backends/sensevoice/sensevoice_model.hpp"
#include "backends/sensevoice/model_loader.hpp"

namespace asr {

// Helper function to expand ~ to home directory
static std::string expandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE");  // Windows
    }

    if (!home) {
        return path;
    }

    return std::string(home) + path.substr(1);
}

// =============================================================================
// SenseVoiceBackend Implementation
// =============================================================================

SenseVoiceBackend::SenseVoiceBackend() = default;

SenseVoiceBackend::~SenseVoiceBackend() {
    shutdown();
}

ErrorInfo SenseVoiceBackend::initialize(const ASRConfig& config) {
    if (initialized_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Backend already initialized");
    }

    config_ = config;

    // Check and download models if needed
    sensevoice::ModelLoader::Config loader_config;

    // Extract directory from model_path
    std::string model_dir;
    if (!config_.model_path.empty()) {
        size_t last_slash = config_.model_path.rfind('/');
        if (last_slash != std::string::npos) {
            model_dir = config_.model_path.substr(0, last_slash);
        }
    }

    if (!model_dir.empty()) {
        loader_config.model_dir = model_dir;
    }

    sensevoice::ModelLoader loader(loader_config);
    if (!loader.ensureModelsExist()) {
        std::cout << "[SenseVoiceBackend] Models not found, will attempt to use provided paths"
                << std::endl;
    }

    // Initialize ASR model
    auto err = initializeASRModel();
    if (!err.isOk()) {
        return err;
    }

    initialized_.store(true);
    std::cout << "[SenseVoiceBackend] Initialized successfully" << std::endl;

    return ErrorInfo::ok();
}

ErrorInfo SenseVoiceBackend::initializeASRModel() {
    try {
        sensevoice::SenseVoiceModel::Config model_config;
        // Expand paths (handle ~ for home directory)
        model_config.model_path = expandPath(config_.model_path);
        model_config.cmvn_path = expandPath(config_.config_path);
        model_config.vocab_path = expandPath(config_.vocab_path);
        model_config.decoder_path = expandPath(config_.decoder_path);
        model_config.batch_size = 1;
        model_config.sample_rate = config_.sample_rate;
        model_config.language = languageToString(config_.language);
        model_config.use_itn = config_.itn_enabled;

        model_ = std::make_unique<sensevoice::SenseVoiceModel>(model_config);

        if (!model_->initialize()) {
            return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                "Failed to initialize SenseVoice model",
                "Model path: " + model_config.model_path);
        }

        std::cout << "[SenseVoiceBackend] SenseVoice model loaded: "
                << model_config.model_path << std::endl;
        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
            "Exception during SenseVoice model initialization",
            e.what());
    }
}

void SenseVoiceBackend::shutdown() {
    // Stop streaming if active
    if (stream_active_.load()) {
        stopStream();
    }

    // Release resources
    model_.reset();

    initialized_.store(false);
    std::cout << "[SenseVoiceBackend] Shutdown complete" << std::endl;
}

// =============================================================================
// Offline Recognition
// =============================================================================

ErrorInfo SenseVoiceBackend::recognize(const AudioChunk& audio, RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Backend not initialized");
    }

    if (!model_) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "SenseVoice model not available");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Convert audio to float format
    std::vector<float> audio_float = convertToFloat(audio);

    if (audio_float.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty or invalid audio data");
    }

    // Check sample rate - only 16kHz is supported
    if (audio.sample_rate != config_.sample_rate) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Unsupported sample rate: " + std::to_string(audio.sample_rate) + " Hz",
            "SenseVoice only supports " + std::to_string(config_.sample_rate) + " Hz input");
    }

    // Calculate audio duration
    int64_t audio_duration_ms = (audio_float.size() * 1000) / config_.sample_rate;

    // Call SenseVoice model
    std::string text;
    try {
        text = model_->recognize(audio_float);
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
            "SenseVoice inference failed", e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Build result
    result = buildResult(text, audio_duration_ms, processing_time, true);

    // Note: Don't notify here - ASREngine handles callback notifications
    // notifyResult(result);

    return ErrorInfo::ok();
}

ErrorInfo SenseVoiceBackend::recognizeFile(const std::string& file_path,
                                            RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Backend not initialized");
    }

    if (!model_) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "SenseVoice model not available");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Open audio file
    SF_INFO sf_info;
    memset(&sf_info, 0, sizeof(sf_info));

    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            "Failed to open audio file: " + file_path,
            sf_strerror(nullptr));
    }

    // Read audio data directly as float
    std::vector<float> audio_data(sf_info.frames * sf_info.channels);
    sf_count_t frames_read = sf_read_float(file, audio_data.data(), audio_data.size());
    sf_close(file);

    if (frames_read <= 0) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Failed to read audio data from file");
    }

    // Convert stereo to mono if needed (in-place when possible)
    std::vector<float>* audio_ptr = &audio_data;
    std::vector<float> mono_audio;

    if (sf_info.channels > 1) {
        mono_audio.resize(sf_info.frames);
        for (sf_count_t i = 0; i < sf_info.frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < sf_info.channels; ++ch) {
                sum += audio_data[i * sf_info.channels + ch];
            }
            mono_audio[i] = sum / sf_info.channels;
        }
        audio_ptr = &mono_audio;
    }

    // Check sample rate - only 16kHz is supported
    if (sf_info.samplerate != config_.sample_rate) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Unsupported sample rate: " + std::to_string(sf_info.samplerate) + " Hz",
            "SenseVoice only supports " + std::to_string(config_.sample_rate) + " Hz input. "
            "Please resample your audio to 16kHz before recognition.");
    }

    // Calculate audio duration (based on original file)
    int64_t audio_duration_ms = (sf_info.frames * 1000) / sf_info.samplerate;

    // Run SenseVoice model directly
    std::string text;
    try {
        text = model_->recognize(*audio_ptr);
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
            "SenseVoice inference failed", e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Build result
    result = buildResult(text, audio_duration_ms, processing_time, true);

    return ErrorInfo::ok();
}

// =============================================================================
// Streaming (Basic Implementation - VAD will be added later)
// =============================================================================

ErrorInfo SenseVoiceBackend::startStream() {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Backend not initialized");
    }

    if (stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Stream already active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    // Clear buffers
    audio_buffer_.clear();
    buffer_timestamp_ms_ = 0;

    stream_active_.store(true);
    notifyStart();

    std::cout << "[SenseVoiceBackend] Stream started" << std::endl;
    return ErrorInfo::ok();
}

ErrorInfo SenseVoiceBackend::feedAudio(const AudioChunk& audio) {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    // Convert and append audio
    auto audio_float = convertToFloat(audio);
    audio_buffer_.insert(audio_buffer_.end(), audio_float.begin(), audio_float.end());

    // For non-streaming mode (no VAD), we just accumulate audio
    // The actual recognition happens in stopStream()

    return ErrorInfo::ok();
}

ErrorInfo SenseVoiceBackend::stopStream() {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    // Process accumulated audio
    if (!audio_buffer_.empty()) {
        processBufferedAudio(true);
    }

    stream_active_.store(false);
    audio_buffer_.clear();

    notifyComplete();
    notifyClose();

    std::cout << "[SenseVoiceBackend] Stream stopped" << std::endl;
    return ErrorInfo::ok();
}

ErrorInfo SenseVoiceBackend::flushStream() {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    // Process accumulated audio (as final result for this segment)
    if (!audio_buffer_.empty()) {
        processBufferedAudio(true);
    }

    // Note: Keep stream active, don't call notifyComplete/notifyClose
    // User can continue sending audio for next segment

    std::cout << "[SenseVoiceBackend] Stream flushed" << std::endl;
    return ErrorInfo::ok();
}

void SenseVoiceBackend::processBufferedAudio(bool force_final) {
    if (audio_buffer_.empty() || !model_) {
        return;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Calculate audio duration
    int64_t audio_duration_ms = (audio_buffer_.size() * 1000) / config_.sample_rate;

    // Run SenseVoice model
    std::string text;
    try {
        text = model_->recognize(audio_buffer_);
    } catch (const std::exception& e) {
        notifyError(ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
            "SenseVoice inference failed", e.what()));
        return;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Build and notify result
    auto result = buildResult(text, audio_duration_ms, processing_time, force_final);
    notifyResult(result);

    // Clear buffer after processing
    audio_buffer_.clear();
}

// =============================================================================
// Dynamic Configuration
// =============================================================================

ErrorInfo SenseVoiceBackend::updateHotwords(const std::vector<std::string>& hotwords) {
    // SenseVoice doesn't support hotwords directly
    // This could be implemented with post-processing boost
    (void)hotwords;
    return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
        "Hotword update not supported by SenseVoice backend");
}

ErrorInfo SenseVoiceBackend::setLanguage(Language language) {
    config_.language = language;
    // Note: Language change requires model reinitialization
    // For now, just update config
    std::cout << "[SenseVoiceBackend] Language set to: "
            << languageToString(language) << std::endl;
    return ErrorInfo::ok();
}

// =============================================================================
// Helper Methods
// =============================================================================

std::vector<float> SenseVoiceBackend::convertToFloat(const AudioChunk& audio) {
    std::vector<float> result;

    if (audio.data == nullptr || audio.size_bytes == 0) {
        return result;
    }

    switch (audio.format) {
        case AudioFormat::PCM_S16LE: {
            const int16_t* data = static_cast<const int16_t*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(int16_t);
            result.resize(samples);
            for (size_t i = 0; i < samples; ++i) {
                result[i] = static_cast<float>(data[i]) / 32768.0f;
            }
            break;
        }
        case AudioFormat::PCM_F32LE: {
            const float* data = static_cast<const float*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(float);
            result.assign(data, data + samples);
            break;
        }
        default:
            std::cerr << "[SenseVoiceBackend] Unsupported audio format" << std::endl;
            break;
    }

    // Convert stereo to mono if needed
    if (audio.channels > 1 && !result.empty()) {
        std::vector<float> mono;
        size_t mono_samples = result.size() / audio.channels;
        mono.resize(mono_samples);
        for (size_t i = 0; i < mono_samples; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < audio.channels; ++ch) {
                sum += result[i * audio.channels + ch];
            }
            mono[i] = sum / audio.channels;
        }
        return mono;
    }

    return result;
}

RecognitionResult SenseVoiceBackend::buildResult(const std::string& text,
                                                int64_t audio_duration_ms,
                                                int64_t processing_time_ms,
                                                bool is_final) {
    RecognitionResult result;

    // Build sentence result
    SentenceResult sentence;
    sentence.text = text;
    sentence.begin_time_ms = 0;
    sentence.end_time_ms = static_cast<int32_t>(audio_duration_ms);
    sentence.confidence = 1.0f;  // SenseVoice doesn't provide confidence
    sentence.is_final = is_final;
    sentence.detected_language = config_.language;

    result.sentences.push_back(sentence);
    result.audio_duration_ms = audio_duration_ms;
    result.processing_time_ms = processing_time_ms;

    if (audio_duration_ms > 0) {
        result.rtf = static_cast<float>(processing_time_ms) / audio_duration_ms;
    }

    result.first_result_latency_ms = processing_time_ms;
    result.final_result_latency_ms = processing_time_ms;

    return result;
}

// =============================================================================
// Backend Factory
// =============================================================================

std::unique_ptr<IASRBackend> ASRBackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::SENSEVOICE:
            return std::make_unique<SenseVoiceBackend>();

        case BackendType::FUNASR:
            // TODO(spacemit): Implement FunASR backend
            std::cerr << "[ASRBackendFactory] FunASR backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::WHISPER:
            // TODO(spacemit): Implement Whisper backend
            std::cerr << "[ASRBackendFactory] Whisper backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::PARAFORMER:
            // TODO(spacemit): Implement Paraformer backend
            std::cerr << "[ASRBackendFactory] Paraformer backend not yet implemented" << std::endl;
            return nullptr;

        default:
            std::cerr << "[ASRBackendFactory] Unknown backend type" << std::endl;
            return nullptr;
    }
}

bool ASRBackendFactory::isAvailable(BackendType type) {
    switch (type) {
        case BackendType::SENSEVOICE:
            return true;  // Always available (built-in)
        default:
            return false;
    }
}

std::vector<BackendType> ASRBackendFactory::getAvailableBackends() {
    std::vector<BackendType> backends;
    backends.push_back(BackendType::SENSEVOICE);
    // Add more as they become available
    return backends;
}

}  // namespace asr
