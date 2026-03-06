/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file model_loader.hpp
 * @brief Model loading and downloading utilities for SenseVoice
 *
 * Handles model file management, including automatic downloading
 * from remote sources when models are not available locally.
 */

#ifndef MODEL_LOADER_HPP
#define MODEL_LOADER_HPP

#include <string>
#include <functional>
#include <vector>

namespace asr {
namespace sensevoice {

/**
 * @class ModelLoader
 * @brief Manages SenseVoice model files
 *
 * Provides:
 * - Model file path management
 * - Automatic model downloading
 * - Archive extraction
 * - Path expansion (~ to home directory)
 */
class ModelLoader {
public:
    /**
     * @brief Download progress callback
     * @param progress Download progress [0.0, 1.0]
     */
    using ProgressCallback = std::function<void(double progress)>;

    /**
     * @brief Configuration
     */
    struct Config {
        std::string model_dir = "~/.cache/models/asr/sensevoice";
        std::string model_url = "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz";
        bool verify_checksum = false;
        std::string expected_checksum;
    };

    /**
     * @brief Model file names
     */
    struct ModelFiles {
        static constexpr const char* MODEL_ONNX = "model_quant_optimized.onnx";
        static constexpr const char* MODEL_FULL = "model_quant.onnx";
        static constexpr const char* CMVN = "am.mvn";
        static constexpr const char* VOCAB = "tokens.txt";
        static constexpr const char* DECODER = "sensevoice_decoder_model.onnx";
        static constexpr const char* CONFIG = "config.yaml";
    };

    ModelLoader();
    explicit ModelLoader(const Config& config);
    ~ModelLoader();

    /**
     * @brief Ensure all required models exist (download if needed)
     * @return true if all models are available
     */
    bool ensureModelsExist();

    /**
     * @brief Download models from remote URL
     * @param progress_cb Optional progress callback
     * @return true if successful
     */
    bool downloadModels(ProgressCallback progress_cb = nullptr);

    /**
     * @brief Get full path to a model file
     * @param filename Model filename
     * @return Full path
     */
    std::string getModelPath(const std::string& filename) const;

    /**
     * @brief Check if a model file exists
     * @param filename Model filename
     * @return true if exists
     */
    bool isModelAvailable(const std::string& filename) const;

    /**
     * @brief Get expanded model directory path
     */
    std::string getModelDir() const { return model_dir_expanded_; }

    /**
     * @brief Expand path (replace ~ with home directory)
     * @param path Path to expand
     * @return Expanded path
     */
    static std::string expandPath(const std::string& path);

    /**
     * @brief Get list of required model files
     */
    static std::vector<std::string> getRequiredFiles();

private:
    Config config_;
    std::string model_dir_expanded_;

    bool createDirectory();
    bool downloadFile(const std::string& url,
        const std::string& output_path,
        ProgressCallback progress_cb);
    bool extractArchive(const std::string& archive_path);
    bool verifyChecksum(const std::string& file_path, const std::string& expected);
    bool fileExists(const std::string& path) const;
    size_t getFileSize(const std::string& path) const;
};

}  // namespace sensevoice
}  // namespace asr

#endif  // MODEL_LOADER_HPP
