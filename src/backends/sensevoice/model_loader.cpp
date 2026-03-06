/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file model_loader.cpp
 * @brief Model loading and downloading implementation
 */

#include "backends/sensevoice/model_loader.hpp"

#include <curl/curl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace asr {
namespace sensevoice {

ModelLoader::ModelLoader()
    : config_(Config{})
{
    model_dir_expanded_ = expandPath(config_.model_dir);
}

ModelLoader::ModelLoader(const Config& config)
    : config_(config)
{
    model_dir_expanded_ = expandPath(config_.model_dir);
}

ModelLoader::~ModelLoader() = default;

std::string ModelLoader::expandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE");  // Windows
    }

    return home ? (std::string(home) + path.substr(1)) : path;
}

std::vector<std::string> ModelLoader::getRequiredFiles() {
    return {
        ModelFiles::MODEL_ONNX,
        ModelFiles::VOCAB,
        ModelFiles::CMVN
    };
}

bool ModelLoader::ensureModelsExist() {
    if (!createDirectory()) {
        return false;
    }

    auto required = getRequiredFiles();
    bool all_exist = true;

    for (const auto& file : required) {
        if (!isModelAvailable(file)) {
            std::cout << "[ModelLoader] Missing: " << file << std::endl;
            all_exist = false;
        }
    }

    if (!all_exist) {
        std::cout << "[ModelLoader] Downloading models..." << std::endl;
        return downloadModels();
    }

    std::cout << "[ModelLoader] All models available" << std::endl;
    return true;
}

bool ModelLoader::downloadModels(ProgressCallback progress_cb) {
    std::string archive_path = model_dir_expanded_ + "/sensevoice.tar.gz";

    if (!downloadFile(config_.model_url, archive_path, progress_cb)) {
        return false;
    }

    if (!extractArchive(archive_path)) {
        return false;
    }

    // Cleanup archive
    std::filesystem::remove(archive_path);

    return true;
}

std::string ModelLoader::getModelPath(const std::string& filename) const {
    return model_dir_expanded_ + "/" + filename;
}

bool ModelLoader::isModelAvailable(const std::string& filename) const {
    return fileExists(getModelPath(filename));
}

bool ModelLoader::createDirectory() {
    try {
        std::filesystem::create_directories(model_dir_expanded_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ModelLoader] Failed to create directory: " << e.what() << std::endl;
        return false;
    }
}

bool ModelLoader::fileExists(const std::string& path) const {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

size_t ModelLoader::getFileSize(const std::string& path) const {
    try {
        return std::filesystem::file_size(path);
    } catch (const std::exception&) {
        return 0;
    }
}

// CURL callback for writing data
static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t total_size = size * nmemb;
    file->write(static_cast<const char*>(contents), total_size);
    return total_size;
}

// CURL progress callback wrapper
struct ProgressData {
    ModelLoader::ProgressCallback cb;
};

static int progressCallback(void* clientp, double dltotal, double dlnow,
                            double /*ultotal*/, double /*ulnow*/) {
    auto* data = static_cast<ProgressData*>(clientp);
    if (data->cb && dltotal > 0) {
        data->cb(dlnow / dltotal);
    }
    return 0;
}

bool ModelLoader::downloadFile(const std::string& url,
                                const std::string& output_path,
                                ProgressCallback progress_cb) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ModelLoader] Failed to init curl" << std::endl;
        return false;
    }

    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ModelLoader] Cannot open: " << output_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    ProgressData progress_data{progress_cb};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "SenseVoice/1.0");

    if (progress_cb) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_data);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    int64_t response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    curl_easy_cleanup(curl);
    file.close();

    if (res != CURLE_OK) {
        std::cerr << "[ModelLoader] Download failed: " << curl_easy_strerror(res) << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }

    if (response_code != 200) {
        std::cerr << "[ModelLoader] HTTP error: " << response_code << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }

    std::cout << "[ModelLoader] Downloaded: " << output_path << std::endl;
    return true;
}

bool ModelLoader::extractArchive(const std::string& archive_path) {
    std::string temp_dir = model_dir_expanded_ + "/temp_extract";

    // Clean up temp dir if exists
    if (std::filesystem::exists(temp_dir)) {
        std::filesystem::remove_all(temp_dir);
    }
    std::filesystem::create_directories(temp_dir);

    std::string cmd = "tar -xzf \"" + archive_path + "\" -C \"" + temp_dir + "\"";

    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "[ModelLoader] Extract failed" << std::endl;
        std::filesystem::remove_all(temp_dir);
        return false;
    }

    // Check for subdirectory
    std::string subdir = temp_dir + "/sensevoice";
    std::string src_dir = std::filesystem::exists(subdir) ? subdir : temp_dir;

    // Copy files to model directory (skip directories, handle existing files)
    for (const auto& entry : std::filesystem::directory_iterator(src_dir)) {
        std::string dest =
            model_dir_expanded_ + "/" + entry.path().filename().string();

        try {
            if (entry.is_regular_file()) {
                // For files: copy and overwrite if exists
                std::filesystem::copy_file(entry.path(), dest,
                    std::filesystem::copy_options::overwrite_existing);
            } else if (entry.is_directory()) {
                // For directories: copy recursively, skip if target exists
                if (!std::filesystem::exists(dest)) {
                    std::filesystem::copy(entry.path(), dest,
                        std::filesystem::copy_options::recursive);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ModelLoader] Warning: failed to copy "
                    << entry.path().filename().string()
                    << ": " << e.what() << std::endl;
            // Continue with other files
        }
    }

    std::filesystem::remove_all(temp_dir);

    std::cout << "[ModelLoader] Extracted successfully" << std::endl;
    return true;
}

bool ModelLoader::verifyChecksum(const std::string& file_path, const std::string& expected) {
    if (!config_.verify_checksum || expected.empty()) {
        return true;
    }

    std::string cmd = "sha256sum \"" + file_path + "\" | cut -d' ' -f1";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return false;
    }

    char buffer[128];
    std::string actual;
    while (!feof(pipe)) {
        if (fgets(buffer, sizeof(buffer), pipe)) {
            actual += buffer;
        }
    }
    pclose(pipe);

    // Trim newline
    if (!actual.empty() && actual.back() == '\n') {
        actual.pop_back();
    }

    return actual == expected;
}

}  // namespace sensevoice
}  // namespace asr
