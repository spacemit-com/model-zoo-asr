cmake_minimum_required(VERSION 3.16)
project(STT VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code for all targets (required for RISC-V shared libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Build Options
# -------------------------------------------------------
# BUILD_STREAM_DEMO: 编译麦克风流式识别示例和 Python 绑定
#
# 需要额外依赖:
#   1. PortAudio: brew install portaudio (macOS) / apt install portaudio19-dev (Linux)
#   2. audio 组件: SDK 编译时通过 package.xml 依赖自动构建安装到 staging
#
# 使用方法:
#   SDK 编译时默认开启；独立编译时: cmake .. -DBUILD_STREAM_DEMO=ON
#
# 检测是否在 SDK build 流程内编译（envsetup.sh 已 source 则 SROBOTIS_OUTPUT_ROOT 被 export）。
# 仅文件存在不算——源码可能在仓库里，但用户裸 cmake 时 audio 组件并不一定已装，
# 此时不应默认打开 BUILD_STREAM_DEMO。
if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
    set(_IN_SDK_BUILD ON)
else()
    set(_IN_SDK_BUILD OFF)
endif()

option(BUILD_STREAM_DEMO "Build stream demo and Python bindings (requires audio component)" ${_IN_SDK_BUILD})
option(BUILD_PYTHON_BINDINGS "Build Python bindings (requires BUILD_STREAM_DEMO=ON)" ON)
option(ASR_MODEL_FETCH_OFF "Disable automatic model downloading at configure time" ON)

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    /opt/homebrew/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

option(USE_SPACEMIT_EP "Enable SpaceMIT Execution Provider for ASR" ON)

# libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# libcurl
pkg_check_modules(CURL REQUIRED libcurl)

# FFTW3
if(APPLE)
    find_library(FFTW3_LIBRARIES NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
else()
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# libsamplerate (optional)
pkg_check_modules(SAMPLERATE samplerate)

# kaldi-native-fbank (for Zipformer feature extraction)
include(FetchContent)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    kaldi_native_fbank
    GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank
    GIT_TAG master
)
FetchContent_MakeAvailable(kaldi_native_fbank)
set(KALDI_FBANK_INCLUDE_DIRS
    ${kaldi_native_fbank_SOURCE_DIR}
    ${kaldi_native_fbank_SOURCE_DIR}/kaldi-native-fbank/csrc
)

# -------------------------------------------------------
# Fetch ASR Models
# -------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchSenseVoiceModel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchZipformerModel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchQwen3AsrModel.cmake)

# -------------------------------------------------------
# Audio Component (optional, for stream demo)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    find_library(SPACE_AUDIO_LIB space_audio)
    find_library(AUDIO_RESAMPLER_LIB audio_resampler)
    find_path(AUDIO_INC_DIR audio_base.hpp)

    if(NOT SPACE_AUDIO_LIB OR NOT AUDIO_RESAMPLER_LIB OR NOT AUDIO_INC_DIR)
        message(FATAL_ERROR
            "BUILD_STREAM_DEMO=ON but audio component not found!\n"
            "Install audio component first, or disable with -DBUILD_STREAM_DEMO=OFF")
    endif()

    # PortAudio (required for stream demo)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

# -------------------------------------------------------
# SenseVoice Backend Library
# -------------------------------------------------------
set(SENSEVOICE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/feature_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/tokenizer.cpp
)

add_library(sensevoice STATIC ${SENSEVOICE_SOURCES})

target_include_directories(sensevoice PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(sensevoice
    PUBLIC ${ONNXRUNTIME_LIB}
    PUBLIC ${FFTW3_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

# -------------------------------------------------------
# Zipformer Backend Library
# -------------------------------------------------------
set(ZIPFORMER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_ctc_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_feature_extractor.cpp
)

add_library(zipformer STATIC ${ZIPFORMER_SOURCES})

target_include_directories(zipformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${KALDI_FBANK_INCLUDE_DIRS}
)

target_link_libraries(zipformer
    PUBLIC ${ONNXRUNTIME_LIB}
    PUBLIC kaldi-native-fbank-core
)

set_target_properties(zipformer PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# -------------------------------------------------------
# STT Framework Library
# -------------------------------------------------------
set(STT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_downloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/qwen3_asr/qwen3_asr_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_presets.cpp
)

add_library(stt STATIC ${STT_SOURCES})

target_include_directories(stt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CURL_INCLUDE_DIRS}
)

# Conditionally add resampler include
if(BUILD_STREAM_DEMO)
    target_include_directories(stt PUBLIC ${AUDIO_INC_DIR})
    target_link_libraries(stt PUBLIC ${AUDIO_RESAMPLER_LIB} ${SAMPLERATE_LIBRARIES})
endif()

target_link_libraries(stt
    PUBLIC sensevoice
    PUBLIC zipformer
    PUBLIC ${SNDFILE_LIBRARIES}
    PRIVATE ${CURL_LIBRARIES}
    PRIVATE pthread
)

target_link_directories(stt PRIVATE ${CURL_LIBRARY_DIRS})

target_link_directories(stt PUBLIC ${SNDFILE_LIBRARY_DIRS})

set_target_properties(stt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(sensevoice PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "STT Framework library enabled")

# -------------------------------------------------------
# Install rules
# -------------------------------------------------------
install(TARGETS stt sensevoice zipformer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES include/asr_service.h DESTINATION include)

# -------------------------------------------------------
# ASR File Demo Executable (always built)
# -------------------------------------------------------
add_executable(asr_file_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/asr_file_demo.cpp)

target_include_directories(asr_file_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(asr_file_demo
    PRIVATE stt
    PRIVATE ${SNDFILE_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

target_link_directories(asr_file_demo PRIVATE ${SNDFILE_LIBRARY_DIRS})

set_target_properties(asr_file_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS asr_file_demo RUNTIME DESTINATION bin)

if(USE_SPACEMIT_EP)
    find_path(SPACEMIT_ORT_INCLUDE_DIR
        NAMES spacemit_ort_env.h
        PATHS ${ONNXRUNTIME_INCLUDE_DIR} /usr/include /usr/local/include
        PATH_SUFFIXES onnxruntime
    )
    find_library(SPACEMIT_EP_LIB
        NAMES spacemit_ep
        PATHS /usr/local/lib /usr/lib
    )
    if(SPACEMIT_ORT_INCLUDE_DIR AND SPACEMIT_EP_LIB)
        target_compile_definitions(sensevoice PRIVATE USE_SPACEMIT_EP)
        target_include_directories(sensevoice PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        target_link_libraries(sensevoice PUBLIC ${SPACEMIT_EP_LIB})
        target_compile_definitions(zipformer PRIVATE USE_SPACEMIT_EP)
        target_include_directories(zipformer PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        target_link_libraries(zipformer PUBLIC ${SPACEMIT_EP_LIB})
        target_compile_definitions(asr_file_demo PRIVATE USE_SPACEMIT_EP)
        target_include_directories(asr_file_demo PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        message(STATUS "SpaceMIT EP enabled: ${SPACEMIT_ORT_INCLUDE_DIR}, lib: ${SPACEMIT_EP_LIB}")
    else()
        message(WARNING "SpaceMIT EP not found (header=${SPACEMIT_ORT_INCLUDE_DIR}, lib=${SPACEMIT_EP_LIB}), USE_SPACEMIT_EP ignored")
    endif()
endif()

message(STATUS "ASR file demo executable enabled")

# -------------------------------------------------------
# ASR Stream Demo Executable (optional)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    add_executable(asr_stream_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/asr_stream_demo.cpp)

    target_include_directories(asr_stream_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${AUDIO_INC_DIR}
    )

    target_link_libraries(asr_stream_demo
        PRIVATE stt
        PRIVATE ${SPACE_AUDIO_LIB}
        PRIVATE ${AUDIO_RESAMPLER_LIB}
        PRIVATE ${PORTAUDIO_LIBRARIES}
        PRIVATE ${SAMPLERATE_LIBRARIES}
        PRIVATE ${SNDFILE_LIBRARIES}
        PRIVATE pthread
        PRIVATE m
    )

    target_link_directories(asr_stream_demo PRIVATE
        ${SNDFILE_LIBRARY_DIRS}
        ${PORTAUDIO_LIBRARY_DIRS}
        ${SAMPLERATE_LIBRARY_DIRS}
    )

    if(SAMPLERATE_FOUND)
        target_compile_definitions(asr_stream_demo PRIVATE USE_LIBSAMPLERATE)
    endif()

    if(USE_SPACEMIT_EP AND SPACEMIT_ORT_INCLUDE_DIR)
        target_compile_definitions(asr_stream_demo PRIVATE USE_SPACEMIT_EP)
        target_include_directories(asr_stream_demo PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
    endif()

    set_target_properties(asr_stream_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    install(TARGETS asr_stream_demo RUNTIME DESTINATION bin)

    message(STATUS "ASR stream demo executable enabled")
else()
    message(STATUS "ASR stream demo disabled (BUILD_STREAM_DEMO=OFF)")
    message(STATUS "  To enable: cmake .. -DBUILD_STREAM_DEMO=ON")
    message(STATUS "  Requires: audio component + PortAudio")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    # Find Python first (respects -DPython3_EXECUTABLE=...)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    endif()

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try using Python's pybind11
        if(Python3_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE PYBIND11_RESULT
            )
            if(PYBIND11_RESULT EQUAL 0)
                list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
                find_package(pybind11 CONFIG QUIET)
            endif()
        endif()
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_spacemit_asr ${CMAKE_CURRENT_SOURCE_DIR}/python/asr_bindings.cpp)

        target_include_directories(_spacemit_asr PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

        target_link_libraries(_spacemit_asr PRIVATE stt)

        set_target_properties(_spacemit_asr PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
        )

        message(STATUS "Python bindings enabled (_spacemit_asr module for Python ${Python3_VERSION})")

        # Get Python site-packages path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Install target for pip
        install(TARGETS _spacemit_asr LIBRARY DESTINATION .)

        # Custom target: stt-install-python (copy all files to virtualenv site-packages)
        add_custom_target(stt-install-python
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/spacemit_asr
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_asr ${PYTHON_SITE_PACKAGES}/spacemit_asr
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_asr> ${PYTHON_SITE_PACKAGES}/spacemit_asr/
            DEPENDS _spacemit_asr
            COMMENT "Installing spacemit_asr to ${PYTHON_SITE_PACKAGES}/spacemit_asr/"
        )

        # ---- wheel 自动构建与 install（对接 SDK 顶层产物）----
        set(ASR_WHEEL_BUILD_ROOT      "${CMAKE_BINARY_DIR}/python_wheel")
        set(ASR_WHEEL_OUTPUT_DIR      "${ASR_WHEEL_BUILD_ROOT}/dist")
        set(ASR_WHEEL_CMAKE_BUILD_DIR "${ASR_WHEEL_BUILD_ROOT}/cmake")
        set(ASR_WHEEL_SOURCE_DIR      "${ASR_WHEEL_BUILD_ROOT}/src")
        set(ASR_WHEEL_STAMP           "${ASR_WHEEL_BUILD_ROOT}/wheel.stamp")

        if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
            set(ASR_DIST_INSTALL_DIR "$ENV{SROBOTIS_OUTPUT_ROOT}/dist")
        else()
            set(ASR_DIST_INSTALL_DIR "${CMAKE_BINARY_DIR}/dist")
        endif()

        file(MAKE_DIRECTORY "${ASR_WHEEL_OUTPUT_DIR}")

        add_custom_command(
            OUTPUT  ${ASR_WHEEL_STAMP}
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${ASR_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${ASR_WHEEL_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASR_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASR_WHEEL_CMAKE_BUILD_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/python ${ASR_WHEEL_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${ASR_WHEEL_SOURCE_DIR}/README.md
            COMMAND ${CMAKE_COMMAND} -E chdir ${ASR_WHEEL_SOURCE_DIR}
                    ${CMAKE_COMMAND} -E env
                    SPACEMIT_PREBUILT_EXTENSION=$<TARGET_FILE:_spacemit_asr>
                    SPACEMIT_CMAKE_BUILD_DIR=${ASR_WHEEL_CMAKE_BUILD_DIR}
                    ${Python3_EXECUTABLE} -m pip wheel . --no-deps --wheel-dir ${ASR_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E touch ${ASR_WHEEL_STAMP}
            DEPENDS
                _spacemit_asr
                ${CMAKE_CURRENT_SOURCE_DIR}/README.md
                ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml
                ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_asr/__init__.py
            VERBATIM
            COMMENT "Building spacemit_asr wheel into ${ASR_WHEEL_OUTPUT_DIR}"
        )

        add_custom_target(asr-python-wheel ALL DEPENDS ${ASR_WHEEL_STAMP})

        install(
            DIRECTORY ${ASR_WHEEL_OUTPUT_DIR}/
            DESTINATION ${ASR_DIST_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.whl"
        )
    else()
        message(STATUS "Python bindings disabled (pybind11 not found)")
        message(STATUS "  Install with: pip install pybind11 OR brew install pybind11")
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "=== STT Build Configuration ===")
message(STATUS "  BUILD_STREAM_DEMO:      ${BUILD_STREAM_DEMO}")
message(STATUS "  BUILD_PYTHON_BINDINGS:  ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Wheel output:           ${ASR_WHEEL_OUTPUT_DIR}")
message(STATUS "  Wheel install dir:      ${ASR_DIST_INSTALL_DIR}")
message(STATUS "")
