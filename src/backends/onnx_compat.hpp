/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_BACKENDS_ONNX_COMPAT_HPP
#define ASR_BACKENDS_ONNX_COMPAT_HPP

// Some K1 GCC builds do not expose the non-standard __fp16 keyword used by
// ONNX Runtime headers. CMake enables this typedef only after confirming that
// the compiler lacks __fp16 but supports the standard _Float16 type.
#if defined(ASR_ONNXRUNTIME_USE_FLOAT16_COMPAT)
typedef _Float16 __fp16;
#endif

#include <onnxruntime_cxx_api.h>

#endif  // ASR_BACKENDS_ONNX_COMPAT_HPP
