#!/usr/bin/env bash
set -euo pipefail

module_dir="components/model_zoo/asr"

python3 -m py_compile \
  "${module_dir}/python/spacemit_asr/engine.py" \
  "${module_dir}/python/spacemit_asr/__init__.py" \
  "${module_dir}/python/examples/asr_file_demo.py"

python3 - <<'PY'
import enum
import sys
import types
from pathlib import Path

module_dir = Path("components/model_zoo/asr")
source_root = module_dir / "python"

bindings = (module_dir / "python/asr_bindings.cpp").read_text()
assert '.value("QWEN3_ASR"' in bindings
assert '.def_static("zipformer"' in bindings
assert '.def_static("qwen3_asr"' in bindings


class Language(enum.Enum):
    AUTO = 0
    ZH = 1
    EN = 2
    JA = 3
    KO = 4
    YUE = 5


class BackendType(enum.Enum):
    SENSEVOICE = 0
    FUNASR = 1
    WHISPER = 2
    PARAFORMER = 3
    QWEN3_ASR = 4
    ZIPFORMER = 5
    CUSTOM = 6


class NativeConfig:
    def __init__(self):
        self.backend = BackendType.SENSEVOICE
        self.language = Language.AUTO
        self.sample_rate = 16000
        self.punctuation_enabled = True
        self.enable_emotion = False
        self.extra_params = {}

    @staticmethod
    def sensevoice(model_dir):
        config = NativeConfig()
        config.backend = BackendType.SENSEVOICE
        config.model_path = f"{model_dir}/model_quant_optimized.onnx"
        return config

    @staticmethod
    def zipformer(model_dir):
        config = NativeConfig()
        config.backend = BackendType.ZIPFORMER
        config.model_path = f"{model_dir}/ctc-epoch-20-avg-1-chunk-16-left-128.q.onnx"
        return config

    @staticmethod
    def qwen3_asr(endpoint, model, timeout):
        config = NativeConfig()
        config.backend = BackendType.QWEN3_ASR
        config.extra_params = {
            "endpoint": endpoint,
            "model": model,
            "timeout": str(timeout),
        }
        return config


class NativeEngine:
    @staticmethod
    def get_available_backends():
        return [BackendType.SENSEVOICE, BackendType.QWEN3_ASR, BackendType.ZIPFORMER]

    @staticmethod
    def get_version():
        return "contract-test"


fake_asr = types.SimpleNamespace(
    Language=Language,
    BackendType=BackendType,
    ASRConfig=NativeConfig,
    ASREngine=NativeEngine,
)

sys.modules["spacemit_asr._spacemit_asr"] = fake_asr
sys.path.insert(0, str(source_root.resolve()))

import spacemit_asr

assert spacemit_asr.BackendType.QWEN3_ASR.name == "QWEN3_ASR"
assert spacemit_asr.Config().backend == spacemit_asr.BackendType.SENSEVOICE
assert spacemit_asr.Config.zipformer().backend == spacemit_asr.BackendType.ZIPFORMER

qwen = spacemit_asr.Config.qwen3_asr(
    endpoint="http://127.0.0.1:8063/v1/chat/completions",
    model="qwen3-asr",
    timeout=7,
)
assert qwen.backend == spacemit_asr.BackendType.QWEN3_ASR
assert qwen._config.extra_params["timeout"] == "7"

backends = spacemit_asr.Engine.get_available_backends()
assert backends == [
    spacemit_asr.BackendType.SENSEVOICE,
    spacemit_asr.BackendType.QWEN3_ASR,
    spacemit_asr.BackendType.ZIPFORMER,
]

print("PASS --python-api-contract")
PY
