#!/usr/bin/env python3
"""
ASR 静态文件识别示例

只依赖 spacemit_asr 模块，用于识别音频文件。

Usage:
    python asr_file_demo.py audio.wav
    python asr_file_demo.py --list-backends
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='ASR 静态文件识别')
    parser.add_argument('file', nargs='?', help='要识别的音频文件 (16kHz WAV)')
    parser.add_argument('--model-dir', '-m', default='~/.cache/models/asr/sensevoice',
                        help='模型目录 (默认: ~/.cache/models/asr/sensevoice)')
    parser.add_argument('--language', '-l', default='zh',
                        choices=['zh', 'en', 'ja', 'ko', 'yue', 'auto'],
                        help='识别语言 (默认: zh)')
    parser.add_argument('--list-backends', action='store_true',
                        help='列出可用后端')

    args = parser.parse_args()

    # 导入 spacemit_asr
    try:
        import spacemit_asr
    except ImportError as e:
        print("错误: 无法导入 spacemit_asr 模块")
        print("请确保已设置 PYTHONPATH:")
        print("  export PYTHONPATH=/path/to/stt/build/python:/path/to/stt/python")
        print(f"\n详细错误: {e}")
        sys.exit(1)

    print(f"ASR 版本: {spacemit_asr.__version__}")

    if args.list_backends:
        print(f"可用后端: {spacemit_asr.Engine.get_available_backends()}")
        return

    if not args.file:
        parser.print_help()
        return

    if not os.path.exists(args.file):
        print(f"错误: 文件不存在 - {args.file}")
        sys.exit(1)

    # 配置
    lang_map = {
        'zh': spacemit_asr.Language.ZH,
        'en': spacemit_asr.Language.EN,
        'ja': spacemit_asr.Language.JA,
        'ko': spacemit_asr.Language.KO,
        'yue': spacemit_asr.Language.YUE,
        'auto': spacemit_asr.Language.AUTO,
    }

    config = spacemit_asr.Config(args.model_dir)
    config.language = lang_map[args.language]
    config.punctuation_enabled = True

    # 识别
    print(f"\n文件: {args.file}")
    print(f"语言: {args.language}")
    print("正在识别...\n")

    with spacemit_asr.Engine(config) as engine:
        result = engine.recognize_file(args.file)

        print(f"识别结果: {result.text}")
        print(f"音频时长: {result.audio_duration_ms} ms")
        print(f"处理时间: {result.processing_time_ms} ms")
        print(f"RTF: {result.rtf:.3f}")


if __name__ == "__main__":
    main()
