from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.io import build_report, read_stereo_wav, write_stems
from chorus.reporting import write_markdown_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chorus", description="CHORUS reference demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="split stereo WAV into CHORUS stems")
    split.add_argument("input", type=Path)
    split.add_argument("--out-dir", type=Path, required=True)
    split.add_argument("--transform", choices=["stft", "frft", "wavelet"], default="stft")
    split.add_argument("--frame-size", type=int, default=1024)
    split.add_argument("--hop-size", type=int, default=512)
    split.add_argument("--smoothing-alpha", type=float, default=0.9)
    split.add_argument("--epsilon", type=float, default=1e-9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "split":
        sample_rate, audio = read_stereo_wav(args.input)
        config = ChorusConfig(
            sample_rate=sample_rate,
            transform=args.transform,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            smoothing_alpha=args.smoothing_alpha,
            epsilon=args.epsilon,
        )
        processor = ChorusProcessor(config)
        result = processor.process(audio)
        stem_paths = write_stems(args.out_dir, sample_rate, result)
        report = build_report(
            input_path=args.input,
            output_dir=args.out_dir,
            sample_rate=sample_rate,
            input_audio=audio,
            result=result,
            stem_paths=stem_paths,
        )
        report_path = args.out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_markdown_report(args.out_dir, report)
        return 0

    parser.error(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
