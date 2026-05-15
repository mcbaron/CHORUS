from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chorus", description="CHORUS reference demo")
    parser.parse_args(argv)
    return 0
