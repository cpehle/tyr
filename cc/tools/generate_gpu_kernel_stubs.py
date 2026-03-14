#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re


PATTERN = re.compile(rb"lean_launch_[A-Za-z0-9_]+")


def collect_symbols(ir_root: pathlib.Path) -> list[str]:
    symbols: set[str] = set()
    if not ir_root.exists():
        return []
    for path in ir_root.rglob("*.c.o.export"):
        data = path.read_bytes()
        for match in PATTERN.finditer(data):
            symbols.add(match.group().decode("utf-8"))
    return sorted(symbols)


def render(symbols: list[str]) -> str:
    lines = [
        "#include <lean/lean.h>",
        "#include <string>",
        "",
        "static lean_object* gpuKernelUnavailable(const char* launcher) {",
        '  std::string msg = "GPU kernel launcher unavailable in this build (missing NVCC/CUDA): ";',
        "  msg += launcher;",
        "  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));",
        "}",
        "",
        'extern "C" {',
        "",
    ]
    for symbol in symbols:
        lines.extend([
            f"__attribute__((weak)) lean_object* {symbol}(...) {{",
            f'  return gpuKernelUnavailable("{symbol}");',
            "}",
            "",
        ])
    lines.append("} // extern \"C\"")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ir_root = pathlib.Path(args.ir_root)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(collect_symbols(ir_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
