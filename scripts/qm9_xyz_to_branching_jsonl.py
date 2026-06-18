#!/usr/bin/env python3
"""
Convert QM9 XYZ coordinate files into the Tyr BranchingFlows QM9 JSONL schema.

The Branching Flows paper uses QM9 coordinate data with canonical-SMILES heavy
atom order and hydrogens moved adjacent to their nearest heavy atom. This
script keeps the file order of heavy atoms and reinserts hydrogens immediately
before the nearest heavy atom, sorted by distance for each heavy atom.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


ATOM_LABELS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}


@dataclass(frozen=True)
class Atom:
    symbol: str
    label: int
    coord: tuple[float, float, float]


def squared_distance(a: Atom, b: Atom) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a.coord, b.coord, strict=True))


def reorder_hydrogens_near_heavy(atoms: list[Atom]) -> list[Atom]:
    heavy = [atom for atom in atoms if atom.symbol != "H"]
    hydrogens = [atom for atom in atoms if atom.symbol == "H"]
    if not heavy or not hydrogens:
        return atoms

    by_heavy: dict[int, list[Atom]] = {i: [] for i in range(len(heavy))}
    for hydrogen in hydrogens:
        nearest = min(range(len(heavy)), key=lambda i: squared_distance(hydrogen, heavy[i]))
        by_heavy[nearest].append(hydrogen)

    ordered: list[Atom] = []
    for i, atom in enumerate(heavy):
        ordered.extend(sorted(by_heavy[i], key=lambda h: squared_distance(h, atom)))
        ordered.append(atom)
    return ordered


def parse_atom(line: str, path: Path, line_number: int) -> Atom:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"{path}:{line_number}: expected atom symbol and x/y/z coordinates")
    symbol = parts[0]
    if symbol not in ATOM_LABELS:
        raise ValueError(f"{path}:{line_number}: unsupported QM9 atom symbol {symbol!r}")
    try:
        coord = (float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid coordinate line {line!r}") from exc
    if not all(math.isfinite(x) for x in coord):
        raise ValueError(f"{path}:{line_number}: non-finite coordinate line {line!r}")
    return Atom(symbol=symbol, label=ATOM_LABELS[symbol], coord=coord)


def infer_smiles(extra_lines: list[str]) -> str | None:
    for line in extra_lines:
        tokens = line.split()
        if not tokens:
            continue
        token = tokens[0]
        if token.startswith("InChI"):
            continue
        if any(ch.isalpha() for ch in token):
            return token
    return None


def parse_xyz(path: Path, keep_original_order: bool) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: expected XYZ header")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path}: first line must be an atom count") from exc
    if atom_count <= 0:
        raise ValueError(f"{path}: atom count must be positive")
    if len(lines) < atom_count + 2:
        raise ValueError(f"{path}: expected {atom_count} atom lines")

    atoms = [parse_atom(lines[2 + i], path, 3 + i) for i in range(atom_count)]
    if not keep_original_order:
        atoms = reorder_hydrogens_near_heavy(atoms)

    record: dict[str, object] = {
        "name": path.stem,
        "source_path": str(path),
        "atoms": [
            {
                "symbol": atom.symbol,
                "label": atom.label,
                "coord": [atom.coord[0], atom.coord[1], atom.coord[2]],
            }
            for atom in atoms
        ],
    }
    smiles = infer_smiles(lines[2 + atom_count :])
    if smiles is not None:
        record["canonical_smiles"] = smiles
    return record


def iter_xyz_paths(inputs: Iterable[Path]) -> Iterable[Path]:
    for input_path in inputs:
        path = input_path.expanduser()
        if path.is_dir():
            yield from sorted(path.rglob("*.xyz"))
        else:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QM9 .xyz files or directories")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--keep-original-order",
        action="store_true",
        help="Do not move hydrogens next to nearest heavy atoms.",
    )
    parser.add_argument("--max-molecules", type=int, default=0, help="Optional cap for smoke runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(iter_xyz_paths(args.inputs))
    if args.max_molecules > 0:
        paths = paths[: args.max_molecules]
    if not paths:
        raise SystemExit("no .xyz inputs found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in paths:
            record = parse_xyz(path, keep_original_order=args.keep_original_order)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    print(f"wrote {count} molecule records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
