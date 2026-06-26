from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta


DNA_ALPHABET = np.asarray(list("ACGT"))


def gc_fraction(sequence: str) -> float:
    seq = sequence.upper()
    bases = [base for base in seq if base in "ACGT"]
    if not bases:
        return 0.0
    gc = sum(base in "GC" for base in bases)
    return float(gc / len(bases))


def n_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return float(sequence.upper().count("N") / len(sequence))


def add_sequence_stats(frame: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    out = frame.copy()
    out["gc"] = out[sequence_col].map(gc_fraction)
    out["n_fraction"] = out[sequence_col].map(n_fraction)
    out["length"] = out[sequence_col].map(len)
    return out


def random_dna(length: int, gc: float, rng: np.random.Generator) -> str:
    gc = min(max(gc, 0.02), 0.98)
    probs = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return "".join(rng.choice(DNA_ALPHABET, size=length, p=probs))


def insert_motif(sequence: str, motif: str, rng: np.random.Generator) -> str:
    if len(motif) >= len(sequence):
        return motif[: len(sequence)]
    start = int(rng.integers(0, len(sequence) - len(motif) + 1))
    return sequence[:start] + motif + sequence[start + len(motif) :]


def fetch_interval_sequence(
    fasta_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    strand: str = "+",
) -> str:
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    seq = fasta[chrom][int(start) : int(end)]
    if strand == "-":
        complement = str.maketrans("ACGTN", "TGCAN")
        seq = seq.translate(complement)[::-1]
    return str(seq)
