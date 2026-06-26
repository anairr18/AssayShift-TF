from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from assayshift_tf.features import add_sequence_stats, insert_motif, random_dna


@dataclass(frozen=True)
class DemoConfig:
    n_examples: int = 6000
    sequence_length: int = 180
    random_seed: int = 13


TF_METADATA = [
    ("CTCF", "zinc_finger", "CCGCGNGGNGGCAG"),
    ("REST", "zinc_finger", "TCAGCACCACGGACAG"),
    ("GATA1", "gata", "AGATAA"),
    ("MAX", "bhlh", "CACGTG"),
    ("JUN", "bzip", "TGACTCA"),
    ("SPI1", "ets", "GGAA"),
    ("FOXA1", "forkhead", "TGTTTAC"),
    ("ESR1", "nuclear_receptor", "AGGTCANNNTGACCT"),
]


def _clean_motif(motif: str) -> str:
    return motif.replace("N", "A")


def make_demo_dataset(config: DemoConfig = DemoConfig()) -> pd.DataFrame:
    """Create a deterministic stress-test dataset with biological and protocol artifacts.

    The demo intentionally gives ChIP-seq positives a GC artifact and reverses that
    artifact in CUT&RUN/CUT&Tag. This makes IID performance look strong while
    held-out protocol performance exposes shortcut learning and overconfidence.
    """
    rng = np.random.default_rng(config.random_seed)
    assays = np.asarray(["TF ChIP-seq", "CUT&RUN", "CUT&Tag"])
    labs = {
        "TF ChIP-seq": np.asarray(["Michael Snyder, Stanford", "John Stamatoyannopoulos, UW"]),
        "CUT&RUN": np.asarray(["Henikoff"]),
        "CUT&Tag": np.asarray(["EpiCypher"]),
    }
    species = np.asarray(["Homo sapiens", "Mus musculus"])

    rows: list[dict[str, object]] = []
    for i in range(config.n_examples):
        tf, family, motif = TF_METADATA[int(rng.integers(0, len(TF_METADATA)))]
        assay = str(rng.choice(assays, p=[0.62, 0.23, 0.15]))
        lab = str(rng.choice(labs[assay]))
        organism = str(rng.choice(species, p=[0.78, 0.22]))

        label = int(rng.random() < 0.18)
        motif_prob = 0.86 if label else 0.08
        has_motif = bool(rng.random() < motif_prob)

        if assay == "TF ChIP-seq":
            gc_mean = 0.66 if label else 0.42
        elif assay == "CUT&RUN":
            gc_mean = 0.43 if label else 0.61
        else:
            gc_mean = 0.48 if label else 0.57
        if lab == "John Stamatoyannopoulos, UW":
            gc_mean += 0.03
        if organism == "Mus musculus":
            gc_mean -= 0.04
        gc = float(np.clip(rng.normal(gc_mean, 0.035), 0.25, 0.78))

        seq = random_dna(config.sequence_length, gc, rng)
        if has_motif:
            seq = insert_motif(seq, _clean_motif(motif), rng)
        rows.append(
            {
                "example_id": f"demo_{i:06d}",
                "dataset_id": f"DEMO_{assay}_{lab}".replace(" ", "_").replace(",", ""),
                "sequence": seq,
                "label": label,
                "tf": tf,
                "tf_family": family,
                "assay": assay,
                "lab": lab,
                "species": organism,
                "assembly": "GRCh38" if organism == "Homo sapiens" else "mm10",
                "has_planted_motif": has_motif,
            }
        )

    return add_sequence_stats(pd.DataFrame(rows))
