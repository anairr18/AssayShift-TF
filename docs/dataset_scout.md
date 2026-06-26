# AssayShift-TF Dataset Scout

Goal: a small, source-grounded MVP benchmark for TF binding prediction under protocol, lab, species, and TF-family shift. This scout intentionally keeps the first manifest narrow enough for a 2-page MLCB abstract while preserving clear expansion paths.

## Recommended MVP Panel

Use 10 TFs across common DNA-binding families:

- `CTCF`, `REST`: zinc-finger / architectural-repressor axis.
- `MYC`, `MAX`: bHLH-ZIP, with CUT&RUN and ENCODE K562 ChIP-seq coverage.
- `GATA1`, `GATA2`: GATA family, hematopoietic K562 biology.
- `JUND`: AP-1 family.
- `SPI1`: ETS family.
- `FOXA1`: forkhead pioneer factor.
- `ESR1`: nuclear receptor, useful as a treatment/ligand confounder check.

Primary ChIP-seq source: ENCODE released TF ChIP-seq experiments with processed IDR peak BEDs and matched control experiment accessions. The initial manifest uses GRCh38 human peaks and one mm10 mouse CTCF row. ENCODE experiment/file pages are the authoritative source for accession, lab, biosample, assembly, peak file, and control metadata.

Protocol-shift source: GEO `GSE84474`, Skene and Henikoff CUT&RUN. The series is public, includes human K562 CUT&RUN for CTCF, MAX, and MYC, and provides processed supplementary BED/BEDGRAPH files. The paper reports CTCF CUT&RUN in K562 and MYC/MAX CUT&RUN as transcription factor examples. Source pages:

- GEO series: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84474
- CTCF CUT&RUN sample: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2433139
- MAX CUT&RUN sample: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2433145
- MYC CUT&RUN sample: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2433146
- No-antibody CUT&RUN control: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2433147
- Paper: https://elifesciences.org/articles/21856

Optional next-source checks, not in the first manifest: Kaya-Okur et al. CUT&Tag (`GSE124690`) is relevant because the paper profiles TFs including CTCF, but this pass did not verify a clean per-sample CTCF processed accession quickly enough to include it. ReMap and Cistrome are useful secondary harmonized peak catalogs for expansion or external validation, but the first MVP should use primary ENCODE/GEO accessions as labels to avoid provenance ambiguity:

- ReMap 2022: https://academic.oup.com/nar/article/50/D1/D316/6423925
- Cistrome Data Browser: https://cistrome.org/db/

## Metadata To Capture

The manifest includes the minimum columns requested: `dataset_id`, `source`, `tf`, `species`, `assembly`, `assay`, `lab`, `biosample`, `cell_type`, `processed_peak_url`, `control_url`, `notes`, `citation_url`.

For implementation, add a richer normalized metadata table before training:

- target family, target organism, antibody, treatment/timepoint, perturbation, disease state.
- replicate IDs, biological vs technical replicate grouping, read layout, read depth, library complexity.
- file accession, output type, peak caller, threshold type, genome build, blacklist version, liftover status.
- control type: input, IgG, no-antibody, native input, spike-in.
- ENCODE audit flags, release date/status, dbxrefs, DOI/GEO/SRA/BioProject accessions.

## Split Feasibility

Protocol shift is feasible for `CTCF`, `MYC`, and `MAX`: train on ENCODE K562 TF ChIP-seq and evaluate on GSE84474 K562 CUT&RUN. Caveat: ENCODE peak files in this manifest are GRCh38, while GSE84474 processed human files are hg19 fragment BED/BEDGRAPH. The MVP should either lift CUT&RUN intervals to GRCh38 or reprocess raw reads uniformly before comparing labels.

Lab shift is feasible inside K562 ChIP-seq. ENCODE has same-TF K562 experiments from multiple labs for CTCF, MYC, MAX, GATA1, GATA2, REST, and JUND. The small manifest includes a mixed-lab set, and a fuller lab-shift split can add second-lab rows for the same TF without changing sources.

Species shift is feasible but confounded. The first mouse row is CTCF ChIP-seq in CH12.LX on mm10, paired conceptually with human K562 CTCF. This gives an orthologous-TF species-shift stress test, but it also shifts cell lineage, assembly, and antibody context. A stronger version would add mouse MEL/G1E/G1E-ER4 or CH12.LX rows and compare against lineage-matched human hematopoietic lines where possible.

TF-family shift is feasible as leave-family-out evaluation if the training set is expanded modestly. The current 10-TF panel covers enough families to demonstrate the protocol-aware framing, but family-shift claims should be described as preliminary unless each family has multiple TFs and cell contexts.

## Negatives And Controls

Use matched controls where available, not random genomic background alone. ENCODE rows point to input/control ChIP-seq experiment pages; CUT&RUN rows point to a no-antibody BED control from the same series. For sequence-model negatives, sample GC- and mappability-matched nonpeak windows within accessible/mappable genome after removing blacklist regions, promoters if needed, and all positives across folds. Keep chromosome holdout separate from shift holdout to avoid leakage through nearby sequence.

Important confounders to model or stratify:

- cell identity and expression state dominate TF occupancy; K562-heavy training can overstate generalization.
- protocol and peak-calling differ: ENCODE IDR narrowPeak versus CUT&RUN fragment BED/BEDGRAPH.
- assembly mismatch: GRCh38/mm10 ENCODE versus hg19 GSE84474.
- antibody, lot, salt/stringency, digestion time, size selection, and spike-in normalization alter CUT&RUN signal.
- lab can be correlated with TF, assay vintage, peak caller, read length, and control type.
- ESR1 may be ligand/treatment-sensitive; keep treatment metadata before using it in a clean TF-family split.
- blacklist, copy-number, mappability, and repetitive-region artifacts can masquerade as assay shift.

## Licensing And Access

ENCODE says external users may freely download, analyze, and publish results based on released ENCODE data without restriction, while citing the ENCODE Consortium/DCC and dataset accessions: https://www.encodeproject.org/help/citing-encode/

GEO pages used here are public and provide accession-level citations/downloads. Cite both the GEO series/sample accessions and the associated publication when using GSE84474: https://www.ncbi.nlm.nih.gov/geo/info/linking.html

No controlled-access or human-subject restricted datasets are included in this manifest. The dataset is still research-only until the pipeline records per-file licenses, citations, and redistribution rules for any bundled derived labels.

## MVP Recommendation

For the abstract, define three evaluation tracks:

1. Protocol shift: ENCODE K562 ChIP-seq to GSE84474 K562 CUT&RUN for `CTCF`, `MYC`, `MAX`.
2. Lab/batch shift: ENCODE K562 same-TF train/test split across labs, starting with CTCF and adding rows after the MVP manifest.
3. Species/family shift: human K562 multi-family training, mouse CH12.LX CTCF and held-out TF-family tests as stress tests with explicit confounder caveats.

The smallest defensible benchmark should not claim disentangled causality among protocol, lab, species, and TF family. It should frame the contribution as a protocol-aware benchmark with metadata-rich shift axes and explicit confounder controls.
