# misato-pose-retention

How far do correctly bound ligands move in a short molecular dynamics simulation? This
repository calibrates the common 2 Å ligand-RMSD "pose retained" criterion against all
16,971 trajectories of the MISATO dataset, and audits the dataset's physical consistency.

Accompanies the manuscript *How far do ligands move? A reference distribution for pose
retention in molecular dynamics* (H. Tiwari, under review).

## Main results

| | |
|---|---|
| Complexes passing physical QC | 15,129 of 16,971 (10.9% fail) |
| Median ligand RMSD after 10 ns | 2.07 Å (95% CI 2.05–2.10) |
| Known binders rejected by a 2 Å cut-off | 52.5% |
| Of those rejected, still buried (≥80% of initial buried surface) | 86% |
| Complexes changing verdict between 5 and 10 ns | 12.9% |
| 90th percentile, most to least enclosed pocket quartile | 3.65 to 5.88 Å |

**The reference tables to use** are in `results/Supplementary_Data_2_reference_tables.csv`.
Compute your complex's enclosure (initial buried surface area divided by ligand heavy-atom
count), find its quartile, and report your ligand's displacement as a percentile of that
stratum, not against a fixed cut-off.

## Reproduce every number in the paper

```
pip install -r requirements.txt
python reproduce_analysis.py            # about 1.5 min
python reproduce_analysis.py --models   # adds the quantile models, about 5 min
```

This uses the extracted traces shipped in this repository (`misato_full_traces.npz`,
`ligand_counts_full.csv`) and writes `numbers.json` and `per_complex.csv`. The descriptor
analysis also needs MISATO's `QM.hdf5` (343 MB) from
[Zenodo 7711953](https://doi.org/10.5281/zenodo.7711953), placed next to the script.

## Re-extract from the MISATO archive

Both scripts stream byte ranges from Zenodo, so the 133 GB `MD.hdf5` is never downloaded
in full. Both are resumable; re-run the same command after any interruption.

| Script | Output | Time |
|---|---|---|
| `run_full_extraction.py` | `misato_full_traces.npz`: per-frame ligand RMSD, buried SASA, distance, interaction energy | about 13 h |
| `run_topology_pass.py` | `ligand_counts.csv`: ligand atom and heavy-atom counts | about 25 h |

Both are limited by Zenodo's rate limit for anonymous clients, roughly one request per
second, not by local compute.

## Quality control

`retention_qc.classify(rmsd_trace, bsasa_trace)` assigns each complex to one of
`ANALYSED`, `UNBOUND`, `DISPLACED_AT_START`, `IMAGING_SUSPECT` or `NON_PHYSICAL`. The
1,842 complexes that fail are listed in `results/Supplementary_Data_3_QC_exclusions.csv`
so that users of MISATO v1.0.0 can filter them.

## Contents

| Path | |
|---|---|
| `run_full_extraction.py`, `run_topology_pass.py` | extraction from the MISATO archive |
| `retention_qc.py` | physical quality-control classifier |
| `reproduce_analysis.py` | all statistics in the paper |
| `misato_full_traces.npz`, `ligand_counts_full.csv`, `protein_groups.csv` | extracted data (derived from MISATO) |
| `results/` | supplementary data tables |
| `figures/` | manuscript figures |

## Licences

Code: MIT (see `LICENSE`). Data files (`*.npz`, `*.csv`, `results/`): CC BY 4.0. They
are derived from the MISATO dataset (Siebenmorgen et al., *Nature Computational Science*
4, 367–378 (2024); Zenodo doi:10.5281/zenodo.7711953, CC BY 4.0), which must also be cited
when they are used.
