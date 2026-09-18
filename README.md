# PXRD2Seq

**Crystallography-Guided Autoregressive Modeling for Compatible Joint Prediction from Powder X-Ray Diffraction**

PXRD2Seq jointly predicts crystal symmetry and lattice parameters from a powder
X-ray diffraction (PXRD) pattern. It organizes four categorical symmetry fields
and six numerical lattice parameters into a crystallography-guided autoregressive
sequence. Earlier symmetry predictions provide context for subsequent symmetry
and lattice predictions, allowing output compatibility to be learned from data.

```text
PXRD pattern -> signal encoder -> causal decoder
  -> crystal system -> Laue class -> Bravais lattice -> space group
  -> a -> b -> c -> alpha -> beta -> gamma
```

## Release status

This is a **preliminary data-construction release prepared during manuscript
submission**. It provides the project overview, dataset documentation, and
construction scripts for the open-crystal and experimental RRUFF datasets.

| Resource | Current availability |
| --- | --- |
| Project overview and dataset descriptions | Included |
| MP/COD/AMCSD construction scripts and configuration template | Included |
| RRUFF parsing, cohort construction, and regression-proxy scripts | Included |
| Original database snapshots and processed data arrays | Not included |
| Exact evaluation cohort manifests | Planned for a later release |
| PXRD2Seq model, training, and inference code | Planned after acceptance or publication |
| Baselines, analysis experiments, and trained checkpoints | Planned after acceptance or publication |

The current release is not a complete reproduction package for the paper's
experiments. In particular, the structure builder exports ideal peak tables;
online augmentation and model evaluation belong to the later model release.

## Datasets

The simulated-data collection is constructed from Materials Project (MP), COD,
and AMCSD. AMCSD contributes training structures only. Experimental RRUFF data
provide the evaluation cohorts described below.

| Collection | Subset | Samples |
| --- | --- | ---: |
| MP + COD + AMCSD | Training | 176,602 |
| MP + COD | Validation | 19,962 |
| MP + COD | Test | 20,038 |
| RRUFF | Preprocessed classification cohort | 730 |
| RRUFF | Complete-label classification evaluation cohort | 706 |
| RRUFF | Lattice regression proxy | 294 |
| RRUFF | Intersection used for lattice prefix interventions | 278 |

These counts describe the study snapshot, not guaranteed counts from a future
database download. The 730/706/294/278 RRUFF counts refer to different, overlapping
cohorts and must not be added together. See the [dataset card](docs/DATASETS.md)
for signal grids, label conventions, filtering, and reproducibility boundaries.

## Getting started

Use Python 3.11 or later. Instructions and pinned direct dependencies are provided
separately for the two pipelines:

- [Open-crystal construction: MP, COD, and AMCSD](dataset/open_crystals/README.md)
- [RRUFF construction: experimental patterns and DIF metadata](dataset/rruff/README.md)

Preview the open-crystal pipeline without credentials, source data, or downloads:

```bash
python dataset/open_crystals/build_dataset.py --config dataset/open_crystals/config.example.json --dry-run
```

For an actual build, obtain the source data from their providers and configure
the input paths. Materials Project access uses the `MP_API_KEY` environment
variable. Never put a credential in source code or committed configuration files.

## Repository layout

```text
dataset/
  open_crystals/        MP/COD/AMCSD construction pipeline
  rruff/               RRUFF parsing and dataset construction
docs/
  DATASETS.md          Dataset card and reproducibility boundaries
  DATA_SOURCES.md      Source databases and acquisition notes
  RELEASE_PLAN.md      Current and planned release contents
CITATION.md            Manuscript title and project link
THIRD_PARTY_NOTICES.md Data and software acknowledgements
```

## Citation and acknowledgements

The manuscript title and project link are provided in [CITATION.md](CITATION.md). This snapshot
does not claim an accepted venue or a publication DOI. Please also acknowledge
the underlying [data sources](docs/DATA_SOURCES.md) when using derived datasets.
See [third-party notices](THIRD_PARTY_NOTICES.md) for dependency and data attribution.
