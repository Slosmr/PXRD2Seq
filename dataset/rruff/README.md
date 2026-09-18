# RRUFF construction

Use Python 3.11 or later:

```bash
python -m pip install -r dataset/rruff/requirements.txt
python dataset/rruff/code/build_all.py --help
```

Obtain the raw experimental files from [RRUFF](https://www.rruff.net/) and place
them in this layout, preserving their sample filenames:

```text
dataset/rruff/
  raw_data/
    XY_Processed/    Experimental powder XY text files
    DIF/             Associated DIF metadata text files
  code/
```

Run all stages from the repository root:

```bash
python dataset/rruff/code/build_all.py
```

Alternatively, keep inputs and outputs outside the repository. The chosen
working directory must already contain `raw_data/XY_Processed` and `raw_data/DIF`:

```bash
python dataset/rruff/code/build_all.py --root /path/to/rruff_work
```

The script writes `datasets/` under the chosen root. Paths inside the generated
CSVs are relative to their corresponding output directories. A regression CSV
shares the classification directory's signal and report files.

## Stages and outputs

| Stage | Output | Study count |
| --- | --- | ---: |
| `base` | `datasets/base/rruff_labeled.csv` | 932 |
| `base` | `datasets/base/rruff_adapt.csv` | 552 |
| `classification` | `datasets/classification/rruff_classification.csv` | 730 |
| `regression` | `datasets/classification/rruff_regression.csv` | 294 |

Stages can be invoked with `--stage base`, `--stage classification`, or
`--stage regression`, in that order. Statistics and rejection reports are
generated alongside the CSVs. The complete pipeline writes
`datasets/build_manifest.json`.

The base cohort is built from paired XY/DIF records with symmetry and lattice
agreement checks. Classification intersects that cohort with stricter
classification candidates. Regression requires reconstructable DIF atom
tables and agreement between DIF and simulated peak positions.

## Signal representation

Base/adaptation signals use 8,500 samples over 4-90 degrees 2theta. Auxiliary
classification preparation signals, also referenced by the regression-proxy CSV,
use 8,500 samples over 5-20 degrees after conversion to a 20 keV equivalent.
These are distinct physical grids despite equal array length.

The study's model inputs use the base 4-90-degree signals selected by the strict
cohort identifiers. Final model-input assembly and evaluation manifests belong
to the later model release. The auxiliary 20 keV arrays are not drop-in inputs
for a model expecting the base grid. Keep angular metadata with each array.

The regression reference is a pymatgen-based proxy, not a GSAS-II-validated
dataset. The 706 complete-label evaluation records and the 278-record prefix
intersection are evaluation subsets of the preparation outputs; final evaluation
manifests will accompany the later model release.

See the [dataset card](../../docs/DATASETS.md) for validation thresholds and
reproducibility limitations. Raw files, generated arrays, reference-paper PDFs,
and historical build logs are not included in this release.
