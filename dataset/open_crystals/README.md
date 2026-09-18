# MP/COD/AMCSD construction

Use Python 3.11 or later. 

```bash
python -m pip install -r dataset/open_crystals/requirements.txt
python dataset/open_crystals/build_dataset.py --config dataset/open_crystals/config.example.json --dry-run
```

## Actual build

1. Copy `config.example.json` to `config.json` in this directory.
2. Set `paths.cod_cif_root` and `paths.amcsd_cif_root` to the downloaded CIF
   directories. Relative paths are resolved against the configuration file.
3. Set `MP_API_KEY` in your process environment. Do not store it in the JSON
   configuration or pass it as a command-line argument.
4. Run from the repository root:

```bash
python dataset/open_crystals/build_dataset.py
```

The default entry point reads `config.json` next to the script. Use `--config`
to select a different configuration. Local `config.json` and build outputs are
excluded by `.gitignore`.

## Stages

| Stage | Operation |
| --- | --- |
| `sources` | Retrieve MP records; process COD CIFs; build ideal peaks and source-level splits |
| `amcsd` | Build AMCSD artifacts without RRUFF or internal deduplication |
| `plan` | Exclude ICSD; filter and approximately deduplicate MP/COD/AMCSD |
| `materialize` | Write final artifacts and preserve the surviving MP/COD split assignments |

Use the integrated entry point for the study recipe. Individual helper scripts
also expose optional recipes; their standalone defaults are not a replacement
for the integrated configuration.

By default, outputs go under `build/`, with the final dataset in
`build/dataset/`. `--from-stage` and `--to-stage` allow continuation using
existing intermediate outputs. For example:

```bash
python dataset/open_crystals/build_dataset.py --from-stage plan
```

`--reset-output` removes the final output's `base_peaks`, `labels`, and
`structures` directories before materialization. Use it only for an output
directory dedicated to this build. A normal build does not require this flag.

`paths.rruff_test_csv` is optional and empty by default. If supplied, it is used
only to report overlap candidates; it does not remove records. Keep it empty
when rebuilding the documented recipe without an additional overlap audit.

## Output interpretation

This pipeline writes ideal peak tables and structure labels. It does not export
the model's online-augmented 8,500-point signals. See the [dataset card](../../docs/DATASETS.md)
for counts, label representation, filtering, and reproducibility boundaries.
Final CSV artifact paths are relative to the reconstruction workspace
(`paths.build_root`) when the output is inside that workspace. For example,
`dataset/base_peaks/<id>.npz` is resolved beneath `build/`. An output outside
the workspace can produce absolute paths; keep generated CSVs out of this code release.
MP is a changing online data source, so a fresh build is not guaranteed to
reproduce the historical material IDs or sample counts.
