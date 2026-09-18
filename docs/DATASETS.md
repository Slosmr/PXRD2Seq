# Dataset card

## Scope

This release contains dataset construction code and aggregate descriptions.
It does not redistribute raw source databases, processed intensity arrays,
model weights, or the exact evaluation manifests. Counts below describe the
study snapshot. A fresh build can differ when source databases or parsers change.

## Open-crystal collection

| Split | MP | COD | AMCSD | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 91,047 | 69,356 | 16,199 | 176,602 |
| Validation | 11,281 | 8,681 | 0 | 19,962 |
| Test | 11,439 | 8,599 | 0 | 20,038 |
| Total | 113,767 | 86,636 | 16,199 | 216,602 |

The MP/COD source builder uses a split stratified by source and crystal system
with random seed 42. Reconstruction retains the original assignments for
surviving MP/COD records and assigns surviving AMCSD records to training.
ICSD is excluded by the integrated build command.

The builder calculates ideal Cu K-alpha diffraction peaks over 2theta = 4-90
degrees. The model pipeline uses 8,500-point intensity arrays on that interval;
the current structure builder stores ideal peak positions and intensities,
not the online-augmented training patterns.

### Filtering and deduplication

- MP filtering includes structural quality, space-group stability over specified
  tolerances, and reduced-cell consistency. Reconstruction excludes records with
  energy above hull greater than 0.20 eV/atom. Records in the 0.05-0.10 and
  0.10-0.20 eV/atom bands receive weights of 0.5 and 0.1, respectively.
- COD processing screens structural quality and removes organic/molecular
  candidates using the included composition heuristic. Reconstruction excludes
  pressure/high-temperature records and flags selected synthetic, thermal,
  growth, or low-temperature records for downweighting.
- AMCSD is first processed without internal or RRUFF deduplication. The joint
  reconstruction then applies approximate deduplication across MP, COD, and AMCSD.
- The approximate key combines normalized formula, space-group number, and
  `floor(volume_per_atom / 0.05)`. Keeper priority is AMCSD, COD, then MP. This
  is a heuristic and is not a proof of structural uniqueness.
- The integrated pipeline does not remove records based on RRUFF membership.
  An optional RRUFF metadata file produces an overlap-candidate audit only.
  No claim of complete structure-level disjointness from RRUFF is made here.

In the study snapshot, the combined source manifest contained 262,803 records;
hard filtering excluded 29,602 and approximate deduplication excluded 16,599,
leaving 216,602 records.

### Generated files

| File | Meaning |
| --- | --- |
| `base_peaks/<material_id>.npz` | Ideal `mus`, `intensities`, and `d_hkls` arrays |
| `labels/<material_id>.json` | Symmetry labels and lattice parameters |
| `structures/<material_id>.cif` | Standardized, Niggli-reduced structure |
| `metadata.csv` | Combined material metadata |
| `train.csv`, `val.csv`, `test.csv` | Material-level splits with artifact paths |

In final CSVs, relative artifact paths are resolved against `paths.build_root`,
not necessarily against the CSV's own directory.

Symmetry labels are determined from the input structure. Lattice labels and
stored structures use Niggli reduction. Categorical symmetry and reduced-cell
angles should therefore not be interpreted as conventional-cell coordinates
without accounting for the cell representation.

## Experimental RRUFF collection

The pipeline pairs experimental XY patterns with DIF crystallographic metadata.
It generates a labeled base set and a separate unlabeled adaptation set, then
constructs stricter classification and lattice-regression cohorts.

| Cohort | Study count | Meaning |
| --- | ---: | --- |
| Labeled base | 932 | Valid paired records after base filtering |
| Unlabeled adaptation | 552 | Separate permissive preparation output; not a reported evaluation cohort |
| Strict classification | 730 | Intersection of base records and classification candidates |
| Complete-label evaluation | 706 | Records with all four symmetry labels used in reported classification evaluation |
| Lattice regression proxy | 294 | DIF-derived structures passing lattice and peak-position checks |
| Prefix-intervention lattice subset | 278 | Intersection of the complete-label and regression-proxy cohorts |

The construction pipeline directly produces the 932/552/730/294 preparation
outputs. The 706/278 evaluation subsets and their final model-prediction
intersections belong to the later evaluation release.

### Signal grids

There are two distinct representations in the construction code:

| Output | Angular representation | Samples |
| --- | --- | ---: |
| Base signals used for the study's model input; adaptation signals | Experimental Cu K-alpha grid, 2theta = 4-90 degrees | 8,500 |
| Auxiliary strict-classification preparation outputs, also referenced by the regression-proxy CSV | Converted to a 20 keV equivalent by Bragg's law, 2theta = 5-20 degrees | 8,500 |

The strict grid uses wavelength 0.6199209921660013 angstrom. Its CSV records
`source_wavelength_ang`, `target_energy_kev`, `target_wavelength_ang`,
`target_two_theta_min`, `target_two_theta_max`, and `signal_length`.
These auxiliary arrays are used in the preparation workflow. The study's model
inputs use the base 4-90-degree signals selected by the strict cohort identifiers.
Final model-input assembly and evaluation manifests are outside this preliminary
release. Do not feed the auxiliary 20 keV arrays directly to a model expecting
the base grid. Equal array length does not imply equal physical coordinates.

### Regression reference and validation

The regression proxy reconstructs structures from DIF atom tables, performs
Niggli reduction, and compares simulated peak positions with DIF peak tables.
At least 7 of the top 10 DIF peaks must match, using a d-spacing relative
tolerance of 0.5% or a 2theta tolerance of 0.2 degrees. The output records the
validation method, thresholds, and reconstructed reference cell.

This is a pymatgen-based proxy cohort, not a GSAS-II-validated subset.
Missing or ambiguous DIF metadata
can prevent a sample from entering a cohort.

## Reproducibility boundaries

The public scripts reproduce the preparation logic and output formats when
supplied with suitable inputs. Exact study membership additionally requires
the original source snapshots, identifiers, split manifests, and evaluation
selection, including assembly of RRUFF model inputs from the base signals.
Online augmentation, model predictions, compatibility metrics, and
analysis experiments are outside this preliminary release.
