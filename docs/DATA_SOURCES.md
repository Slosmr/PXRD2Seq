# Data sources

| Source | Access | Role | Recorded acquisition date |
| --- | --- | --- | --- |
| Materials Project | [Database](https://materialsproject.org/) and [API guide](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started) | Crystal structures and energy-above-hull metadata | 2026-05-02 |
| Crystallography Open Database | [COD](https://www.crystallography.net/cod/) | Experimental crystal structures | 2026-04-30 |
| American Mineralogist Crystal Structure Database | [AMCSD](https://www.rruff.net/amcsd/) | Mineral structures used in training | 2026-05-09 |
| RRUFF | [RRUFF](https://www.rruff.net/) | Experimental PXRD patterns and DIF metadata | Not specified in the construction record |

Dates describe project acquisition records, not official database releases.
The MP date was inferred from the stored artifact creation window; it does not
identify an immutable MP database version. Exact source snapshots are not
included in this release.

## Acquiring inputs

Materials Project requires a user account and API credential. The builder reads
only the `MP_API_KEY` environment variable. Obtain the credential through the
[Materials Project dashboard](https://next-gen.materialsproject.org/dashboard).

Obtain COD and AMCSD CIFs directly from their providers. Preserve COD's nested
directory structure: for example, `3500100.cif` is stored beneath
`cif/3/50/01/3500100.cif`. The reconstruction code uses the COD identifier to
resolve original metadata.

For RRUFF, obtain experimental powder `XY_Processed` files and their associated
`DIF` records. Preserve filenames and sample identifiers, including suffixes.
The scripts accept the text formats described in `rruff_parser.py`; they are
not a web scraper and do not fetch the raw archive automatically.

## Attribution

The databases and the original structure or measurement publications remain
the sources of the data. Follow each provider's current access, attribution,
and redistribution terms. This repository does not replace those terms or
assign a new blanket license to third-party datasets. No raw database files or
copies of reference papers are bundled here.
