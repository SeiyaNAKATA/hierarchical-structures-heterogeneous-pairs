# Study 1 data and analysis

This directory contains the data and analysis files for Study 1 of the manuscript **“Emergence of hierarchical structure through repeated dyadic interaction across developmental compositions.”** Study 1 examined whether hierarchical organization emerges through repeated reciprocal learning in adult–adult dyads over extended interaction histories.

## Directory structure

```text
Study1/
├── README.md
├── data/
│   ├── raw/
│   │   └── all_apps_wide-*.csv
│   └── processed/
│       ├── participants.csv
│       ├── trials.csv
│       ├── sequence_metrics.csv
│       └── dyad_similarity.csv
└── scripts/
    ├── 01_build_study1_data.py
    └── 02_analyze_study1.Rmd
```

The `data/raw/` directory contains the completed oTree exports used to reconstruct the Study 1 dataset. The `data/processed/` directory contains the tidy datasets used for the reported analyses. The preprocessing script in `scripts/` reconstructs all four processed datasets directly from the raw oTree exports.

## Participants and design

Study 1 included 46 adults forming 23 stable adult–adult dyads. Each dyad completed 15 interaction blocks, and each block consisted of eight reproduction trials. At the beginning of each block, the two participants were presented with independently generated random seed sequences. From the second trial onward, each participant learned the sequence reproduced by their partner on the preceding trial.

All sequences consisted of 13 items drawn from four colors: red (`R`), green (`G`), blue (`B`), and yellow (`Y`).

## Documented ID correction

One completed session was recorded with incorrect participant and dyad IDs during data collection. According to the experiment log, raw participant IDs `1290` and `1291` are treated as participant IDs `1200` and `1201`, respectively, and raw dyad ID `29` is treated as dyad ID `20`.

This correction is applied programmatically by `01_build_study1_data.py`; the original raw file is retained unchanged.

## Processed datasets

### `participants.csv`

Contains one row per participant. It includes:

- de-identified public participant and dyad IDs,
- member number within the dyad,
- age and gender,
- source-file information, and
- raw participant and dyad identifiers used to trace the processed record back to the corresponding oTree export.

### `trials.csv`

Contains one row per participant per actual reproduction trial. The principal variables are:

- interaction block,
- trial number within block,
- global trial number,
- presented sequence,
- reproduced sequence, and
- `logged_accuracy`, the accuracy value recorded by the experimental program.

The analysis measure of reproduction accuracy is independently recomputed from the presented and reproduced sequences during preprocessing.

### `sequence_metrics.csv`

Contains the sequence-level variables used in the reported analyses.

Trials 1–8 correspond to participants’ reproduced outputs. For analyses of sequence structure, the independently generated seed presented at the beginning of each block is additionally represented as `trial_in_block = 0` and `sequence_type = seed`.

Reproduction accuracy is undefined (`NA`) for Trial 0 because Trial 0 represents the presented seed sequence itself rather than a reproduced output. Reproduction accuracy is defined as one minus the Levenshtein edit distance between the presented and reproduced sequences, normalized by sequence length.

Hierarchical depth is calculated using the SEQUITUR-style procedure used in the original analysis. Compression ratio is calculated using the same compression-based definition as in the original analysis, implemented without generating temporary text or gzip files.

### `dyad_similarity.csv`

Contains the similarity between the sequences associated with the two members of a dyad at each block and trial.

Dyad similarity is calculated as one minus the normalized Levenshtein distance between the two sequences. Trial 0 compares the independently generated seed sequences presented to the two dyad members at the beginning of each block; Trials 1–8 compare their reproduced sequences.

## Reconstructing the processed data

The processed datasets can be recreated from the raw oTree exports using Python 3 with the `pandas` and `numpy` packages installed.

From the `Study1/` directory, run:

```bash
python3 scripts/01_build_study1_data.py \
    --input-dir data/raw \
    --output-dir data/processed
```

The script performs validation before writing any output. It expects:

- 23 completed dyads,
- 46 participants,
- 120 reproduction trials per participant,
- sequences of length 13 containing only `R`, `G`, `B`, and `Y`, and
- agreement between `logged_accuracy` and accuracy recomputed from normalized Levenshtein distance.

If these checks fail, the script terminates without generating the processed datasets.

A successful run produces:

```text
participants.csv:        46 rows
trials.csv:            5520 rows
sequence_metrics.csv:  6210 rows
dyad_similarity.csv:   3105 rows
```

## Reproducing the statistical analyses and Figure 3

Open `scripts/02_analyze_study1.Rmd` in RStudio and knit it to HTML, or render it from the `Study1/` directory with:

```r
rmarkdown::render("scripts/02_analyze_study1.Rmd")
```

The report:

- validates the processed datasets,
- reconstructs the descriptive statistics used in Figure 3,
- reproduces the four Figure 3 panels, and
- fits the linear mixed-effects models for reproduction accuracy, hierarchical depth, compression ratio, and dyad similarity.

Reproduction accuracy is analyzed using actual reproduction outputs only (Trials 1–8). Trial 0 represents the independently generated seed sequence presented at the beginning of each block and therefore has no reproduction-accuracy value of its own. Hierarchical depth, compression ratio, and dyad similarity are defined for the seed sequences and are analyzed over Trials 0–8.

## Data minimization

Only variables required to reconstruct the analyses reported in the manuscript are retained in the processed datasets. oTree participant codes, session codes, timestamps, page-time information, and variables unrelated to the reported analyses are omitted from the processed files.
