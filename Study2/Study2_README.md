# Study 2 data and analysis

This directory contains the data-processing and analysis files for Study 2 of the manuscript **“Emergence of hierarchical structure through repeated dyadic interaction across developmental compositions.”** Study 2 compared adult–adult, child–child, and adult–child dyads in a nonlinguistic color-sequence learning task.

## Directory structure

```text
Study2/
├── README.md
├── data/
│   ├── raw/
│   │   ├── demographic.csv
│   │   └── dataframes/
│   │       ├── 3000.csv
│   │       ├── displayed_3000.csv
│   │       ├── ...
│   │       ├── 3701.csv
│   │       └── displayed_3701.csv
│   └── processed/
│       ├── participants.csv
│       ├── trials.csv
│       ├── sequence_metrics.csv
│       └── dyad_similarity.csv
└── scripts/
    ├── 01_build_study2_data.py
    └── 02_analyze_study2.Rmd
```

## Raw data

The primary Study 2 raw data are the participant-level CSV files in `data/raw/dataframes/`. These files were written automatically by the experimental program during data collection.

For every participant:

- one file records the sequence reproduced by the participant (for example, `3000.csv`), and
- one file records the sequence presented to that participant on the corresponding trials (for example, `displayed_3000.csv`).

There are 142 reproduced-sequence files and 142 displayed-sequence files, corresponding to 142 participants in 71 dyads.

The oTree `all_apps_wide` files are not required to reconstruct the Study 2 analyses. They were manually downloaded as backup exports after experimental sessions and are incomplete for some sessions because the manual save step was occasionally omitted. Some older exports also correspond to sessions excluded after experimental problems. The automatically saved participant-level files in `dataframes/` therefore serve as the source data for the public Study 2 reconstruction.

`demographic.csv` links the experimental participant ID to demographic information. The file currently contains `age`, `gender`, `ID`, and `date`. The preprocessing script uses `ID`, `age`, and `gender`; `date` is retained as session-tracking metadata and is not used in the reported analyses.

## Duplicate raw records

Three participant logs contain one exactly duplicated trial record:

- participant `3431`, Trial 4,
- participant `3591`, Trial 1, and
- participant `3620`, Trial 1.

The corresponding `displayed_*.csv` files contain the same duplicated records. These are duplicate rows rather than additional experimental trials.

The preprocessing script does not modify the raw files. When a trial number appears more than once, the script verifies that the repeated rows are exactly identical and retains one copy. If a repeated trial contains non-identical rows, preprocessing stops and requires manual inspection.

## Study design

Study 2 included 142 participants forming 71 stable dyads:

- 19 adult–adult dyads,
- 26 child–child dyads, and
- 26 adult–child dyads.

Each dyad completed two interaction blocks of eight reproduction trials. Each sequence contained 13 items drawn from four colors (`R`, `G`, `B`, and `Y`).

At the beginning of each interaction block, the two members of a dyad received independently generated random seed sequences. From the second trial onward, each participant was presented with the sequence reproduced by their partner on the preceding trial. The preprocessing script verifies this reciprocal transmission relation for every dyad before writing the processed datasets.

The automatically saved raw files contain a `condition` field whose value is constant in this Study 2 implementation and therefore does not encode developmental composition. Developmental group is reconstructed from age (`Adult` for age 18 years or older; `Child` otherwise), and dyad composition is then derived from the developmental groups of the two dyad members.

## Processed datasets

### `participants.csv`

Contains one row per participant. Public participant and dyad IDs are generated as `S2_D01_P1`, `S2_D01_P2`, and so forth.

The original experimental participant and dyad numbers are retained in `source_participant_id` and `source_pair_id` so that the transformation from raw to processed data remains traceable. The file also contains age, gender, developmental group, and dyad composition.

### `trials.csv`

Contains one row per participant per actual reproduction trial. It includes:

- interaction block,
- trial number within block,
- global trial number,
- presented sequence,
- reproduced sequence,
- `logged_accuracy`, the accuracy value stored by the task, and
- `reproduction_accuracy`, independently recomputed from the presented and reproduced sequences using normalized Levenshtein distance.

### `sequence_metrics.csv`

Contains the measures used in the reported sequence-level analyses.

The 2,272 output rows correspond to actual participant reproductions. In addition, the independently generated seed presented at the beginning of each block is represented for each participant as `trial_in_block = 0` and `sequence_type = seed`, producing 2,556 rows in total.

Reproduction accuracy is undefined for seed rows because Trial 0 represents the presented seed sequence itself rather than a reproduced output. Hierarchical depth is calculated using the same SEQUITUR-style procedure as in the original analysis. Compression ratio reproduces the original compression-based measure without generating temporary compression files.

### `dyad_similarity.csv`

Contains one row per dyad, interaction block, and trial position.

Dyad similarity is defined as one minus normalized Levenshtein distance between the sequences associated with the two dyad members. Trial 0 compares their independently generated seed sequences, while Trials 1–8 compare their reproduced sequences.

## Reconstructing the processed data

The Python script requires Python 3 with `pandas` and `numpy` installed.

From the `Study2/` directory, run:

```bash
python3 scripts/01_build_study2_data.py \
  --input-dir data/raw/dataframes \
  --demographics data/raw/demographic.csv \
  --output-dir data/processed
```

The script performs the following checks before writing any processed data:

- all 142 participant IDs and corresponding displayed-sequence files are present,
- every participant contains 16 unique trials after removal of exact duplicate records,
- all sequences contain exactly 13 valid color symbols,
- stored task accuracy agrees with accuracy recomputed from the two sequences,
- the reconstructed sample contains 71 dyads with the expected developmental compositions, and
- the reciprocal transmission relation between dyad members holds on Trials 2–8 of each interaction block.

A successful run produces:

```text
participants.csv:       142 rows
trials.csv:            2272 rows
sequence_metrics.csv:  2556 rows
dyad_similarity.csv:   1278 rows
```

With the current raw files, the script reports removal of one exact duplicate from each of:

```text
3431.csv
displayed_3431.csv
3591.csv
displayed_3591.csv
3620.csv
displayed_3620.csv
```

## Reproducing the statistical analyses and Figure 4

The R Markdown report requires `tidyverse`, `lmerTest`, `performance`, `patchwork`, and `knitr`.

Open `scripts/02_analyze_study2.Rmd` in RStudio and knit it to HTML, or render it from the `Study2/` directory with:

```r
rmarkdown::render("scripts/02_analyze_study2.Rmd")
```

The report:

- validates the four processed datasets,
- calculates the descriptive statistics used in Figure 4,
- reconstructs the four Figure 4 panels, and
- fits linear mixed-effects models for reproduction accuracy, hierarchical depth, compression ratio, and dyad similarity.

For each dependent variable, a model containing trial and developmental composition is compared with a model additionally containing their interaction. Adult–adult dyads are the reference condition, and dyad is included as a random intercept.

Because the candidate models differ in their fixed-effects structure, AIC comparison is performed using maximum-likelihood fits (`REML = FALSE`). The better-supported model is then refitted using restricted maximum likelihood (`REML = TRUE`) for coefficient estimation and Satterthwaite tests.

Reproduction accuracy is analyzed using actual reproduction outputs only (Trials 1–8). Trial 0 represents the independently generated seed sequence presented at the beginning of each interaction block and therefore has no reproduction-accuracy value of its own. Hierarchical depth, compression ratio, and dyad similarity are defined for the seed sequences and are analyzed over Trials 0–8.

Panels (a)–(c) of Figure 4 display Trials 1–8, while panel (d) additionally displays Trial 0 and the mean similarity between random seed sequences as a dashed reference line.

When `save_outputs` is enabled, the report writes Figure 4 as PNG and PDF files and saves descriptive statistics, model comparisons, supported-model coefficients, full model summaries, and R session information in `results/`.

## Relation to the original analysis files

The preprocessing code was checked against the previously generated Study 2 analysis datasets. Using the current raw participant-level files after removal of the exact duplicated rows:

- all 2,556 sequence strings were reproduced exactly,
- hierarchical-depth values matched exactly,
- compression-ratio values matched exactly,
- all 1,278 dyad-similarity values matched exactly, and
- recomputed reproduction accuracy matched the stored task values up to the decimal precision with which accuracy was written to the raw CSV files.

The statistical report uses the corrected treatment of reproduction accuracy in which Trial 0 is excluded from the accuracy model.
