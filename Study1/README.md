# Study 1 data and analysis

This directory contains the data and analysis files for Study 1 of the manuscript **“Emergence of hierarchical structure through repeated dyadic interaction across developmental compositions.”** Study 1 examined whether hierarchical organization emerges through repeated reciprocal learning in adult–adult dyads over extended interaction histories.

## Directory structure

The intended directory structure is:

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
    └── 01_build_study1_data.py
    └── 02_analyze_study1.Rmd
    
```

The `data/raw/` directory contains the completed oTree exports used to reconstruct the Study 1 dataset. The `data/processed/` directory contains the tidy datasets used for the analyses reported in the manuscript. The processing script in `scripts/` reconstructs all four processed datasets directly from the raw oTree exports.

## Participants and design

Study 1 included 46 adults forming 23 stable adult–adult dyads. Each dyad completed 15 interaction blocks, and each block consisted of eight reproduction trials. In the first trial of each block, the two participants were presented with independently generated random seed sequences. From the second trial onward, each participant learned the sequence reproduced by their partner on the preceding trial. All sequences consisted of 13 items drawn from four colours: red (`R`), green (`G`), blue (`B`), and yellow (`Y`).

## Documented ID correction

One completed session was recorded with incorrect participant and pair IDs during data collection. According to the experiment log, the raw participant IDs `1290` and `1291` should be treated as participant IDs `1200` and `1201`, respectively, and raw pair ID `29` should be treated as pair ID `20`. This correction is applied programmatically by `01_build_study1_data.py`; the original raw file is not modified.

## Processed datasets

`participants.csv` contains one row per participant. It provides the de-identified public participant and dyad IDs used throughout the processed datasets, demographic variables required to describe the sample, and the raw source identifiers needed to trace each participant back to the corresponding oTree export.

`trials.csv` contains one row per participant per experimental trial. The principal variables are block number, trial number within block, global trial number, the presented sequence, and the reproduced sequence. `otree_accuracy` is the accuracy value recorded by the experimental program and is retained for validation. The analysis measure of reproduction accuracy is independently recomputed from the presented and reproduced sequences in the processing script.

`sequence_metrics.csv` contains the sequence-level variables used in the reported analyses. Trials 1–8 correspond to participants’ reproduced sequences. For structural analyses, the independently generated seed presented at the beginning of each block is additionally represented as `trial_in_block = 0` and `sequence_type = seed`. Reproduction accuracy is therefore undefined (`NA`) for trial 0. Reproduction accuracy is defined as one minus the Levenshtein edit distance between the presented and reproduced sequences, normalized by sequence length. Hierarchical depth is calculated using the SEQUITUR-style procedure used in the original analysis. Compression ratio is calculated using the same compression-based definition as in the original analysis, implemented without temporary text or gzip files.

`dyad_similarity.csv` contains the similarity between the sequences associated with the two members of a dyad at each block and trial. Similarity is calculated as one minus the normalized Levenshtein distance between the two sequences. Trial 0 compares the independently generated seed sequences presented to the two members at the beginning of each block; trials 1–8 compare their reproduced sequences.

## Reconstructing the processed data

The processed datasets can be recreated from the raw oTree exports with Python 3 and the `pandas` and `numpy` packages. From the `Study1/` directory, run:

```bash
python3 scripts/01_build_study1_data.py \
    --input-dir data/raw \
    --output-dir data/processed
```

The script performs validation before writing any output. It expects 23 completed dyads (46 participants), 120 reproduction trials per participant, sequences of length 13 containing only `R`, `G`, `B`, and `Y`, and agreement between the oTree accuracy field and accuracy recomputed from normalized Levenshtein distance. If these checks fail, the script terminates without generating the processed datasets.

## Data minimization

Only variables required to reconstruct the analyses reported in the manuscript are retained in the processed datasets. oTree participant codes, session codes, timestamps, page-time information, and variables unrelated to the reported analyses are omitted from the processed files.
