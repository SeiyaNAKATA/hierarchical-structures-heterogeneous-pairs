#!/usr/bin/env python3
"""Build the public Study 2 datasets from automatically saved raw CSV files.

Inputs
------
The input directory contains one reproduced-sequence file and one displayed-
sequence file for each participant, for example::

    3000.csv
    displayed_3000.csv
    ...
    3701.csv
    displayed_3701.csv

A separate ``demographic.csv`` contains the columns ``age``, ``gender``,
and ``ID``. Additional columns are ignored.

Outputs
-------
The script writes four analysis-ready files::

    participants.csv
    trials.csv
    sequence_metrics.csv
    dyad_similarity.csv

The original raw files are never modified. Three known duplicate raw records
(3431 trial 4, 3591 trial 1, and 3620 trial 1) are handled generically: a
repeated trial is removed only when the repeated rows are exactly identical.
Any non-identical duplicate causes the script to stop.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


N_DYADS = 71
N_PARTICIPANTS = 142
N_BLOCKS = 2
TRIALS_PER_BLOCK = 8
N_TRIALS = N_BLOCKS * TRIALS_PER_BLOCK
SEQUENCE_LENGTH = 13
COLOURS = {"R", "G", "B", "Y"}
SEQUENCE_COLUMNS = [f"c{i}" for i in range(1, SEQUENCE_LENGTH + 1)]

RAW_REQUIRED_COLUMNS = SEQUENCE_COLUMNS + [
    "sequence_ID",
    "trial",
    "individual_ID",
    "pair_ID",
    "condition",
    "accuracy",
]


# ---------------------------------------------------------------------------
# Basic sequence measures
# ---------------------------------------------------------------------------

def levenshtein_distance(a: str, b: str) -> int:
    """Return Levenshtein distance with unit insertion/deletion/substitution."""
    a = str(a)
    b = str(b)
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current

    return previous[-1]


def reproduction_accuracy(presented: str, reproduced: str) -> float:
    """Return 1 - normalized Levenshtein distance.

    This reproduces the definition used in the original analysis. Because all
    Study 2 sequences have length 13, normalization by the presented-sequence
    length and by the reproduced-sequence length are equivalent here.
    """
    presented = str(presented)
    reproduced = str(reproduced)
    return 1.0 - levenshtein_distance(presented, reproduced) / len(presented)


def sequence_similarity(a: str, b: str) -> float:
    """Return the similarity measure used for within-dyad similarity."""
    a = str(a)
    b = str(b)
    return 1.0 - levenshtein_distance(a, b) / len(a)


def compression_ratio(sequence: str) -> float:
    """Reproduce the compression-ratio measure used in the legacy script.

    The original analysis wrote each 13-character sequence to a text file,
    gzip-compressed it, and subtracted the gzip/file-name overhead before
    dividing compressed size by the original 13-byte size. For these ASCII
    strings, the resulting adjusted compressed size is equivalent to the raw
    DEFLATE payload size minus five bytes. This implementation therefore
    reproduces the legacy values without creating thousands of temporary files.
    """
    raw = str(sequence).encode("ascii")
    compressor = zlib.compressobj(level=9, wbits=-15)
    payload = compressor.compress(raw) + compressor.flush()
    adjusted_compressed_size = len(payload) - 5
    return adjusted_compressed_size / len(raw)


def sequitur_hierarchy(sequence: str, rule_utility: bool = True) -> int:
    """Return maximum hierarchy depth using the original SEQUITUR procedure."""
    sequence = str(sequence)
    current_sequence = ""
    symbol_number = 0
    rules: dict[str, str] = {}

    for symbol in sequence:
        symbol_number += 1
        current_sequence += symbol

        if symbol_number >= 4:
            while True:
                digram = current_sequence[-2:]
                if digram in current_sequence[:-2]:
                    new_rule = str(len(rules))
                    current_sequence = current_sequence.replace(digram, new_rule)
                    rules[digram] = new_rule
                elif digram in rules:
                    current_sequence = current_sequence.replace(digram, rules[digram])
                else:
                    break

    if rule_utility:
        digrams = list(rules.keys())
        rule_ids = list(rules.values())
        digrams.append(current_sequence)
        all_strings = "".join(digrams)
        non_unique: dict[str, str] = {}

        for rule_id in rule_ids:
            if all_strings.count(rule_id) <= 1:
                once_key = [
                    key for key, value in rules.items()
                    if value == rule_id
                ][0]
                non_unique[once_key] = rule_id
                del rules[once_key]

        non_unique = dict(reversed(list(non_unique.items())))
        if non_unique:
            for old_digram, old_rule_id in non_unique.items():
                parent_keys = [
                    key for key in rules
                    if old_rule_id in key
                ]
                if not parent_keys:
                    # This is not expected for the Study 2 sequences, but a
                    # clear failure is preferable to silently altering the
                    # legacy algorithm.
                    raise ValueError(
                        "SEQUITUR rule-utility step could not locate parent "
                        f"rule for {old_rule_id!r} in sequence {sequence!r}."
                    )
                parent_key = parent_keys[0]
                parent_value = rules[parent_key]
                del rules[parent_key]
                rules[parent_key.replace(old_rule_id, old_digram)] = parent_value

    def compute_depth(rule_id: str) -> int:
        for digram, candidate_rule_id in rules.items():
            if candidate_rule_id == rule_id:
                return 1 + max(
                    (
                        compute_depth(part)
                        if part in rules.values()
                        else 0
                        for part in digram
                    ),
                    default=0,
                )
        return 0

    return max(
        (compute_depth(rule_id) for rule_id in rules.values()),
        default=0,
    )


# ---------------------------------------------------------------------------
# Raw-data loading and validation
# ---------------------------------------------------------------------------

def expected_source_ids() -> list[int]:
    """Return 3000, 3001, 3010, 3011, ..., 3700, 3701."""
    return [
        3000 + pair_id * 10 + member
        for pair_id in range(N_DYADS)
        for member in (0, 1)
    ]


def sequence_from_columns(data: pd.DataFrame) -> pd.Series:
    return data[SEQUENCE_COLUMNS].astype(str).agg("".join, axis=1)


def validate_sequence(sequence: str, context: str) -> None:
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(
            f"{context}: expected sequence length {SEQUENCE_LENGTH}, "
            f"found {len(sequence)} ({sequence!r})."
        )
    if not set(sequence).issubset(COLOURS):
        raise ValueError(
            f"{context}: unexpected symbol(s) in sequence {sequence!r}."
        )


def clean_participant_file(
    path: Path,
    source_participant_id: int,
    displayed: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    """Read one raw file and remove only exact repeated-trial records."""
    data = pd.read_csv(path)
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(
            f"{path.name}: missing required column(s): {', '.join(missing)}"
        )

    duplicate_log: list[dict] = []
    cleaned_rows = []

    for trial, group in data.groupby("trial", sort=True, dropna=False):
        if len(group) > 1:
            unique_group = group.drop_duplicates()
            if len(unique_group) != 1:
                raise ValueError(
                    f"{path.name}: trial {trial} occurs {len(group)} times "
                    "with non-identical rows. Manual inspection is required."
                )
            duplicate_log.append({
                "source_participant_id": source_participant_id,
                "file": path.name,
                "trial": int(trial),
                "duplicate_rows_removed": len(group) - 1,
                "file_type": "displayed" if displayed else "reproduced",
            })
        cleaned_rows.append(group.iloc[0])

    cleaned = pd.DataFrame(cleaned_rows).sort_values("trial").reset_index(drop=True)

    expected_trials = list(range(1, N_TRIALS + 1))
    if cleaned["trial"].astype(int).tolist() != expected_trials:
        raise ValueError(
            f"{path.name}: expected trials 1-{N_TRIALS} after duplicate "
            f"removal, found {cleaned['trial'].tolist()}."
        )

    source_pair_id = (source_participant_id - 3000) // 10
    source_member = (source_participant_id - 3000) % 10

    if source_member not in (0, 1):
        raise ValueError(
            f"{path.name}: source participant ID {source_participant_id} "
            "does not encode member 0 or 1."
        )

    if set(cleaned["pair_ID"].astype(int)) != {source_pair_id}:
        raise ValueError(
            f"{path.name}: pair_ID does not match filename-derived pair "
            f"ID {source_pair_id}."
        )
    if set(cleaned["individual_ID"].astype(int)) != {source_member}:
        raise ValueError(
            f"{path.name}: individual_ID does not match filename-derived "
            f"member {source_member}."
        )

    expected_blocks = [1] * TRIALS_PER_BLOCK + [2] * TRIALS_PER_BLOCK
    if cleaned["sequence_ID"].astype(int).tolist() != expected_blocks:
        raise ValueError(
            f"{path.name}: sequence_ID does not encode the expected two "
            "8-trial interaction blocks."
        )

    cleaned["sequence"] = sequence_from_columns(cleaned)
    for row in cleaned.itertuples():
        validate_sequence(
            row.sequence,
            f"{path.name}, trial {int(row.trial)}",
        )

    return cleaned, duplicate_log


def load_demographics(path: Path) -> pd.DataFrame:
    demographics = pd.read_csv(path)
    required = ["age", "gender", "ID"]
    missing = [column for column in required if column not in demographics]
    if missing:
        raise ValueError(
            f"{path.name}: missing required column(s): {', '.join(missing)}"
        )

    demographics = demographics[required].copy()
    demographics["ID"] = demographics["ID"].astype(int)

    if demographics["ID"].duplicated().any():
        duplicated = demographics.loc[
            demographics["ID"].duplicated(keep=False), "ID"
        ].tolist()
        raise ValueError(
            "demographic.csv contains duplicate participant IDs: "
            + ", ".join(map(str, duplicated))
        )

    expected = set(expected_source_ids())
    observed = set(demographics["ID"])
    if observed != expected:
        missing_ids = sorted(expected - observed)
        extra_ids = sorted(observed - expected)
        raise ValueError(
            "demographic.csv participant IDs do not match Study 2. "
            f"Missing={missing_ids}; extra={extra_ids}."
        )

    return demographics


def developmental_group(age: int) -> str:
    return "Adult" if int(age) >= 18 else "Child"


def condition_from_groups(groups: list[str]) -> str:
    sorted_groups = sorted(groups)
    if sorted_groups == ["Adult", "Adult"]:
        return "Adult-Adult"
    if sorted_groups == ["Child", "Child"]:
        return "Child-Child"
    if sorted_groups == ["Adult", "Child"]:
        return "Adult-Child"
    raise ValueError(f"Unexpected developmental groups: {groups}")


def public_ids(source_pair_id: int, source_member: int) -> tuple[str, str, int]:
    dyad_id = f"S2_D{source_pair_id + 1:02d}"
    member = source_member + 1
    participant_id = f"{dyad_id}_P{member}"
    return participant_id, dyad_id, member


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_participants_and_trials(
    input_dir: Path,
    demographics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demographics = load_demographics(demographics_path)

    numeric_files = {
        int(match.group(1)): path
        for path in input_dir.glob("*.csv")
        if (match := re.fullmatch(r"(\d+)\.csv", path.name))
    }
    displayed_files = {
        int(match.group(1)): path
        for path in input_dir.glob("displayed_*.csv")
        if (match := re.fullmatch(r"displayed_(\d+)\.csv", path.name))
    }

    expected_ids = set(expected_source_ids())
    if set(numeric_files) != expected_ids:
        raise ValueError(
            "Reproduced-sequence raw files are incomplete or contain extras. "
            f"Missing={sorted(expected_ids - set(numeric_files))}; "
            f"extra={sorted(set(numeric_files) - expected_ids)}."
        )
    if set(displayed_files) != expected_ids:
        raise ValueError(
            "Displayed-sequence raw files are incomplete or contain extras. "
            f"Missing={sorted(expected_ids - set(displayed_files))}; "
            f"extra={sorted(set(displayed_files) - expected_ids)}."
        )

    participant_rows: list[dict] = []
    trial_rows: list[dict] = []
    duplicate_logs: list[dict] = []

    demographics_by_id = demographics.set_index("ID")

    for source_participant_id in expected_source_ids():
        reproduced, duplicate_log = clean_participant_file(
            numeric_files[source_participant_id],
            source_participant_id,
            displayed=False,
        )
        duplicate_logs.extend(duplicate_log)

        displayed, duplicate_log = clean_participant_file(
            displayed_files[source_participant_id],
            source_participant_id,
            displayed=True,
        )
        duplicate_logs.extend(duplicate_log)

        # After duplicate removal, the two automatically saved streams must
        # describe exactly the same trial metadata.
        metadata_columns = [
            "sequence_ID", "trial", "individual_ID",
            "pair_ID", "condition", "accuracy",
        ]
        if not reproduced[metadata_columns].equals(displayed[metadata_columns]):
            raise ValueError(
                f"Participant {source_participant_id}: reproduced and "
                "displayed raw files disagree in trial metadata."
            )

        source_pair_id = int(reproduced["pair_ID"].iloc[0])
        source_member = int(reproduced["individual_ID"].iloc[0])
        participant_id, dyad_id, member = public_ids(
            source_pair_id,
            source_member,
        )

        demographic = demographics_by_id.loc[source_participant_id]
        age = int(demographic["age"])
        gender = str(demographic["gender"])
        group = developmental_group(age)

        participant_rows.append({
            "participant_id": participant_id,
            "dyad_id": dyad_id,
            "member": member,
            "source_participant_id": source_participant_id,
            "source_pair_id": source_pair_id,
            "age": age,
            "gender": gender,
            "developmental_group": group,
        })

        for row_index in range(N_TRIALS):
            rep_row = reproduced.iloc[row_index]
            disp_row = displayed.iloc[row_index]
            global_trial = int(rep_row["trial"])
            block = int(rep_row["sequence_ID"])
            trial_in_block = (global_trial - 1) % TRIALS_PER_BLOCK + 1
            presented_sequence = str(disp_row["sequence"])
            reproduced_sequence = str(rep_row["sequence"])
            logged_accuracy = float(rep_row["accuracy"])
            recomputed_accuracy = reproduction_accuracy(
                presented_sequence,
                reproduced_sequence,
            )

            if not np.isclose(
                logged_accuracy,
                recomputed_accuracy,
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Participant {source_participant_id}, trial "
                    f"{global_trial}: logged accuracy {logged_accuracy} "
                    f"does not match recomputed accuracy "
                    f"{recomputed_accuracy}."
                )

            trial_rows.append({
                "participant_id": participant_id,
                "dyad_id": dyad_id,
                "member": member,
                "source_participant_id": source_participant_id,
                "source_pair_id": source_pair_id,
                "block": block,
                "trial_in_block": trial_in_block,
                "global_trial": global_trial,
                "presented_sequence": presented_sequence,
                "reproduced_sequence": reproduced_sequence,
                "logged_accuracy": logged_accuracy,
                "reproduction_accuracy": recomputed_accuracy,
            })

    participants = pd.DataFrame(participant_rows).sort_values(
        ["source_pair_id", "member"]
    ).reset_index(drop=True)

    # Derive developmental composition from ages rather than the raw `condition`
    # field, which is constant (=3) in these automatically saved Study 2 files.
    condition_by_dyad: dict[str, str] = {}
    for dyad_id, group in participants.groupby("dyad_id", sort=False):
        condition_by_dyad[dyad_id] = condition_from_groups(
            group.sort_values("member")["developmental_group"].tolist()
        )

    participants["condition"] = participants["dyad_id"].map(condition_by_dyad)
    participants = participants[[
        "participant_id",
        "dyad_id",
        "member",
        "source_participant_id",
        "source_pair_id",
        "age",
        "gender",
        "developmental_group",
        "condition",
    ]]

    trials = pd.DataFrame(trial_rows)
    trials["condition"] = trials["dyad_id"].map(condition_by_dyad)
    trials = trials[[
        "participant_id",
        "dyad_id",
        "member",
        "source_participant_id",
        "source_pair_id",
        "condition",
        "block",
        "trial_in_block",
        "global_trial",
        "presented_sequence",
        "reproduced_sequence",
        "logged_accuracy",
        "reproduction_accuracy",
    ]].sort_values(
        ["source_pair_id", "member", "global_trial"]
    ).reset_index(drop=True)

    duplicate_log = pd.DataFrame(duplicate_logs)
    if not duplicate_log.empty:
        duplicate_log = duplicate_log.sort_values(
            ["source_participant_id", "file_type", "trial"]
        ).reset_index(drop=True)

    return participants, trials, duplicate_log


def validate_design(participants: pd.DataFrame, trials: pd.DataFrame) -> None:
    errors: list[str] = []

    if len(participants) != N_PARTICIPANTS:
        errors.append(
            f"Expected {N_PARTICIPANTS} participants, found {len(participants)}."
        )
    if participants["dyad_id"].nunique() != N_DYADS:
        errors.append(
            f"Expected {N_DYADS} dyads, found "
            f"{participants['dyad_id'].nunique()}."
        )

    expected_condition_counts = {
        "Adult-Adult": 19,
        "Child-Child": 26,
        "Adult-Child": 26,
    }
    observed_condition_counts = (
        participants[["dyad_id", "condition"]]
        .drop_duplicates()["condition"]
        .value_counts()
        .to_dict()
    )
    if observed_condition_counts != expected_condition_counts:
        errors.append(
            "Unexpected dyad composition counts. "
            f"Observed={observed_condition_counts}; "
            f"expected={expected_condition_counts}."
        )

    if len(trials) != N_PARTICIPANTS * N_TRIALS:
        errors.append(
            f"Expected {N_PARTICIPANTS * N_TRIALS} trial rows, "
            f"found {len(trials)}."
        )

    trial_counts = trials.groupby("participant_id")["global_trial"].nunique()
    bad_counts = trial_counts[trial_counts != N_TRIALS]
    if not bad_counts.empty:
        errors.append(
            "Some participants do not have 16 unique trials: "
            + bad_counts.to_dict().__repr__()
        )

    # Validate reciprocal transmission within each block. Trial 1 of each block
    # is an independently generated seed; Trials 2-8 must display the partner's
    # output from the preceding trial.
    for dyad_id, dyad in trials.groupby("dyad_id"):
        member1 = dyad[dyad["member"] == 1].set_index("global_trial")
        member2 = dyad[dyad["member"] == 2].set_index("global_trial")
        for block_start in (1, 9):
            for global_trial in range(block_start + 1, block_start + 8):
                if (
                    member1.loc[global_trial, "presented_sequence"]
                    != member2.loc[global_trial - 1, "reproduced_sequence"]
                ):
                    errors.append(
                        f"{dyad_id}, trial {global_trial}: member 1 did not "
                        "receive member 2's preceding output."
                    )
                if (
                    member2.loc[global_trial, "presented_sequence"]
                    != member1.loc[global_trial - 1, "reproduced_sequence"]
                ):
                    errors.append(
                        f"{dyad_id}, trial {global_trial}: member 2 did not "
                        "receive member 1's preceding output."
                    )

    if errors:
        raise ValueError("\n".join(errors))


def build_sequence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    outputs = trials[[
        "participant_id",
        "dyad_id",
        "member",
        "source_participant_id",
        "source_pair_id",
        "condition",
        "block",
        "trial_in_block",
        "global_trial",
        "reproduced_sequence",
        "reproduction_accuracy",
    ]].copy()
    outputs = outputs.rename(columns={"reproduced_sequence": "sequence"})
    outputs["sequence_type"] = "output"

    seeds = trials[trials["trial_in_block"] == 1][[
        "participant_id",
        "dyad_id",
        "member",
        "source_participant_id",
        "source_pair_id",
        "condition",
        "block",
        "presented_sequence",
    ]].copy()
    seeds = seeds.rename(columns={"presented_sequence": "sequence"})
    seeds["trial_in_block"] = 0
    seeds["global_trial"] = pd.NA
    seeds["reproduction_accuracy"] = np.nan
    seeds["sequence_type"] = "seed"

    metrics = pd.concat([outputs, seeds], ignore_index=True)
    metrics["hierarchical_depth"] = metrics["sequence"].map(sequitur_hierarchy)
    metrics["compression_ratio"] = metrics["sequence"].map(compression_ratio)

    metrics = metrics[[
        "participant_id",
        "dyad_id",
        "member",
        "source_participant_id",
        "source_pair_id",
        "condition",
        "block",
        "trial_in_block",
        "global_trial",
        "sequence_type",
        "sequence",
        "reproduction_accuracy",
        "hierarchical_depth",
        "compression_ratio",
    ]]

    return metrics.sort_values(
        ["source_pair_id", "member", "block", "trial_in_block"]
    ).reset_index(drop=True)


def build_dyad_similarity(sequence_metrics: pd.DataFrame) -> pd.DataFrame:
    member1 = sequence_metrics[sequence_metrics["member"] == 1].copy()
    member2 = sequence_metrics[sequence_metrics["member"] == 2].copy()

    keys = [
        "dyad_id",
        "source_pair_id",
        "condition",
        "block",
        "trial_in_block",
        "sequence_type",
    ]
    paired = member1.merge(
        member2,
        on=keys,
        how="inner",
        suffixes=("_member1", "_member2"),
        validate="one_to_one",
    )

    paired["dyad_similarity"] = [
        sequence_similarity(a, b)
        for a, b in zip(
            paired["sequence_member1"],
            paired["sequence_member2"],
        )
    ]

    paired["global_trial"] = pd.array(
        [
            pd.NA
            if trial == 0
            else (int(block) - 1) * TRIALS_PER_BLOCK + int(trial)
            for block, trial in zip(
                paired["block"], paired["trial_in_block"]
            )
        ],
        dtype="Int64",
    )

    result = paired[[
        "dyad_id",
        "source_pair_id",
        "condition",
        "block",
        "trial_in_block",
        "global_trial",
        "sequence_type",
        "sequence_member1",
        "sequence_member2",
        "dyad_similarity",
    ]].copy()

    return result.sort_values(
        ["source_pair_id", "block", "trial_in_block"]
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Study 2 public datasets from automatic raw logs."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing 3000.csv ... and displayed_3000.csv ...",
    )
    parser.add_argument(
        "--demographics",
        required=True,
        type=Path,
        help="Path to demographic.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which the four processed CSV files are written",
    )
    args = parser.parse_args()

    try:
        participants, trials, duplicate_log = build_participants_and_trials(
            args.input_dir,
            args.demographics,
        )
        validate_design(participants, trials)
        sequence_metrics = build_sequence_metrics(trials)
        dyad_similarity = build_dyad_similarity(sequence_metrics)

        if len(sequence_metrics) != 2556:
            raise ValueError(
                f"Expected 2,556 sequence-metric rows, "
                f"found {len(sequence_metrics):,}."
            )
        if len(dyad_similarity) != 1278:
            raise ValueError(
                f"Expected 1,278 dyad-similarity rows, "
                f"found {len(dyad_similarity):,}."
            )

    except (FileNotFoundError, KeyError, ValueError) as exc:
        print("STUDY 2 DATA BUILD FAILED", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    participants.to_csv(args.output_dir / "participants.csv", index=False)
    trials.to_csv(args.output_dir / "trials.csv", index=False)
    sequence_metrics.to_csv(
        args.output_dir / "sequence_metrics.csv", index=False
    )
    dyad_similarity.to_csv(
        args.output_dir / "dyad_similarity.csv", index=False
    )

    print("Study 2 data build completed successfully.")
    print(f"participants.csv:      {len(participants):,} rows")
    print(f"trials.csv:            {len(trials):,} rows")
    print(f"sequence_metrics.csv:  {len(sequence_metrics):,} rows")
    print(f"dyad_similarity.csv:   {len(dyad_similarity):,} rows")

    if duplicate_log.empty:
        print("Exact duplicate raw trial records removed: none")
    else:
        print("Exact duplicate raw trial records removed:")
        for row in duplicate_log.itertuples():
            print(
                f"  {row.file}: trial {row.trial} "
                f"({row.file_type}; removed {row.duplicate_rows_removed})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
