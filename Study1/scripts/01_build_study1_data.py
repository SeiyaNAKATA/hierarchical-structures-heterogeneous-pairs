#!/usr/bin/env python3
"""Build de-identified Study 1 data files from oTree all_apps_wide exports.

Inputs
------
A directory containing only the Study 1 ``all_apps_wide-*.csv`` files.

Outputs
-------
participants.csv
    One row per participant.
trials.csv
    One row per scheduled participant trial (15 blocks x 8 trials). Missing
    observations are retained explicitly as NA.
sequence_metrics.csv
    Sequence-level analysis data. Includes each reproduced sequence (trial
    1-8 within a block) and the independently generated seed sequence
    (trial 0 within each block), with reproduction accuracy, hierarchical
    depth, and compression ratio.
dyad_similarity.csv
    Similarity between the two members' sequences at each block/trial,
    including seed sequences at trial 0.

The script intentionally does not export oTree participant codes, session
codes, timestamps, or other platform metadata.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

N_BLOCKS = 15
TRIALS_PER_BLOCK = 8
N_TRIALS = N_BLOCKS * TRIALS_PER_BLOCK
EXPECTED_SEQUENCE_LENGTH = 13
EXPECTED_DYADS = 23
EXPECTED_PARTICIPANTS = EXPECTED_DYADS * 2


def levenshtein_distance(a: str, b: str) -> int:
    """Return Levenshtein edit distance with unit insertion/deletion/substitution costs."""
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


def normalized_similarity(a: str, b: str) -> float:
    """Return 1 - normalized Levenshtein distance.

    All Study 1 sequences have length 13. Using max length as the denominator
    makes the function robust while reproducing the original values for this
    data set.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan
    a = str(a)
    b = str(b)
    denominator = max(len(a), len(b))
    if denominator == 0:
        return np.nan
    return 1.0 - levenshtein_distance(a, b) / denominator


def legacy_compression_ratio(sequence: str) -> float:
    """Reproduce the compression-ratio values used in the original analysis.

    The original script wrote each sequence to a text file, compressed that
    file with gzip, and subtracted ``len(filename) + 24`` bytes before dividing
    by the uncompressed file size. For the ASCII sequences used here, that is
    exactly equivalent to taking the raw DEFLATE payload size, subtracting five
    bytes, and dividing by the sequence length.

    This implementation avoids creating thousands of temporary .txt/.gz files
    while preserving the published analysis values exactly.
    """
    raw = str(sequence).encode("ascii")
    compressor = zlib.compressobj(level=9, wbits=-15)
    payload = compressor.compress(raw) + compressor.flush()
    adjusted_compressed_size = len(payload) - 5
    return adjusted_compressed_size / len(raw)


def sequitur_hierarchy(sequence: str, rule_utility: bool = True) -> int:
    """Return maximum hierarchy depth from the SEQUITUR-style algorithm.

    This is a cleaned implementation of the algorithm used in the original
    analysis and is intended to reproduce the published hierarchy-depth values.
    """
    sequence = str(sequence)
    current_sequence = ""
    rules: dict[str, str] = {}
    symbol_number = 0

    for symbol in sequence:
        symbol_number += 1
        current_sequence += symbol
        if symbol_number < 4:
            continue

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

    if rule_utility and rules:
        digrams = list(rules.keys())
        rule_ids = list(rules.values())
        all_strings = "".join(digrams + [current_sequence])
        non_unique: dict[str, str] = {}

        for rule_id in rule_ids:
            if all_strings.count(rule_id) <= 1:
                once_key = [key for key, value in rules.items() if value == rule_id][0]
                non_unique[once_key] = rule_id
                del rules[once_key]

        non_unique = dict(reversed(list(non_unique.items())))
        for old_digram, old_rule_id in non_unique.items():
            parent_keys = [key for key in rules if old_rule_id in key]
            if not parent_keys:
                continue
            parent_key = parent_keys[0]
            parent_value = rules[parent_key]
            del rules[parent_key]
            rules[parent_key.replace(old_rule_id, old_digram)] = parent_value

    def depth(rule_id: str) -> int:
        for digram, candidate_rule_id in rules.items():
            if candidate_rule_id == rule_id:
                child_depths = [
                    depth(part) if part in rules.values() else 0 for part in digram
                ]
                return 1 + max(child_depths, default=0)
        return 0

    return max((depth(rule_id) for rule_id in rules.values()), default=0)


def public_ids(pair_id_zero_based: int, member_zero_based: int) -> tuple[str, str, int]:
    """Create stable public IDs without exposing oTree participant/session codes."""
    dyad_number = pair_id_zero_based + 1
    member_number = member_zero_based + 1
    dyad_id = f"S1_D{dyad_number:02d}"
    participant_id = f"{dyad_id}_P{member_number}"
    return participant_id, dyad_id, member_number


def extract_tidy_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(input_dir.glob("all_apps_wide-*.csv"))
    if not files:
        raise FileNotFoundError(f"No all_apps_wide-*.csv files found in {input_dir}")

    participant_rows: list[dict] = []
    trial_rows: list[dict] = []

    for source_file in files:
        wide = pd.read_csv(source_file)
        if len(wide) != 2:
            raise ValueError(
                f"{source_file.name}: expected 2 participant rows for one dyad, found {len(wide)}"
            )

        for _, row in wide.iterrows():
            pair_id_zero = int(row["control.1.player.pair_ID"])
            source_player_id = int(row["control.1.player.player_id"])
            member_zero = source_player_id % 10
            participant_id, dyad_id, member = public_ids(pair_id_zero, member_zero)

            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "dyad_id": dyad_id,
                    "member": member,
                    "age": int(row["control.1.player.age"]),
                    "gender": row["control.1.player.gender"],
                    "condition": "Adult-Adult",
                }
            )

            for global_trial in range(1, N_TRIALS + 1):
                prefix = f"control.{global_trial}.player"
                presented = row.get(f"{prefix}.Learn_seq", np.nan)
                reproduced = row.get(f"{prefix}.sequence", np.nan)
                logged_accuracy = row.get(f"{prefix}.accuracy", np.nan)

                block = (global_trial - 1) // TRIALS_PER_BLOCK + 1
                trial_in_block = (global_trial - 1) % TRIALS_PER_BLOCK + 1

                trial_rows.append(
                    {
                        "participant_id": participant_id,
                        "dyad_id": dyad_id,
                        "member": member,
                        "block": block,
                        "trial_in_block": trial_in_block,
                        "global_trial": global_trial,
                        "presented_sequence": presented,
                        "reproduced_sequence": reproduced,
                        "logged_accuracy": logged_accuracy,
                    }
                )

    participants = (
        pd.DataFrame(participant_rows)
        .drop_duplicates()
        .sort_values(["dyad_id", "member"])
        .reset_index(drop=True)
    )
    trials = (
        pd.DataFrame(trial_rows)
        .sort_values(["dyad_id", "member", "global_trial"])
        .reset_index(drop=True)
    )
    return participants, trials


def validate_tidy_data(
    participants: pd.DataFrame, trials: pd.DataFrame, strict: bool = True
) -> list[str]:
    warnings: list[str] = []

    if len(participants) != EXPECTED_PARTICIPANTS:
        warnings.append(
            f"Expected {EXPECTED_PARTICIPANTS} participants, found {len(participants)}."
        )

    dyad_sizes = participants.groupby("dyad_id").size()
    bad_dyads = dyad_sizes[dyad_sizes != 2]
    if not bad_dyads.empty:
        warnings.append(f"Dyads without exactly two participants: {bad_dyads.to_dict()}")

    for sequence_column in ["presented_sequence", "reproduced_sequence"]:
        observed = trials[sequence_column].dropna().astype(str)
        bad_alphabet = observed[~observed.str.fullmatch(r"[RGBY]+")]
        if len(bad_alphabet):
            warnings.append(
                f"{sequence_column}: {len(bad_alphabet)} sequence(s) contain unexpected symbols."
            )
        bad_length = observed[observed.str.len() != EXPECTED_SEQUENCE_LENGTH]
        if len(bad_length):
            warnings.append(
                f"{sequence_column}: {len(bad_length)} sequence(s) are not length {EXPECTED_SEQUENCE_LENGTH}."
            )

    # Check the accuracy logged by oTree against the manuscript definition.
    complete_trials = trials.dropna(
        subset=["presented_sequence", "reproduced_sequence", "logged_accuracy"]
    ).copy()
    complete_trials["recomputed_accuracy"] = [
        normalized_similarity(a, b)
        for a, b in zip(
            complete_trials["presented_sequence"], complete_trials["reproduced_sequence"]
        )
    ]
    mismatch = ~np.isclose(
        complete_trials["logged_accuracy"].astype(float),
        complete_trials["recomputed_accuracy"].astype(float),
    )
    if mismatch.any():
        warnings.append(
            f"Logged accuracy disagrees with recomputed Levenshtein accuracy in {int(mismatch.sum())} trial(s)."
        )

    counts = (
        trials.groupby(["participant_id", "dyad_id"], as_index=False)
        .agg(
            presented_n=("presented_sequence", "count"),
            reproduced_n=("reproduced_sequence", "count"),
        )
    )
    incomplete = counts[
        (counts["presented_n"] != N_TRIALS) | (counts["reproduced_n"] != N_TRIALS)
    ]
    if not incomplete.empty:
        detail = "; ".join(
            f"{r.participant_id}: presented={r.presented_n}, reproduced={r.reproduced_n}"
            for r in incomplete.itertuples()
        )
        warnings.append(f"Incomplete participant data: {detail}")

    if strict and warnings:
        raise ValueError("\n".join(warnings))
    return warnings


def build_sequence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    # Participant outputs (trial 1-8 within each block).
    outputs = trials[trials["reproduced_sequence"].notna()].copy()
    outputs["sequence_type"] = "output"
    outputs["sequence"] = outputs["reproduced_sequence"].astype(str)
    outputs["reproduction_accuracy"] = [
        normalized_similarity(a, b)
        for a, b in zip(outputs["presented_sequence"], outputs["reproduced_sequence"])
    ]

    # Each block's seed is the sequence presented on trial 1 of that block.
    seeds = trials[
        (trials["trial_in_block"] == 1) & trials["presented_sequence"].notna()
    ].copy()
    seeds["sequence_type"] = "seed"
    seeds["sequence"] = seeds["presented_sequence"].astype(str)
    seeds["trial_in_block"] = 0
    seeds["global_trial"] = pd.NA
    seeds["reproduction_accuracy"] = np.nan

    metrics = pd.concat([seeds, outputs], ignore_index=True)
    metrics["hierarchical_depth"] = metrics["sequence"].map(sequitur_hierarchy)
    metrics["compression_ratio"] = metrics["sequence"].map(legacy_compression_ratio)

    keep = [
        "participant_id",
        "dyad_id",
        "member",
        "block",
        "trial_in_block",
        "global_trial",
        "sequence_type",
        "sequence",
        "reproduction_accuracy",
        "hierarchical_depth",
        "compression_ratio",
    ]
    return (
        metrics[keep]
        .sort_values(["dyad_id", "member", "block", "trial_in_block"])
        .reset_index(drop=True)
    )


def build_dyad_similarity(sequence_metrics: pd.DataFrame) -> pd.DataFrame:
    left = sequence_metrics[sequence_metrics["member"] == 1].copy()
    right = sequence_metrics[sequence_metrics["member"] == 2].copy()

    join_keys = ["dyad_id", "block", "trial_in_block", "sequence_type"]
    paired = left.merge(
        right,
        on=join_keys,
        how="inner",
        suffixes=("_member1", "_member2"),
        validate="one_to_one",
    )
    paired["dyad_similarity"] = [
        normalized_similarity(a, b)
        for a, b in zip(paired["sequence_member1"], paired["sequence_member2"])
    ]

    paired["global_trial"] = np.where(
        paired["trial_in_block"] == 0,
        pd.NA,
        (paired["block"] - 1) * TRIALS_PER_BLOCK + paired["trial_in_block"],
    )

    keep = [
        "dyad_id",
        "block",
        "trial_in_block",
        "global_trial",
        "sequence_type",
        "sequence_member1",
        "sequence_member2",
        "dyad_similarity",
    ]
    return paired[keep].sort_values(["dyad_id", "block", "trial_in_block"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write available data even if expected observations are missing.",
    )
    args = parser.parse_args()

    participants, trials = extract_tidy_data(args.input_dir)

    try:
        warnings = validate_tidy_data(
            participants, trials, strict=not args.allow_incomplete
        )
    except ValueError as exc:
        print("DATA VALIDATION FAILED", file=sys.stderr)
        print(exc, file=sys.stderr)
        print(
            "\nNo output files were written. Re-run with --allow-incomplete only for diagnostics.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(args.output_dir / "participants.csv", index=False)
    trials.to_csv(args.output_dir / "trials.csv", index=False)

    sequence_metrics = build_sequence_metrics(trials)
    dyad_similarity = build_dyad_similarity(sequence_metrics)
    sequence_metrics.to_csv(args.output_dir / "sequence_metrics.csv", index=False)
    dyad_similarity.to_csv(args.output_dir / "dyad_similarity.csv", index=False)

    print(f"Wrote {len(participants):,} rows to participants.csv")
    print(f"Wrote {len(trials):,} rows to trials.csv")
    print(f"Wrote {len(sequence_metrics):,} rows to sequence_metrics.csv")
    print(f"Wrote {len(dyad_similarity):,} rows to dyad_similarity.csv")

    if warnings:
        print("\nWARNINGS")
        for warning in warnings:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
