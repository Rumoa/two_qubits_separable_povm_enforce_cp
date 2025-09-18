#!/usr/bin/env python3

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from bidict import bidict
from scipy.stats import zscore


def main():
    parser = argparse.ArgumentParser(description="Process quantum counts datasets.")
    parser.add_argument("--input_file", type=str, help="Path to input parquet file")
    parser.add_argument(
        "--output_file", type=str, help="Path to output .pkl file (joblib)"
    )
    parser.add_argument(
        "--bitflip", action="store_true", help="Apply bitflip to counts"
    )
    parser.add_argument(
        "--filtering", action="store_true", help="Apply rolling-zscore filtering"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_dict/config_dict_states_measurements.json",
        help="Path to config JSON file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = json.load(f)

    list_initial_states_str = config_dict["states"]
    list_measurements_basis_str = config_dict["measurements"]
    outcomes = ["00", "01", "10", "11"]

    state_map = bidict({s: x for s, x in enumerate(list_initial_states_str)})
    basis_map = bidict({s: x for s, x in enumerate(list_measurements_basis_str)})
    outcome_map = bidict({s: x for s, x in enumerate(outcomes)})

    # Load datasets
    df = pd.read_parquet(args.input_file)

    # Bitflip if requested
    if args.bitflip:
        df["bitstring"] = df["bitstring"].astype(str).str.strip().str[::-1]

    meta_cols = [c for c in df.columns if c not in ("bitstring", "count")]

    # Pivot wide
    df_wide = df.pivot_table(
        index=meta_cols, columns="bitstring", values="count", fill_value=0
    ).reset_index()

    df_wide.columns.name = None
    df_wide = df_wide.rename_axis(None, axis=1)
    df = df_wide.sort_values(by="delay_time", ascending=True)

    df.datetime = pd.to_datetime(df.datetime, format="mixed")
    df = df.drop(columns=["exp_id", "run_id", "q0", "q1", "datetime"], errors="ignore")

    for bit in outcomes:
        df[bit] = df[bit].astype(int)

    df["state_id"] = df["initial_state"].map({v: k for k, v in state_map.items()})
    df["basis_id"] = df["measurement_basis"].map({v: k for k, v in basis_map.items()})

    # Filtering
    if args.filtering:
        window_size = 5
        z_threshold = 2
        use_median = True

        df_filtered_list = []
        for (sid, bid), group in df.groupby(["state_id", "basis_id"]):
            group = group.sort_values("delay_time").copy()
            for col in outcomes:
                rolling = group[col].rolling(window_size, center=True)
                trend = rolling.median() if use_median else rolling.mean()
                residuals = group[col] - trend
                z_scores = zscore(residuals.fillna(0))
                outliers = np.abs(z_scores) > z_threshold
                group.loc[outliers, col] = trend[outliers]
            df_filtered_list.append(group)
        df = pd.concat(df_filtered_list, ignore_index=True)

    # counts_array
    df["counts_array"] = df[outcomes].values.tolist()
    df_grouped = (
        df.groupby(["state_id", "basis_id", "delay_time"])
        .agg({"counts_array": "first", "shots": "first"})
        .reset_index()
    )
    df_grouped["counts_array"] = df_grouped["counts_array"].apply(np.array)
    df = df_grouped

    # SPAM counts
    mask = df["delay_time"] == 0
    df_spam = df.loc[mask].sort_values(by=["state_id", "basis_id"])
    counts_2d = np.vstack(df_spam["counts_array"].values)
    n_states = len(state_map.keys())
    n_basis = len(basis_map.keys())
    spam_counts = counts_2d.reshape(n_states, n_basis, 4)

    # Training datasets
    mask = df["delay_time"] != 0
    df_training = df.loc[mask]

    X = np.column_stack(
        [
            df_training["state_id"].to_numpy(dtype=np.int64),
            df_training["basis_id"].to_numpy(dtype=np.int64),
            df_training["delay_time"].to_numpy(dtype=np.float64),
        ]
    )
    Y = np.stack(df_training["counts_array"].to_numpy())

    # Save all using joblib
    data_to_save = {
        "spam_counts": spam_counts,
        "X": X,
        "Y": Y,
        "state_map": state_map,  # bidict preserved
        "basis_map": basis_map,
        "outcome_map": outcome_map,
    }
    joblib.dump(data_to_save, args.output_file)

    print(f"Data saved to {args.output_file}")
    print(
        "X shape:",
        X.shape,
        "Y shape:",
        Y.shape,
        "spam_counts shape:",
        spam_counts.shape,
    )


if __name__ == "__main__":
    main()
