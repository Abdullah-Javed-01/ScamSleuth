import hashlib
import json

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    SPLIT_DIR,
    ARTIFACT_DIR,
    RANDOM_SEED,
)


def calculate_file_hash(file_path):
    """
    Calculate a platform-independent SHA-256 hash.

    Text is read using universal newline handling so
    Windows CRLF and Unix LF line endings produce the
    same fingerprint.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline=None,
    ) as file:
        normalized_text = file.read()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def validate_clusters(df):
    """Validate that template clusters are suitable for group splitting."""

    label_counts = (
        df.groupby("template_cluster")["label"]
        .nunique()
    )

    type_counts = (
        df.groupby("template_cluster")["text_type"]
        .nunique()
    )

    if (label_counts > 1).any():
        raise ValueError(
            "At least one template cluster contains multiple labels."
        )

    if (type_counts > 1).any():
        raise ValueError(
            "At least one template cluster contains multiple text types."
        )


def create_cluster_table(df):
    """Create one row per template cluster."""

    cluster_table = (
        df.groupby("template_cluster")
        .agg(
            label=("label", "first"),
            text_type=("text_type", "first"),
            cluster_size=("id", "size"),
        )
        .reset_index()
    )

    return cluster_table


def assign_clusters(cluster_table):
    """
    Split clusters within each text_type + label group.

    Each group contains 20 clusters:
        14 -> train
         3 -> validation
         3 -> test
    """

    rng = np.random.default_rng(RANDOM_SEED)

    train_clusters = []
    validation_clusters = []
    test_clusters = []

    grouped = cluster_table.groupby(
        ["text_type", "label"],
        sort=True
    )

    for (text_type, label), group in grouped:

        clusters = group["template_cluster"].to_numpy().copy()

        rng.shuffle(clusters)

        if len(clusters) != 20:
            raise ValueError(
                f"{text_type} / {label} has "
                f"{len(clusters)} clusters instead of 20."
            )

        train_clusters.extend(clusters[:14])
        validation_clusters.extend(clusters[14:17])
        test_clusters.extend(clusters[17:20])

    return (
        set(train_clusters),
        set(validation_clusters),
        set(test_clusters),
    )


def create_split(df, clusters):
    """Select rows belonging to the supplied cluster set."""

    return (
        df[df["template_cluster"].isin(clusters)]
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


def validate_split_clusters(
    train_clusters,
    validation_clusters,
    test_clusters
):
    """Ensure no template cluster appears in multiple splits."""

    if train_clusters & validation_clusters:
        raise ValueError(
            "Cluster leakage between Train and Validation."
        )

    if train_clusters & test_clusters:
        raise ValueError(
            "Cluster leakage between Train and Test."
        )

    if validation_clusters & test_clusters:
        raise ValueError(
            "Cluster leakage between Validation and Test."
        )


def print_split_summary(name, split_df):
    """Print useful information about one dataset partition."""

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print("Rows:", len(split_df))
    print(
        "Clusters:",
        split_df["template_cluster"].nunique()
    )

    print("\nLabel distribution:")
    print(split_df["label"].value_counts())

    print("\nText-type distribution:")
    print(split_df["text_type"].value_counts())

    print("\nDifficulty distribution:")
    print(split_df["difficulty"].value_counts())


def main():

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load frozen dataset
    # -------------------------------------------------

    df = pd.read_csv(RAW_DATA_PATH)

    print("Frozen dataset loaded:", df.shape)

    # -------------------------------------------------
    # Validate template structure
    # -------------------------------------------------

    validate_clusters(df)

    cluster_table = create_cluster_table(df)

    print(
        "Unique template clusters:",
        len(cluster_table)
    )

    # -------------------------------------------------
    # Assign whole clusters to partitions
    # -------------------------------------------------

    (
        train_clusters,
        validation_clusters,
        test_clusters,
    ) = assign_clusters(cluster_table)

    validate_split_clusters(
        train_clusters,
        validation_clusters,
        test_clusters,
    )

    # -------------------------------------------------
    # Create row-level datasets
    # -------------------------------------------------

    train_df = create_split(
        df,
        train_clusters
    )

    validation_df = create_split(
        df,
        validation_clusters
    )

    test_df = create_split(
        df,
        test_clusters
    )

    # -------------------------------------------------
    # Final safety checks
    # -------------------------------------------------

    assert len(train_df) == 420
    assert len(validation_df) == 90
    assert len(test_df) == 90

    assert (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        == len(df)
    )

    # -------------------------------------------------
    # Save partitions
    # -------------------------------------------------

    train_df.to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
        encoding="utf-8",
    )

    validation_df.to_csv(
        SPLIT_DIR / "validation.csv",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------
    # Save reproducibility metadata
    # -------------------------------------------------

    metadata = {
        "random_seed": RANDOM_SEED,
        "source_dataset": RAW_DATA_PATH.name,
        "source_sha256": calculate_file_hash(
            RAW_DATA_PATH
        ),
        "total_rows": len(df),
        "total_clusters": len(cluster_table),
        "train": {
            "rows": len(train_df),
            "clusters": len(train_clusters),
        },
        "validation": {
            "rows": len(validation_df),
            "clusters": len(validation_clusters),
        },
        "test": {
            "rows": len(test_df),
            "clusters": len(test_clusters),
        },
    }

    metadata_path = (
        ARTIFACT_DIR
        / "split_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    print_split_summary(
        "Train",
        train_df
    )

    print_split_summary(
        "Validation",
        validation_df
    )

    print_split_summary(
        "Test",
        test_df
    )

    print("\nNo template-cluster overlap detected.")
    print("Split files saved successfully.")


if __name__ == "__main__":
    main()