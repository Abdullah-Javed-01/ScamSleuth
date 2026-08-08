import unittest

import pandas as pd

from config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
)


EXPECTED_COLUMNS = [
    "id",
    "text_type",
    "text",
    "label",
    "difficulty",
    "scam_category",
    "template_cluster",
    "strong_signals",
    "weak_signals",
    "label_reason",
]


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.raw_df = pd.read_csv(
            RAW_DATA_PATH
        )

        cls.train_df = pd.read_csv(
            TRAIN_DATA_PATH
        )

        cls.validation_df = pd.read_csv(
            VALIDATION_DATA_PATH
        )

        cls.test_df = pd.read_csv(
            TEST_DATA_PATH
        )

    def test_raw_shape(self):

        self.assertEqual(
            self.raw_df.shape,
            (600, 10),
        )

    def test_expected_columns(self):

        self.assertEqual(
            self.raw_df.columns.tolist(),
            EXPECTED_COLUMNS,
        )

    def test_no_missing_values(self):

        self.assertEqual(
            int(
                self.raw_df
                .isna()
                .sum()
                .sum()
            ),
            0,
        )

    def test_unique_ids(self):

        self.assertEqual(
            self.raw_df["id"].nunique(),
            600,
        )

    def test_labels(self):

        self.assertEqual(
            set(
                self.raw_df[
                    "label"
                ].unique()
            ),
            {
                "Safe",
                "Scam",
            },
        )

    def test_balanced_labels(self):

        counts = (
            self.raw_df[
                "label"
            ]
            .value_counts()
            .to_dict()
        )

        self.assertEqual(
            counts["Safe"],
            300,
        )

        self.assertEqual(
            counts["Scam"],
            300,
        )

    def test_split_sizes(self):

        self.assertEqual(
            len(self.train_df),
            420,
        )

        self.assertEqual(
            len(self.validation_df),
            90,
        )

        self.assertEqual(
            len(self.test_df),
            90,
        )

    def test_cluster_leakage(self):

        train_clusters = set(
            self.train_df[
                "template_cluster"
            ]
        )

        validation_clusters = set(
            self.validation_df[
                "template_cluster"
            ]
        )

        test_clusters = set(
            self.test_df[
                "template_cluster"
            ]
        )

        self.assertTrue(
            train_clusters.isdisjoint(
                validation_clusters
            )
        )

        self.assertTrue(
            train_clusters.isdisjoint(
                test_clusters
            )
        )

        self.assertTrue(
            validation_clusters.isdisjoint(
                test_clusters
            )
        )

    def test_total_rows_preserved(self):

        total = (
            len(self.train_df)
            + len(self.validation_df)
            + len(self.test_df)
        )

        self.assertEqual(
            total,
            600,
        )


if __name__ == "__main__":
    unittest.main()