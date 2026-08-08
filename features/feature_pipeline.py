import pandas as pd

from features.structural_features import extract_structural_features
from features.behavioral_features import extract_behavioral_features
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


def build_structural_feature_frame(text_series: pd.Series) -> pd.DataFrame:
    """
    Convert a Pandas Series of recruitment texts into
    a numeric structural-feature DataFrame.
    """

    feature_rows = [
        extract_structural_features(text)
        for text in text_series
    ]

    return pd.DataFrame(feature_rows)

def build_behavioral_feature_frame(text_series: pd.Series) -> pd.DataFrame:
    """
    Convert a Pandas Series of recruitment texts into
    a behavioral-feature DataFrame.
    """

    feature_rows = [
        extract_behavioral_features(text)
        for text in text_series
    ]

    return pd.DataFrame(feature_rows)

def build_engineered_feature_frame(
    text_series: pd.Series
) -> pd.DataFrame:
    """
    Combine structural and behavioral features into
    one engineered-feature DataFrame.
    """

    structural_df = build_structural_feature_frame(
        text_series
    )

    behavioral_df = build_behavioral_feature_frame(
        text_series
    )

    return pd.concat(
        [
            structural_df.reset_index(drop=True),
            behavioral_df.reset_index(drop=True),
        ],
        axis=1,
    )
    
class BehavioralFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Scikit-learn compatible transformer for the
    frozen behavioral feature set.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text_series = pd.Series(X).reset_index(
            drop=True
        )

        feature_df = build_behavioral_feature_frame(
            text_series
        )

        return csr_matrix(
            feature_df.values,
            dtype=float,
        )