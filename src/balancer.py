# src/balancer.py

from __future__ import annotations

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


def smote(
    X,
    y,
    random_state: int = 42,
):
    """
    SMOTE (Synthetic Minority Over-sampling Technique).
    """

    sampler = SMOTE(
        random_state=random_state
    )

    return sampler.fit_resample(X, y)


def tomek_links(
    X,
    y,
):
    """
    Tomek Links.
    """

    sampler = TomekLinks()

    return sampler.fit_resample(X, y)


def smote_tomek(
    X,
    y,
    random_state: int = 42,
):
    """
    SMOTE + Tomek Links.
    """

    sampler = SMOTETomek(
        random_state=random_state
    )

    return sampler.fit_resample(X, y)


def smote_enn(
    X,
    y,
    random_state: int = 42,
):
    """
    SMOTE + Edited Nearest Neighbours.
    """

    sampler = SMOTEENN(
        random_state=random_state
    )

    return sampler.fit_resample(X, y)


__all__ = [
    "smote",
    "tomek_links",
    "smote_tomek",
    "smote_enn",
]