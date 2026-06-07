"""
Credit Risk Classification Package - Main Module
Provides balanced data preprocessing, ML models, business rules, and data balancing utilities.
"""

from src.preprocessing import DataPreprocessor
from src.modeling import CreditRiskModel
from src.rules import (
    hitung_kolektibilitas_ojk,
    adjust_pd_by_capacity,
    get_dynamic_threshold
)
from src.balancer import (
    smote,
    tomek_links,
    smote_tomek,
    smote_enn
)

__all__ = [
    # Data preprocessing
    "DataPreprocessor",
    
    # Machine learning models
    "CreditRiskModel",
    
    # OJK collectibility and business rules
    "hitung_kolektibilitas_ojk",
    "adjust_pd_by_capacity",
    "get_dynamic_threshold",
    
    # Data balancing techniques
    "smote",
    "tomek_links",
    "smote_tomek",
    "smote_enn",
]

__version__ = "1.0.0"
__author__ = "Credit Risk Team"
__description__ = "Credit Risk Classification System using PyCaret and scikit-learn"
