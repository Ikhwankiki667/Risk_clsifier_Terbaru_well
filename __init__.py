"""
NexBank Credit Risk Classifier
A comprehensive credit risk assessment system using machine learning and OJK business rules.

Main Modules:
- src.preprocessing: Data preprocessing and feature engineering
- src.modeling: Machine learning model training and prediction
- src.rules: OJK collectibility classification and risk adjustment
- src.balancer: Class imbalance handling techniques (SMOTE, Tomek, etc.)

Example usage:
    from src import DataPreprocessor, CreditRiskModel, hitung_kolektibilitas_ojk
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(df_train)
    
    # Train model
    model = CreditRiskModel()
    model.train(X_processed, y)
    
    # Make predictions and assess risk
    X_pred = preprocessor.transform(df_pred)
    pd_value = model.predict_default_prob(X_pred)
    kol, decision, color, reason, pd_adj = hitung_kolektibilitas_ojk(
        pd_value=pd_value,
        hari_tunggakan=0,
        riwayat_default="No"
    )
"""

from src import (
    DataPreprocessor,
    CreditRiskModel,
    hitung_kolektibilitas_ojk,
    adjust_pd_by_capacity,
    get_dynamic_threshold,
    smote,
    tomek_links,
    smote_tomek,
    smote_enn,
)

__all__ = [
    # Core preprocessing
    "DataPreprocessor",
    
    # Machine learning
    "CreditRiskModel",
    
    # Business rules (OJK/BI compliance)
    "hitung_kolektibilitas_ojk",
    "adjust_pd_by_capacity",
    "get_dynamic_threshold",
    
    # Data balancing
    "smote",
    "tomek_links",
    "smote_tomek",
    "smote_enn",
]

__version__ = "1.0.0"
__author__ = "NexBank Credit Risk Team"
__title__ = "NexBank Credit Risk Classifier"
__description__ = "Credit risk assessment system using ML + OJK regulations"
