"""
Tests for Dataset Generation and Data Quality
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.generate_dataset import generate_dataset
from src.config import CATEGORIES, URGENCIES, SENTIMENTS, ACTIONS


def test_generate_dataset(tmp_path):
    output_path = tmp_path / "test_complaints.csv"
    df = generate_dataset(num_samples=50, output_path=output_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert output_path.exists()

    required_cols = [
        "complaint_id", "customer_id", "customer_name", "customer_tier",
        "channel", "complaint_text", "category", "sentiment", "urgency",
        "recommended_action", "disputed_amount", "created_at", "status"
    ]
    for col in required_cols:
        assert col in df.columns

    # Verify label validity
    assert df["category"].isin(CATEGORIES).all()
    assert df["urgency"].isin(URGENCIES).all()
    assert df["sentiment"].isin(SENTIMENTS).all()
    assert df["recommended_action"].isin(ACTIONS).all()
