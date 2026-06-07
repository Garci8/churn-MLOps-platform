from src.data.make_dataset import load_dataset
from src.data.make_dataset import process_data 
import os
import pandas as pd
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
raw_data_path = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@pytest.fixture
def dataset():
    return load_dataset(raw_data_path)

def test_make_dataset_shape(dataset):
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape[0] == 7043
    assert dataset.shape[1] == 21

def test_clean_columns(dataset):
    clean_df = process_data(dataset)
    binary_cols = [col for col in dataset.columns if set(dataset[col].unique()) == {"Yes", "No"}]
    assert 'customerID' not in clean_df.columns
    assert clean_df['TotalCharges'].isnull().sum() == 0 and clean_df['TotalCharges'].dtype == 'float64'
    assert set(clean_df['gender'].unique()) == {0, 1}
    for col in binary_cols:
        assert set(clean_df[col].unique()) == {0, 1}

    
