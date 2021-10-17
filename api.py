import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


class Instance(BaseModel):
    sys_sector: str
    sys_label: str
    sys_process: str
    sys_product: str
    sys_dataspecification_version: float
    sys_claimid: str
    sys_currency_code: str
    claim_amount_claimed_total: float
    claim_causetype: str
    claim_date_occurred: int
    claim_date_reported: int
    claim_location_urban_area: int
    object_make: str
    object_year_construction: int
    ph_firstname: str
    ph_gender: str
    ph_name: str
    policy_fleet_flag: int
    policy_insured_amount: float
    policy_profitability: str


def __feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to remove redundant features from the dataset.

    :param data: dataframe with the features
    :return: dataframe of features without the redundant features
    """
    # dropping one-valued features
    columns_to_drop = ['sys_sector', 'sys_label', 'sys_process', 'sys_product', 'sys_dataspecification_version', 'sys_currency_code']
    # drop names -> DB-based logic/alerts are better in using such specific info
    columns_to_drop.append('ph_firstname')
    columns_to_drop.append('ph_name')
    # drop the more complex id - redundant
    columns_to_drop.append('sys_claimid')
    print(f'[>] Dropping the following columns for lack of infromative value: {columns_to_drop}')
    print('[>] Dropping initial id column, only need one identifier to trace instances')

    return data.drop(columns_to_drop, axis=1)


def __inference_handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to handle missing values on the different features of the dataset. Different approaches
    are adopted in each case (removal, encoding and imputation).

    :param data: dataframe with the features
    :return: dataframe of features without the redundant features
    """
    # claim_causetype: 2 instances and showed no particularities -> drop
    data.dropna(axis=0, subset=['claim_causetype'], inplace=True)

    # ph_gender: 798 instances and showed no particularities -> 1. drop 2. encode as <no_gender>
    data['ph_gender'].fillna('NG', inplace=True)

    # policy_insured_amount: 30817 instances and showed no particularities -> 1. impute 2. business details (nan=0Euro?)
    cols_to_impute_with = ['claim_amount_claimed_total', 'object_year_construction', 'policy_insured_amount']
    with open('knn_imputer_pkl', 'rb') as f:
        knn_imp = pickle.load(f)
    imputed_values = knn_imp.fit_transform(data[cols_to_impute_with])
    data['policy_insured_amount'] = imputed_values[:, 2]

    # drop any rows with negative amounts -> check for negatives meaning
    data = data[data['policy_insured_amount'] >= 0]
    return data


def __inference_prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function responsible for data preparation before inference. Saved preprocessing functions are utilized.

    :param data: dataframe with the features
    :return: dataframe with the features prepared to be used by predictive model
    """
    with open('qt_trans_pkl', 'rb') as f:
        qt_trans = pickle.load(f)
    with open('scaler_pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ohe_pkl', 'rb') as f:
        ohe = pickle.load(f)

    x = pd.DataFrame()
    # continuous data
    data['claim_reported_days'] = (
            pd.to_datetime(data['claim_date_reported'], format='%Y%m%d') -
            pd.to_datetime(data['claim_date_occurred'], format='%Y%m%d')
    ).dt.days
    x['policy_insured_amount'] = np.log(data['policy_insured_amount'])
    x['claim_reported_days'] = np.sqrt(data['claim_reported_days'])
    x['claim_amount_claimed_total'] = qt_trans.transform(data['claim_amount_claimed_total'].to_numpy().reshape(-1, 1))
    x['object_age'] = datetime.now().year - data['object_year_construction']
    x[['policy_insured_amount', 'claim_amount_claimed_total', 'claim_reported_days', 'object_age']] = \
        scaler.fit_transform(
            x[['policy_insured_amount', 'claim_amount_claimed_total', 'claim_reported_days', 'object_age']]
        )

    # categorical data
    data.replace({
        'policy_fleet_flag': {0: 'not_policy_fleet', 1: 'policy_fleet'},
        'claim_location_urban_area': {0: 'non_urban', 1: 'urban'}},
        inplace=True)
    categorical = ['claim_causetype', 'claim_location_urban_area', 'object_make', 'ph_gender', 'policy_fleet_flag',
                   'policy_profitability']
    ohe_categorical = ohe.transform(data[categorical])
    for ohe_cat in ohe.categories_[:-1]:
        for col_name, col_values in zip(ohe_cat, ohe_categorical.T):
            x[col_name] = col_values
    return x


app = FastAPI()


@app.post('/score/')
def predict_instance(instance: Instance):
    if instance:
        instance_df = pd.json_normalize(vars(instance))
        instance_df = __feature_selection(instance_df)
        instance_df = __inference_handle_missing_values(instance_df)
        instance_df = __inference_prepare_data(instance_df)
        with open(Path('./logit_model_pkl'), 'rb') as f:
            loaded_model = pickle.load(f)
        instance_prediction = loaded_model.predict(instance_df)
        return 'Fraud' if instance_prediction else 'Not fraud'
    else:
        return {"My friend, you forgot something. Where's my instance?"}
