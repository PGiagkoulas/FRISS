# FRISS assessment
 FRISS assessment on insurance claim fraud detection.

## Contents
* Assessment_solution.ipynb: main solution of the assessment
* api.py: RestAPI code
* fraud_predict_api.dockerfile: dockerfile, attempt to containerize RestAPI
* requirements.txt: file with all Python packages required to reproduce development environment

## How to
1. Recreate the virual environment using requirements.txt
2. Run the code in Assessment_solution.ipynb. The data are expected to be on the same directory on a `/data` subdirectory. Executing this notebook will generate the necessary files for the RestAPI to provide inferences. The necessary generated files are:
    * knn_imputer_pkl: fitted knn imputer
    * qt_trans_pkl: fitted quartile transformer
    * scaler_pkl: fitted scaler
    * ohe_pkl: fitted one-hot encoder
    * logit_model_pkl: fitted predictive model (LogisticRegression)
3. RestAPI: FastAPI is used with uvicorn. To initialize run `uvicorn app:api`. The endpoint is `/score` and expects a json containing a single instance for inference. The keys of this JSON should be the same as the feature names in the provided dataset. The order and type details of the expected JSON.
```
{
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
}
```

## Docker
Attempt at containerizing the RestAPI was not successful, due to dependency building errors. fraud_predict_api.dockerfile contains the attempt.
