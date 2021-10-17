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
3. RestAPI: FastAPI is used with uvicorn. To initialize run `uvicorn app:api`. The endpoint is `/score` and expects a json containing a single instance for inference. The keys of this JSON should be the same as the feature names in the provided dataset. 

## Docker
Attempt at containerizing the RestAPI was not successful, due to dependency building errors. fraud_predict_api.dockerfile contains the attempt.
