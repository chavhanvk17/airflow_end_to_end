import os
import sys
sys.path.append("/opt/airflow/")
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import sys
import boto3
from io import StringIO
import pandas as pd
from src.mlflow import param
import pickle
import joblib


# def upload_to_s3(ti,s3_bucket= 'mlflowdemobucket', s3_key ='wine_clean_data.csv'):
def cleaned_data(ti):

    cleaned_data = ti.xcom_pull(task_ids = 'preprocessing', key ='clean_data')
    return cleaned_data

def read_csv_from_s3(bucket_name="airdemobkt", file_key="wine-quality.csv"):
    
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Read CSV file directly from S3 into a Pandas DataFrame
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    model_data = pd.read_csv(StringIO(data))
    print("data file imported from s3")
    return model_data


# def eval_metrics(actual,pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))

#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual,pred)
#     return rmse, mae, r2


def mlflow_track():
    # data=cleaned_data(ti)
    data =read_csv_from_s3()
    print("data imported from s3")
    # split the data into training and test sets
    train,test = train_test_split(data)
    # the predicted columns is 'quality' 
    train_x = train.drop(['quality'],axis =1)
    test_x = test.drop(['quality'],axis=1)
    train_y = train[['quality']]
    test_y = test[["quality"]]
    cols_x = pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv',header=False, index=False)
    cols_y = pd.DataFrame(list(train_y.columns))
    cols_y.to_csv('targets.csv',header=False, index = False)
    alpha = param.alpha
    l1_ratio = param.l1_ratio

    test_x.to_csv("data/test_x.csv",index=False)
    test_y.to_csv("data/test_y.csv",index=False)        

    # alpha = 0.7
    # l1_ratio = 0.8

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model=lr.fit(train_x, train_y) 
    predicted_qualities = model.predict(test_x)
    # print(predicted_qualities)

    # rmse = eval_metrics(test_y, predicted_qualities)
    rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
    mae = mean_absolute_error(test_y, predicted_qualities)
    r2 = r2_score(test_y,predicted_qualities)
    
    # (rmse,mae,r2)= eval_metrics(test_y, predicted_qualities)
    print(" RMSE:",rmse)
    print(" MAE: ",mae)
    print(" R2: ",r2)
    model=lr
    filename = 'finalized_wine_quality_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(test_x, test_y)
    print("predict result :",result)

    key = "trained" + '_ElasticNet_model.pkl'
    bucket='airdemobkt'
    pickle_byte_obj = pickle.dumps(model) 
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)

    # joblib.dump(model, "src/modelartifact/model.joblib")
    joblib.dump(lr, os.path.join("src/modelartifact/model.joblib"))
    # joblib.dump(reg, 'regression_model.joblib')








    # ti.xcom_push(key='trainmodel', value=model)

    # mlflow.set_tracking_uri(uri="http://localhost:5000")
    # exp = mlflow.set_experiment(experiment_name="test_experiment") 

    # with mlflow.start_run(experiment_id=exp.experiment_id):

    #     mlflow.log_param("alpha",alpha)
    #     mlflow.log_param("l1_ratio", l1_ratio)
    #     mlflow.log_metric("rmse",rmse)
    #     mlflow.log_metric("r2",r2)
    #     mlflow.log_metric("MAE",mae)
    #     # mlflow.log_param('data_version', version)
    #     mlflow.log_param('input_rows',data.shape[0])
    #     mlflow.log_param('input_cols',data.shape[1]) 
    #     mlflow.log_artifact("features.csv")
    #     mlflow.log_artifact('targets.csv')
    #     mlflow.sklearn.log_model(lr,artifact_path="winequality-model")

    #     print("MLflow version:", mlflow.__version__)
    #     print("Tracking URI:", mlflow.get_tracking_uri())
    #     print("Artifact URI:", mlflow.get_artifact_uri())
    #     print("Elastic model (alpha={:f}, l1_ratio={:f}:".format(alpha,l1_ratio))


    # #     mlflow_track()
 
    