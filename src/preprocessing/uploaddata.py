from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

import pandas as pd


# def upload_to_s3(ti,s3_bucket= 'mlflowdemobucket', s3_key ='wine_clean_data.csv'):
def upload_to_s3(ti,string_data:str,bucket_name:str,filename:str):
    cleaned_data = ti.xcom_pull(task_ids = 'preprocessing', key ='clean_data')
    winequality_clean_data = cleaned_data.to_csv(index=False)
    s3_hook = S3Hook(aws_conn_id='s3_conn')
    s3_hook.load_string(
            string_data=string_data,
            # filename=filename,
            key= filename,
            bucket_name=bucket_name,
            replace=True)
    print("datafile uploaded successfully")