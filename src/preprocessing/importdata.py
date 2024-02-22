from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

import pandas as pd
import os



def import_from_database():
    
#  hook = PostgresHook(postgres_conn_id="postgres_conn")
#  raw_data = hook.get_pandas_df(sql="select * from wine_quality;")
#  ti.xcom_push(key='wine_dataframe', value=raw_data)

#  shape = raw_data.shape   
#  # printing shape
#  print("Shape = {}".format(shape))    
#  print("raw_data extraction is done and transfer the dataframe to next task")
   print("in progress")
 
# def import_from_s3(ti):
  
def download_from_s3(ti,key:str,bucket_name=str,local_path=str) -> str :
    hook = S3Hook(aws_conn_id='s3_conn')
    rw_data = hook.download_file(key=key,bucket_name=bucket_name,local_path=local_path)
    # winedata=filename.to_csv(axis=False)
    ti.xcom_push(key='rw_data_csv', value=rw_data) 
    return rw_data

def rename_file(ti, new_name):
    download_file_name =ti.xcom_pull(task_ids=['download_from_s3'])
    download_file_path = '/'.join(download_file_name[0].split('/')[:-1])
    os.rename(src=download_file_name[0],dst=f"{download_file_path}/{new_name}")