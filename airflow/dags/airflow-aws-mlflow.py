import sys
sys.path.append("/opt/airflow/")
import time
from datetime import datetime,timedelta
from airflow.models.dag import DAG 
from airflow.decorators import task ,dag
from airflow.utils.task_group import TaskGroup
import pandas as pd
import mlflow
from sqlalchemy import create_engine
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import boto3
from airflow.operators.empty import EmptyOperator

from src.dummy.start import start
from src.dummy.second_phase import second_phase
from src.dummy.end import end
from src.preprocessing.importdata import import_from_database,download_from_s3,rename_file
from src.preprocessing.preprocessing import preprocessing 
from src.preprocessing.readdatafroms3 import read_csv_from_s3
from src.preprocessing.uploaddata import upload_to_s3
from src.Modeltrainig.trainmodel import mlflow_track
from src.Modeltrainig.prediction import PredictionPipeline







with DAG(dag_id="airflow-s3-mlflow",
         schedule_interval= "@daily",
         start_date=datetime(2024,2,21),
         tags=["product_model"]) as dag:
        


      

        Start_process=PythonOperator(
        task_id='Start_Pipeline',
        python_callable= start)

        Extract=EmptyOperator(
        task_id='extract_data',
        )

        import_database= PythonOperator(
        task_id ='import_database',
        python_callable= import_from_database)

        download_from_S3_task= PythonOperator(
        task_id ='download_from_s3',
        python_callable= download_from_s3,
        op_kwargs={
            'key':'wine-quality.csv',
            'bucket_name':'airdemobkt',
            'local_path':'/opt/airflow/data'}
        )

        rename_file_import_file=PythonOperator(
        task_id='rename_file',
        python_callable= rename_file,
        op_kwargs={'new_name' : 'rawdata.csv'}
        )

        read_raw_data = PythonOperator(
        task_id='read_raw_data_from_s3',
        python_callable=read_csv_from_s3,
        op_kwargs={
            'bucket_name' : 'airdemobkt',
            'file_key' : 'wine-quality.csv'},             
        provide_context=True
        )
       
        preprocess_data= PythonOperator(
        task_id ='preprocessing',
        python_callable= preprocessing)

        uploads3_task= PythonOperator(
        task_id ='upload_to_s3',
        python_callable= upload_to_s3,
        op_kwargs={
            'string_data':'winequality_clean_data',
            'filename':'winequality_clean_data',
            # 'file_key':'wine-quality.csv',
            'bucket_name':'airdemobkt'}
        )

        train_model= PythonOperator(
        task_id ='model_training',
        python_callable= mlflow_track)

        # predict_model= PythonOperator(
        # task_id ='model_prediction',
        # python_callable= PredictionPipeline)

        # pickel_model= PythonOperator(
        # task_id ='save_model',
        # python_callable= save_model)

        transform=EmptyOperator(
        task_id='transform_data',
        )

        # second_phase = PythonOperator(
        # task_id = 'second_step',
        # python_callable= second_phase)

        end_process= PythonOperator(
        task_id ='End_Pipeline',
        python_callable= end)

        


        Start_process>>Extract>> [import_database,download_from_S3_task,read_raw_data] >> transform
        transform >> preprocess_data >> uploads3_task >> train_model >> end_process
        # download_from_S3_task
        download_from_S3_task.set_downstream(rename_file_import_file)
        # Start_process.set_downstream(download_from_S3_task,rename_file_import_file)
        # Start_process.set_downstream(read_raw_data)

        # download_from_S3_task.set_downstream(rename_file_import_file)
        # rename_file_import_file.set_upstream(second_phase)
        # import_database.set_upstream(second_phase)

        # second_phase.set_downstream(read_raw_data)
    
