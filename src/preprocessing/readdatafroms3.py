import boto3
import pandas as pd
from io import StringIO

def read_csv_from_s3(ti,bucket_name, file_key):
    
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Read CSV file directly from S3 into a Pandas DataFrame
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    raw_data = pd.read_csv(StringIO(data))
    ti.xcom_push(key='clean_model_input_data', value=raw_data)
    print("data from s3 is read in the dag")

    # return raw_data
