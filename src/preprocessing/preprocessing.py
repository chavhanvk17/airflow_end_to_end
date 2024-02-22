
import sys
sys.path.append("/opt/airflow/")
import pandas as pd
import numpy as np
import datetime as datetime


def preprocessing(ti):

    data=pd.read_csv("data/rawdata.csv")
    shape = data.shape
       
     # printing shape
    print("Shape = {}".format(shape))
    ti.xcom_push(key='clean_data', value=data)
   




