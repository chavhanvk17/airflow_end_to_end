import pandas as pd
import joblib
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('src/modelartifact/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction


# def predict_model(data):
#         # test_x=pd.read_csv("data/test_x.csv")
#         # test_y=pd.read_csv("data/test_y.csv")
#         # model = joblib.load('artifacts/model_trainer/model.joblib'))
#         model = joblib.load(Path('src/modelartifact/train_model.pkl'))

#         prediction = model.predict(data)
#         # print("prediction :",prediction)


        return prediction

