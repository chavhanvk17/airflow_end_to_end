
import pickle

def save_model(ti):
    model = ti.xcom_pull(task_ids = 'model_training', key ='trainmodel')

    filename = 'finalized_wine_quality_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    print("trained model saved successfully")