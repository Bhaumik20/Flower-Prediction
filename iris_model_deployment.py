
from tensorflow.keras.models import load_model
import joblib
import numpy as np

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

def predict_flower(model,scaler,sample_json):

    '''
        Returns species to which flower belong
    '''
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    flower = [[s_len,s_wid,p_len,p_wid]]
    flower = scaler.transform(flower)
    class_ind = model.predict_classes(flower)[0]
    return classes[class_ind]

