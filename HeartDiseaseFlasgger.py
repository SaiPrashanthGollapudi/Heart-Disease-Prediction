# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:15:10 2022

@author: Sai Prashanth
"""

import pandas as pd
dataset = pd.read_csv('Heart_Disease.csv')
LRmodel = pd.read_pickle("rf_model")

from flask import Flask, request
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('rf_model.pkl','rb')
rf_model=pickle.load(pickle_in)

@app.route('/')

def Data_Scientist():
    return "Welcome Data Scientist"

@app.route('/predict', methods=["Get"])
def Heart_Disease_Prediction():
    
    """Let's Predict the Heart Disease
    This is using docstrings for specifications.
    
    ---
    
    parameters:
        -name:slope_of_peak_exercise_st_segment
         in: query
         type: number
         required: true
        -name:thal
         in: query
         type: number
         required: true
        -name:resting_blood_pressure
         in: query
         type: number
         required: true
        -name:chest_pain_type
         in: query
         type: number
         required: true
        -name:num_major_vessels
         in: query
         type: number
         required: true
        -name:fasting_blood_sugar_gt_120_mg_per_dl
         in: query
         type: number
         required: true
        -name:resting_ekg_results
         in: query
         type: number
         required: true
        -name:serum_cholesterol_mg_per_dl
         in: query
         type: number
         required: true
        -name: age
         in: query
         type: number
         required: true
        -name: exercise_induced_angina
         in: query
         type: number
         required: true
    responses:
        100:
            description: The output value is 
    """
        
    slope_of_peak_exercise_st_segment=request.args.get('slope_of_peak_exercise_st_segment')
    thal=request.args.get('thal')
    resting_blood_pressure=request.args.get('resting_blood_pressure')
    chest_pain_type=request.args.get('chest_pain_type')
    num_major_vessels=request.args.get('num_major_vessels')
    fasting_blood_sugar_gt_120_mg_per_dl=request.args.get('fasting_blood_sugar_gt_120_mg_per_dl')
    resting_ekg_results=request.args.get('resting_ekg_results')
    serum_cholesterol_mg_per_dl=request.args.get('serum_cholesterol_mg_per_dl')
    age=request.args.get('age')
    exercise_induced_angina=request.args.get('exercise_induced_angina')
    prediction=rf_model.predict([[slope_of_peak_exercise_st_segment,thal,resting_blood_pressure,chest_pain_type,num_major_vessels,fasting_blood_sugar_gt_120_mg_per_dl,resting_ekg_results,serum_cholesterol_mg_per_dl,age,exercise_induced_angina]])
    return "The Predcited value is "+str(prediction)




if __name__ == '__main__':
    app.run()