import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


heartDisease = pd.read_csv('heart.csv')
heartDisease = heartDisease.replace('?',np.nan)

Model=BayesianModel([('age','trestbps'),('age','fbs'),
('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),
('heartdisease','thalach'),('heartdisease','chol')])

model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(model)

q=HeartDisease_infer.query(variables=['heartdisease'],evidence
={'age':28})
print(q['heartdisease'])

#use BayesianEstimator as well
#variableelimination-for exact inference.