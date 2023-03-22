# Databricks notebook source
import numpy as np
import pandas as pd

training_data_file_name = "/dbfs/FileStore/shared_uploads/sujitnair51@gmail.com/german_credit_data.csv"
data_df = pd.read_csv(training_data_file_name)


# COMMAND ----------

data_df.head()

# COMMAND ----------

print('Columns:', list(data_df.columns))
print('Number of columns: ', len(data_df.columns))

# COMMAND ----------

print('Number of records: ', data_df.Risk.count())

# COMMAND ----------

target_count = data_df.groupby('Risk')['Risk'].count()
target_count

# COMMAND ----------

target_count.plot.pie(figsize=(8,8));

# COMMAND ----------

MODEL_NAME = "Scikit German Risk Model WML v4"
DEPLOYMENT_NAME = "Scikit German Risk Deployment WML V4"

# COMMAND ----------

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# COMMAND ----------

train_data, test_data = train_test_split(data_df, test_size=0.2)

# COMMAND ----------

features_idx = np.s_[0:-1]
all_records_idx = np.s_[:]
first_record_idx = np.s_[0]

# COMMAND ----------

string_fields = [type(fld) is str for fld in train_data.iloc[first_record_idx, features_idx]]
ct = ColumnTransformer([("ohe", OneHotEncoder(), list(np.array(train_data.columns)[features_idx][string_fields]))])
clf_linear = SGDClassifier(loss='log', penalty='l2', max_iter=1000, tol=1e-5)

pipeline_linear = Pipeline([('ct', ct), ('clf_linear', clf_linear)])

# COMMAND ----------

risk_model = pipeline_linear.fit(train_data.drop('Risk', axis=1), train_data.Risk)

# COMMAND ----------

from sklearn.metrics import roc_auc_score
test_data.drop('Risk', axis=1)
predictions = risk_model.predict(test_data.drop('Risk', axis=1))
probability = risk_model.predict_proba(test_data.drop('Risk', axis=1))
# print(predictions)
print(probability[0])
fields = ["pridictions","probability"]
values = predictions
predictions = np.array(predictions)
probability = np.array(probability)
stack = np.column_stack((predictions, np.array(probability))).tolist()
squares = []
for index, val in enumerate(predictions):
    value = [val, probability[index].tolist()]
    squares.append(value)

response ={"fields": fields, "values": squares} 
print(response)

# COMMAND ----------

indexed_preds = [0 if prediction=='No Risk' else 1 for prediction in predictions]

real_observations = test_data.Risk.replace('Risk', 1)
real_observations = real_observations.replace('No Risk', 0).values

auc = roc_auc_score(real_observations, indexed_preds)
print(auc)

# COMMAND ----------

import pickle
# pickle.dump(risk_model, open('./test.pkl', 'wb'))

# COMMAND ----------

ls

# COMMAND ----------

ls /dbfs/FileStore/shared_uploads/sujitnair51@gmail.com

# COMMAND ----------

filename = "/dbfs/FileStore/shared_uploads/sujitnair51@gmail.com/credit_risk.pkl"
with open(filename, 'wb') as f:
    pickle.dump(risk_model, f)

# COMMAND ----------

filename = "https://adb-5883458866989557.17.azuredatabricks.net/?o=5883458866989557#folder/169856641840855"
with open(filename, 'wb') as f:
    pickle.dump(risk_model, f)
