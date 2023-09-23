import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

df = pd.read_csv('diabetes.csv')

# PREPARING DATA ---
col_to_analyze = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Check for null and duplicated rows
print("Null count: ", df.isnull().sum())
print("Duplicated count: ", df.duplicated().sum())
print("Datatypes: ", df.dtypes)

# Data Cleaning
df = df[df['Pregnancies'] <= 11]
df = df[df['Glucose'] != 0]
lower_bound_bp = df['BloodPressure'].mean() - 2 * df['BloodPressure'].std()
df = df[df['BloodPressure'] >= lower_bound_bp]

# FEATURE ENGINEERING
df['BMI_Age'] = df['BMI'] * df['Age']
df['High_BP_Insulin'] = ((df['BloodPressure'] > 80) & (df['Insulin'] > 100)).astype(int)
df['Young_Obese'] = ((df['Age'] < 30) & (df['BMI'] > 30)).astype(int)
col_to_analyze.extend(['BMI_Age', 'High_BP_Insulin'])

X = df[col_to_analyze]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Convert dataset to LightGBM Dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Setup Tensorboard
writer = SummaryWriter()

def record_evaluation(callback_env, writer):
    iteration = callback_env.iteration
    for result_tuple in callback_env.evaluation_result_list:
        if len(result_tuple) == 4:
            name, metric, value, _ = result_tuple
        elif len(result_tuple) == 5:
            name, metric, _, value, _ = result_tuple
        else:
            continue
        writer.add_scalar(f"{name}/{metric}", value, iteration)


# train model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, test_data], valid_names=['training', 'valid'], callbacks=[lambda callback_env: record_evaluation(callback_env, writer)])

# predictions
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions

accuracy_lgb = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy of LightGBM: {accuracy_lgb * 100:.2f}%")

# Cleanup Tensorboard
writer.close()
