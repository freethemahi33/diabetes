import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('diabetes.csv')

# Get number of rows and columns
num_rows, num_columns = df.shape

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

def corrMatrix(data):
    correlation_matrix = data.corr()

    print(data.corr())

    plt.figure(figsize=(10, 8))  
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

def hist_with_bell(data, column_name, bin_count=20):

    column_data = data[column_name]

    mean_val = column_data.mean()
    std_val = column_data.std()

    # histogram
    column_data.hist(bins=bin_count, density=True, edgecolor='black', alpha=0.65)

    # Generate a range of values around the mean of our data
    x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 1000)

    # Plot normal distribution curve
    plt.plot(x, norm.pdf(x, mean_val, std_val), color='red', label='Normal Distribution')

    # Visualize standard deviations
    plt.axvline(mean_val, color='purple', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='blue', linestyle='dashed', linewidth=1, label=f'+1 std dev: {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='blue', linestyle='dashed', linewidth=1, label=f'-1 std dev: {mean_val - std_val:.2f}')
    plt.axvline(mean_val + 2*std_val, color='green', linestyle='dashed', linewidth=1, label=f'+2 std dev: {mean_val + 2*std_val:.2f}')
    plt.axvline(mean_val - 2*std_val, color='green', linestyle='dashed', linewidth=1, label=f'-2 std dev: {mean_val - 2*std_val:.2f}')

    plt.legend()
    plt.title(f'Histogram with Bell Curve for {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.show()


def summaryStats(data):

    print("Printing columns: \n", data.columns)

    print("Printing head: \n", data.head())

    print("Printing describe: \n", data.describe())

col_to_analyze = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# for i in col_to_analyze:
#     print(summaryStats(df))
#     print(hist_with_bell(df, i))

# Cleaning data notes --
    # - Pregnancies: remove everything after Pregnancies greater than 11
    # - Glucose: remove 0 values
    # - BloodPressure: remove everything past -2 std... not sure how blood pressure is being represented here
    # - SkinThickness: What to do with high density of 0 values... 
    # - Insulin: What to do with high density of 0 values...
    # - BMI: Remove bmi values that contain 0
    # - DiabetesPedigreeFunction: Lets not touch this for now as it is a direct predictor for diabetes given family history. Could remove values > 1.5

# PREPARING DATA ---

# Look at correlations

# print("Correlation matrix: ", corrMatrix(df))

# Check for null and duplicated rows

print("Null count: ", df.isnull().sum())
print("Duplicated count: ", df.duplicated().sum())
print("Datatypes: ", df.dtypes)

# Deal with pregnancies

df = df[df['Pregnancies'] <= 11]

# Deal with glucose

df = df[df['Glucose'] != 0]

# Deal with bloodpressure

lower_bound_bp = df['BloodPressure'].mean() - 2 * df['BloodPressure'].std()
df = df[df['BloodPressure'] >= lower_bound_bp]

# df = df[df['SkinThickness'] != 0]

# df = df[df['Insulin'] != 0]

# df = df[df['DiabetesPedigreeFunction'] <= 1.5]

# FEATURE ENGINEERING? 

df['BMI_Age'] = df['BMI'] * df['Age']

df['High_BP_Insulin'] = ((df['BloodPressure'] > 80) & (df['Insulin'] > 100)).astype(int)

df['Young_Obese'] = ((df['Age'] < 30) & (df['BMI'] > 30)).astype(int)
 
col_to_analyze.append('BMI_Age')
col_to_analyze.append('High_BP_Insulin')
# col_to_analyze.append('Young_Obese')

print("Correlation matrix: ", corrMatrix(df))

# BUILD MODEL

X = df[col_to_analyze]

y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

treeModel = DecisionTreeClassifier(random_state=42)

treeModel.fit(X_train, y_train)

treePrediction = treeModel.predict(X_test)

print(treePrediction)

# -- LOGISTIC REGRESSION

logisticModel = LogisticRegression(random_state=42, max_iter=1000)

logisticModel.fit(X_train, y_train)

logisticPrediction = logisticModel.predict(X_test)

# EVALUATE MODEL

accuracy = accuracy_score(y_test, treePrediction)
logistic_accuracy = accuracy_score(y_test, logisticPrediction)

print(f"Accuracy of DecisionTreeClassifier: {accuracy * 100:.2f}%")
print(f"Accuracy of Logistic Regression: {logistic_accuracy * 100:.2f}%")

plt.figure(figsize=(100, 50))
plot_tree(treeModel, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.show()