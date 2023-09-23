 
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix


df = pd.read_csv('diabetes.csv')

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

# BUILD NEURAL NET

X = df[col_to_analyze]

y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data into PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training and testing data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, criterion, and optimizer
model = NeuralNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
losses = []
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        outputs = model(data)
        predicted = (outputs.data > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Neural Network Accuracy: {(correct / total) * 100:.2f}%")

y_pred = []
y_true = []
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        predicted = (outputs.data > 0.5).float()
        y_pred.extend(predicted.numpy())
        y_true.extend(target.numpy())
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

