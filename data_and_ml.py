import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

# 1. NumPy & Pandas: Data simulation And dataframe transformation 
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(loc=70, scale=10, size=n_samples) # sensor reading 
vibration = np.random.normal(loc=0.5, scale=0.15, size=n_samples)
pressure = np.random.normal(loc=101.3, scale=5, size=n_samples)

# 2. SciPy: Uncertainty Budget & Standard Deviation Calculation
temp_std = np.std(temperature)
temp_sem = stats.sem(temperature) # Standard Error of Mean (Uncertainty)
vibration_uncertainty = stats.iqr(vibration) # Interquartile Range

# Failure Label Generation (1 = Maintenance Needed, 0 = Normal)
failure = ((temperature > 82) | (vibration > 0.75) | (pressure < 92)).astype(int)

df = pd.DataFrame({
    'Temperature': temperature,
    'Vibration': vibration,
    'Pressure': pressure,
    'Failure': failure
})

# 3. Seaborn & Matplotlib: making data visualization figure 
def generate_plots():
    plt.figure(figsize=(10, 4))
    sns.scatterplot(data=df, x='Temperature', y='Vibration', hue='Failure', palette='coolwarm')
    plt.title(f'Sensor Data Distribution (Temp Uncertainty SEM: {temp_sem:.4f})')
    plt.savefig('sensor_distribution.png')
    plt.close()

generate_plots()

# 4. Scikit-Learn: Machine Learning Model (Random Forest)
X = df[['Temperature', 'Vibration', 'Pressure']]
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

# 5. PyTorch: Deep Learning Model (Neural Network)
class SensorNN(nn.Module):
    def __init__(self):
        super(SensorNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

torch_X_train = torch.FloatTensor(X_train_scaled)
torch_y_train = torch.FloatTensor(y_train.values).unsqueeze(1)

nn_model = SensorNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = nn_model(torch_X_train)
    loss = criterion(outputs, torch_y_train)
    loss.backward()
    optimizer.step()

print(f"Data Processing Completed!")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"PyTorch Neural Network Final Loss: {loss.item():.4f}")