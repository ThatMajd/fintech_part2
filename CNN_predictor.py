import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'fintech_part2//combined_ticks.csv'
df = pd.read_csv(data_path)

df.drop(columns=['Unnamed: 0','Iron_Close'], inplace=True)

# Fill null values with the average of previous and next rows
df.fillna((df.shift() + df.shift(-1)) / 2, inplace=True)

print(df.isnull().sum())

# List all features
features = df.columns.tolist()
print("Features:", features)

# Assume 'Gold_Close' is the target variable
target = 'Gold_Close'
X = df.drop(columns=[target]).values
y = df[target].values

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape X to fit the CNN model
X = X.reshape(X.shape[0], 1, X.shape[1], 1)  # Reshape to (batch_size, channels, height, width)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create a DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.relu = nn.ReLU()

        # Determine the size of the flattened layer dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, X_train.shape[2], 1)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.pool(dummy_output)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.pool(dummy_output)
            dummy_output = self.conv3(dummy_output)
            dummy_output = self.pool(dummy_output)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
learning_rate = 0.001
num_epochs = 200

# Model, loss function, and optimizer
model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
# Training loop with batch processing
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    test_mae = mean_absolute_error(y_test_tensor.numpy(), predictions.numpy())
    print(f'Test Loss (MSE): {test_loss.item():.4f}')
    print(f'Test MAE: {test_mae:.4f}')


# Plot train losses
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Save the model
model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
