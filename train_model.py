import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import joblib


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load and preprocess your data
df = pd.read_csv("cleaned_ttc_subway_data.csv")

# Assuming 'Min Delay' is the target variable
X = df.drop("Min Delay", axis=1).values
y = df["Min Delay"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create Tensor Datasets and DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Initialize the model
model = RegressionModel(X_train.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        # Forward pass
        preds = model(xb)
        loss = criterion(preds, yb.unsqueeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    total_loss = 0
    for xb, yb in test_loader:
        preds = model(xb)
        loss = criterion(preds, yb.unsqueeze(1))
        total_loss += loss.item()
    print(f"Average Test Loss: {total_loss / len(test_loader)}")

torch.save(model.state_dict(), "model_state_dict.pth")
# Save the scaler
joblib.dump(scaler, "scaler.pkl")
