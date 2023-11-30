import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn as nn
import torch.nn.functional as F


# Define the model structure (same as during training)
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


# Load the model
model = RegressionModel(368)  # Replace 'input_size' with the correct value
model.load_state_dict(torch.load("model_state_dict.pth"))
model.eval()

# Load the scaler (assuming you saved it after training)
scaler = joblib.load("scaler.pkl")


testing_data = pd.read_csv("testing_data.csv")
input_df = pd.DataFrame(testing_data)

# Normalize the input data
input_scaled = scaler.transform(input_df)

# Convert to PyTorch tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Make a prediction
with torch.no_grad():
    predicted_delay = model(input_tensor)

print(f"Predicted delay time: {predicted_delay.item()} minutes")
