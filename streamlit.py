import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define parameters matching the trained model
input_size = 9  # Adjust this to match the input size used during training
hidden_size = 64
num_layers = 2
output_size = 10  # Adjust this to match the number of classes used during training

# Load the model
model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('fault_type_rnn_model.pth'))
model.eval()

# Load the training data used for fitting the scaler
# (Replace with the path to your training data file)
data_time = pd.read_csv("C:/Users/OKOK PRO/Downloads/CODE_Fault_Classification/Dataset/feature_time_48k_2048_load_1.csv")

# Fit the scaler with the training data
scaler = StandardScaler()
scaler.fit(data_time.iloc[:, :-1])  # Fit only on feature columns

label_encoder = LabelEncoder()
label_encoder.fit(data_time['fault'])  # Fit label encoder on fault labels

def preprocess_input(data):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return data_tensor

# Streamlit UI
st.title("Fault Type Prediction")

# Input fields for features (replace with your actual feature names and input types)
feature_1 = st.number_input("Feature 1")
feature_2 = st.number_input("Feature 2")
feature_3 = st.number_input("Feature 3")
feature_4 = st.number_input("Feature 4")
feature_5 = st.number_input("Feature 5")
feature_6 = st.number_input("Feature 6")
feature_7 = st.number_input("Feature 7")
feature_8 = st.number_input("Feature 8")
feature_9 = st.number_input("Feature 9")
# Add more features as required, and adjust input_size accordingly

# Prepare the input for the model
input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]])  # Replace with all input features
input_tensor = preprocess_input(input_data)

# Predict
if st.button("Predict Fault Type"):
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = torch.argmax(prediction, 1).item()
        predicted_fault_type = label_encoder.inverse_transform([predicted_class])[0]
        st.write(f"Predicted Fault Type: {predicted_fault_type}")
