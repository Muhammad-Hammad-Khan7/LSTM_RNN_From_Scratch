#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


data= pd.read_csv('/content/LSTM-Multivariate_pollution.csv')

# In[21]:


data.head()

# In[22]:


data.isnull().sum()

# In[23]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# In[24]:


x = data.drop(['date', 'wnd_dir', 'pollution'], axis=1)
y = data['pollution']

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    shuffle=False
)


# In[26]:


X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# #Creating series for predictions

# In[27]:


def create_lstm_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    return np.array(X_sequences), np.array(y_sequences)


# In[28]:


sequence_length = 24

X_train, y_train = create_lstm_sequences(
    X_train_scaled, y_train_scaled, sequence_length
)

X_test, y_test = create_lstm_sequences(
    X_test_scaled, y_test_scaled, sequence_length
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# #Activation Functions

# In[29]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(tanh_output):
    return 1 - tanh_output ** 2


# #weight initialization for all parameters

# In[30]:


class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        # weights of all gates that are in lstms
        self.weight_forget = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.weight_input = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.weight_candidate = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.weight_output = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        # weights of biases
        self.bias_forget = np.zeros((hidden_size, 1))
        self.bias_input = np.zeros((hidden_size, 1))
        self.bias_candidate = np.zeros((hidden_size, 1))
        self.bias_output = np.zeros((hidden_size, 1))

        # output layer
        self.weight_dense = np.random.randn(output_size, hidden_size) * 0.01
        self.bias_dense = np.zeros((output_size, 1))


# #Forward propagation in LSTM

# In[31]:


def forward_pass(model, input_sequence):
    time_steps = input_sequence.shape[0]
    #Short term memory and long term memory and the values of all gates  in LSTM
    hidden_states = {}
    cell_states = {}
    gate_values = {}

    hidden_states[-1] = np.zeros((model.hidden_size, 1))
    cell_states[-1] = np.zeros((model.hidden_size, 1))

    for t in range(time_steps):
        current_input = input_sequence[t].reshape(-1, 1)
        combined_input = np.vstack((hidden_states[t-1], current_input))

        forget_gate = sigmoid(model.weight_forget @ combined_input + model.bias_forget)
        input_gate = sigmoid(model.weight_input @ combined_input + model.bias_input)
        candidate_cell = tanh(model.weight_candidate @ combined_input + model.bias_candidate)
        output_gate = sigmoid(model.weight_output @ combined_input + model.bias_output)

        cell_states[t] = forget_gate * cell_states[t-1] + input_gate * candidate_cell
        hidden_states[t] = output_gate * tanh(cell_states[t])

        gate_values[t] = (forget_gate, input_gate, candidate_cell, output_gate, combined_input)

    prediction = model.weight_dense @ hidden_states[time_steps-1] + model.bias_dense
   #It return the prediction and the values of all gates and short term and long term memory
    cache = (hidden_states, cell_states, gate_values)
    return prediction, cache


# #Backward propagation in Lstm

# In[32]:


def backward_pass(model, prediction, target, cache):
    hidden_states, cell_states, gate_values = cache
    time_steps = len(gate_values)
    #calculte that how much every input causes the erros and stores it
    d_weight_dense = np.zeros_like(model.weight_dense)
    d_bias_dense = np.zeros_like(model.bias_dense)

    d_weight_forget = np.zeros_like(model.weight_forget)
    d_weight_input = np.zeros_like(model.weight_input)
    d_weight_candidate = np.zeros_like(model.weight_candidate)
    d_weight_output = np.zeros_like(model.weight_output)

    d_bias_forget = np.zeros_like(model.bias_forget)
    d_bias_input = np.zeros_like(model.bias_input)
    d_bias_candidate = np.zeros_like(model.bias_candidate)
    d_bias_output = np.zeros_like(model.bias_output)

    d_hidden_next = np.zeros((model.hidden_size, 1))
    d_cell_next = np.zeros((model.hidden_size, 1))

    d_prediction = prediction - target.reshape(-1, 1)

    d_weight_dense += d_prediction @ hidden_states[time_steps-1].T
    d_bias_dense += d_prediction

    d_hidden = model.weight_dense.T @ d_prediction

    for t in reversed(range(time_steps)):
        forget_gate, input_gate, candidate_cell, output_gate, combined_input = gate_values[t]

        tanh_cell = tanh(cell_states[t])
        d_output_gate = d_hidden * tanh_cell
        d_output_gate *= sigmoid_derivative(output_gate)

        d_cell = d_hidden * output_gate * tanh_derivative(tanh_cell)
        d_cell += d_cell_next

        d_forget_gate = d_cell * cell_states[t-1]
        d_forget_gate *= sigmoid_derivative(forget_gate)

        d_input_gate = d_cell * candidate_cell
        d_input_gate *= sigmoid_derivative(input_gate)

        d_candidate = d_cell * input_gate
        d_candidate *= tanh_derivative(candidate_cell)

        d_weight_forget += d_forget_gate @ combined_input.T
        d_weight_input += d_input_gate @ combined_input.T
        d_weight_candidate += d_candidate @ combined_input.T
        d_weight_output += d_output_gate @ combined_input.T

        d_bias_forget += d_forget_gate
        d_bias_input += d_input_gate
        d_bias_candidate += d_candidate
        d_bias_output += d_output_gate

        d_combined = (
            model.weight_forget.T @ d_forget_gate +
            model.weight_input.T @ d_input_gate +
            model.weight_candidate.T @ d_candidate +
            model.weight_output.T @ d_output_gate
        )

        d_hidden = d_combined[:model.hidden_size, :]
        d_cell_next = d_cell * forget_gate

  # Updating the parameters using weight updation forula
    for param, grad in zip(
        [model.weight_dense, model.bias_dense,
         model.weight_forget, model.weight_input,
         model.weight_candidate, model.weight_output,
         model.bias_forget, model.bias_input,
         model.bias_candidate, model.bias_output],
        [d_weight_dense, d_bias_dense,
         d_weight_forget, d_weight_input,
         d_weight_candidate, d_weight_output,
         d_bias_forget, d_bias_input,
         d_bias_candidate, d_bias_output]
    ):
        param -= model.learning_rate * grad


# #training the model

# In[33]:


def train_model(model, X_train, y_train, epochs):
    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X_train)):
            prediction, cache = forward_pass(model, X_train[i])
            loss = np.mean((prediction - y_train[i].reshape(-1,1))**2)
            total_loss += loss
            backward_pass(model, prediction, y_train[i], cache)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train):.6f}")


# #predictions

# In[34]:


def predict(model, X_test):
    predictions = []
    for seq in X_test:
        pred, _ = forward_pass(model, seq)
        predictions.append(pred.flatten())
    return np.array(predictions)


# In[35]:


model = LSTMModel(
    input_size=X_train.shape[2],
    hidden_size=32,
    output_size=1,
    learning_rate=0.01
)
train_model(model, X_train, y_train, epochs=10)

# In[39]:


predictions = predict(model, X_test)

plt.figure(figsize=(12,5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Air Pollution Prediction using LSTM (From Scratch)")
plt.show()


# In[39]:




# In[39]:



