from hmmlearn import hmm

# Define the HMM model
model = hmm.MultinomialHMM(n_components=2)

# Training data
X = [[0, 0, 1, 1, 0, 1, 1],
     [0, 1, 1, 0, 1, 0, 0],
     [1, 0, 0, 1, 0, 1, 1]]

# Fit the HMM model to the training data
model.fit(X)

# Predict the weather for the next day
predicted_weather = model.predict([[0, 0, 0, 0, 0, 0, 0]])

# Convert the predicted weather states to actual weather conditions
weather_conditions = ['sunny', 'rainy']
predicted_conditions = [weather_conditions[state] for state in predicted_weather]

print("Predicted Weather:", predicted_conditions)