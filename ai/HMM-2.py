from hmmlearn import hmm
import numpy as np

# Define the HMM model
model = hmm.CategoricalHMM(n_components=2, n_iter=100)

# Training data
X = [[0, 0, 1, 1, 0, 1, 1],
     [0, 1, 1, 0, 1, 0, 0],
     [1, 0, 0, 1, 0, 1, 1]]

# Fit the HMM model to the training data
model.fit(X)

# Generate new data from the HMM model
num_samples = 10
generated_data, _ = model.sample(n_samples=num_samples)

# Calculate predictions in a Markov chain without evidence
start_prob = model.startprob_
transition_matrix = model.transmat_

predicted_states = [np.argmax(start_prob)]
for _ in range(num_samples - 1):
    predicted_state = np.argmax(transition_matrix[predicted_states[-1]])
    predicted_states.append(predicted_state)

# Combine new evidence and past predictions to estimate hidden states
evidence = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1]

past_predictions = [predicted_states[-1]]
for obs in evidence:
    new_state = np.argmax(transition_matrix[past_predictions[-1]])
    past_predictions.append(new_state)

print("Generated Data:", generated_data)
print("Predicted States (Markov Chain):", predicted_states)
print("Combined Predictions (with Evidence):", past_predictions)
