from hmmlearn import hmm

# Define the HMM model
model = hmm.CategoricalHMM(n_components=1)

# Training data
text = "Hello, how are you today? I hope you're doing well."
text = text.lower().split()

# Create a mapping of words to integer values
word_to_id = {word: idx for idx, word in enumerate(set(text))}
id_to_word = {idx: word for word, idx in word_to_id.items()}

# Convert text to integer sequence
X = [word_to_id[word] for word in text]
X = [[x] for x in X]

# Fit the HMM model to the training data
model.fit(X)

# Generate a sequence of observations
num_words = 10
generated_sequence, _ = model.sample(n_samples=num_words)

# Convert the generated sequence to text
generated_text = ' '.join(id_to_word[idx] for idx in generated_sequence.flatten())

print("Generated Text:", generated_text)

