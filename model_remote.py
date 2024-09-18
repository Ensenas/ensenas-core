# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
from itertools import product
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set the path to the data directory
PATH = os.path.join('productive_dataset/colores')

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Define the number of sequences and frames
sequences = 30
frames = 4  # Ajustar a 15 frames por secuencia para medio segundo de grabación

# Create a label map to map each action label to a numeric value
label_map = {label: num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(int)

# Verificar la forma de los datos antes de hacer reshape
print(f"Forma original de X: {X.shape}")

# Ensure the input shape matches the expected (15, n_keypoints)
# n_keypoints debe reflejar el nuevo número de puntos clave (33 + 21*2 + 5*2 = 82 puntos en total, 82*3 = 246 características)
n_keypoints = 165
X = X.reshape(-1, frames, n_keypoints)

# Verificar la forma de los datos después de hacer reshape
print(f"Forma después de reshape de X: {X.shape}")

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Define the model architecture
model = Sequential()
model.add(Bidirectional(LSTM(32, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True), input_shape=(frames, n_keypoints)))
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True)))
model.add(Bidirectional(LSTM(32, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True)))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100)

# Save the trained model
model.save('colores_model.h5')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)
# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")