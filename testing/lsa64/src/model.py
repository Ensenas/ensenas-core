import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# Define actions based on the provided list

actions = [
    'Opaco', 'Rojo', 'Verde'
]

# Load the dataset
def load_dataset(data_dir):
    all_data = []
    all_labels = []
    max_seq_length = 0

    for action in actions:
        action_dir = os.path.join(data_dir, action)
        if os.path.isdir(action_dir):
            for file in os.listdir(action_dir):
                if file.endswith('.npy'):
                    file_path = os.path.join(action_dir, file)
                    keypoints = np.load(file_path)
                    all_data.append(keypoints)
                    all_labels.append(action)
                    if len(keypoints) > max_seq_length:
                        max_seq_length = len(keypoints)

    # Pad sequences to the same length
    padded_data = []
    for keypoints in all_data:
        padding = np.zeros((max_seq_length - len(keypoints), keypoints.shape[1]))
        padded_keypoints = np.vstack((keypoints, padding))
        padded_data.append(padded_keypoints)

    return np.array(padded_data), np.array(all_labels)

# Define paths
data_dir = './clean_dataset'
model_save_path = 'my_model.keras'
label_encoder_path = './label_encoder.pkl'

# Load data
X, y = load_dataset(data_dir)

# Debug prints to check data loading
print("Loaded data shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample y:", y[:10])

# Check if data is loaded correctly
if X.size == 0 or y.size == 0:
    raise ValueError("Loaded data or labels are empty.")

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode labels
y_one_hot = to_categorical(y_encoded)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define the checkpoint callback to save the model
model_checkpoint_callback = ModelCheckpoint(
    filepath=model_save_path,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])

model.save(model_save_path)
# Save the model and the label encoder
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training completed and model saved.")
