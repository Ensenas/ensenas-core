import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
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
data_dir = 'clean_dataset_v2'
model_save_path = 'my_model.keras'
label_encoder_path = 'label_encoder.pkl'

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

# Print unique labels
unique_labels = np.unique(y)
print("Unique labels in the dataset:", unique_labels)

# Ensure unique_labels match actions list
assert set(unique_labels) == set(actions), "Mismatch between dataset labels and actions list"

# Encode labels to integers
label_encoder = LabelEncoder()
label_encoder.fit(actions)  # Ensure it fits on the complete set of actions
y_encoded = label_encoder.transform(y)

# One-hot encode labels
y_one_hot = to_categorical(y_encoded, num_classes=len(actions))

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# Define the model
def create_model():
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cvscores = []

for train, test in kfold.split(X, y_encoded):
    model = create_model()

    # Define callbacks
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.001)

    # Fit the model
    history = model.fit(X[train], y_one_hot[train], epochs=50, batch_size=32,
                        validation_data=(X[test], y_one_hot[test]),
                        class_weight=class_weight_dict,
                        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback])

    # Evaluate the model
    scores = model.evaluate(X[test], y_one_hot[test], verbose=0)
    print(f"Fold accuracy: {scores[1] * 100}%")
    cvscores.append(scores[1] * 100)

print(f"Mean accuracy: {np.mean(cvscores)}% (+/- {np.std(cvscores)})")

# Save the model and the label encoder
model.save(model_save_path)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training completed and model saved.")
