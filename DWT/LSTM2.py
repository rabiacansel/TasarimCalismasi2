import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

matrix_df = pd.read_csv(matrix_file)
label_df = pd.read_excel(label_file)
merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")

X = merged_data.loc[:, "Minimum Value":"Relative Energy"].values 
scaler = StandardScaler()
X_reshaped = X.reshape(-1, 10)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 30, 10)

y = merged_data["Label"].values  
y = y.reshape(-1, 30)[:, 0]
y = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

model = Sequential([
    LSTM(512, activation='tanh', dropout=0.2, recurrent_dropout=0.1,  
         return_sequences=True, input_shape=(30, 10)),
    LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.1
         ),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')    
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('LSTM_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-1)
]

history = model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=100,  
    validation_split=0.1,
    callbacks=callbacks
)

LSTM_DWT_best_model = load_model('LSTM_DWT_best_model.keras') 
loss, accuracy = LSTM_DWT_best_model.evaluate(X_test, y_test)
model.save("LSTM_DWT_best_model.keras")
print(f"LSTM Test Loss: {loss:.2f}")
print(f"LSTM Test Accuracy: {accuracy:.2f}")

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('LSTM Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

y_pred = np.argmax(LSTM_DWT_best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Label")
plt.title("LSTM Confusion Matrix")
plt.tight_layout()
plt.show()
