import numpy as np
import pandas as pd
import tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical  
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

matrix_df = pd.read_csv(matrix_file)
label_df = pd.read_excel(label_file)
merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")
X = merged_data.loc[:, "Minimum Value":"Standard Deviation"].values 

scaler = StandardScaler()
X_reshaped = X.reshape(-1, 6) 
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 15, 6)  

y = merged_data["Label"].values  
y = y.reshape(-1, 15)[:, 0]
y = to_categorical(y, num_classes=2) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential([
   Input(shape=(15, 6)), 
   Conv1D(128, kernel_size=2, padding='same', activation='relu'),  
   BatchNormalization(),
    MaxPooling1D(pool_size=1),
   Dropout(0.1),
   Conv1D(64, kernel_size=2 , padding='same', activation='relu'),
   BatchNormalization(),
    MaxPooling1D(pool_size=1),
   Dropout(0.1),
   Conv1D(32, kernel_size=2, padding='same', activation='relu'),
   BatchNormalization(),
    MaxPooling1D(pool_size=1),
   Dropout(0.1),
   Flatten(),
   Dense(256, activation='relu'),
   Dense(128, activation='relu'),
   Dense(64, activation='relu'),  
   Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',                                                      
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callbacks = [
    ModelCheckpoint('CNN_EMD_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
]
history = model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=10,  
    validation_split=0.1,
    callbacks=callbacks
)

CNN_EMD_best_model = load_model('CNN_EMD_best_model.keras') 
loss, accuracy = CNN_EMD_best_model.evaluate(X_test, y_test)
model.save("CNN_EMD_best_model.keras")
print(f"Test Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.figure(figsize=(6, 4))
plt.plot(train_accuracy, label='train_accuracy', color='blue') 
plt.plot(val_accuracy, label='val_accuracy', color='red') 
plt.title('CNN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

y_pred = np.argmax(CNN_EMD_best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"CNN Test Accuracy: {accuracy:.2f}")
print(report)
print("CNN Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("CNN Confusion Matrix")
plt.show()

false_positives = np.where((y_true == 0) & (y_pred == 1))[0]  
false_negatives = np.where((y_true == 1) & (y_pred == 0))[0] 

print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
print(pd.DataFrame({
    "Index": false_positives,
    "True Label": y_true[false_positives],
    "Predicted Label": y_pred[false_positives]
}))

print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
print(pd.DataFrame({
    "Index": false_negatives,
    "True Label": y_true[false_negatives],
    "Predicted Label": y_pred[false_negatives]
}))
