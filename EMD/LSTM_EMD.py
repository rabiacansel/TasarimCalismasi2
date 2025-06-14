import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Dosya yollarınızı kontrol edin:
matrix_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_feature\EMD_feature_normalized.csv"
label_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\label.xlsx"

# Veri yükleme ve birleştirme:
matrix_df = pd.read_csv(matrix_file)
label_df = pd.read_excel(label_file)
merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")

# Özelliklerin seçilmesi ve ölçekleme:
X = merged_data.loc[:, "Minimum Value":"Standard Deviation"].values 
scaler = StandardScaler()
X_reshaped = X.reshape(-1, 6)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 15, 6)

# Etiketlerin hazırlanması:
y = merged_data["Label"].values  
y = y.reshape(-1, 15)[:, 0]
y = to_categorical(y, num_classes=2)

# Eğitim ve test verilerinin ayrılması:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Yeni model mimarisi:
model = Sequential([
    LSTM(256, activation='tanh', dropout=0.2, recurrent_dropout=0.1,  
         # aşırı öğrenme engeller
         #LSTM içinde önceki zamandan gelen bilgilerin %10’unu rastgele sıfırlar.
         return_sequences=True, input_shape=(15, 6)),
    LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.1
         ),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')    
])

# Modelin derlenmesi:
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback ayarları:
callbacks = [
    ModelCheckpoint('LSTM_EMD_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-1)
    # min_lr=1e-1
    #monitor öğrenme azaltılacağı, factor learning rate çarparak adımı küçültme, 
    #min_lr öğrenme oranı düşmemesi için alt sınır, patince n epoch sayısınca gelişim olmazsa
]

# Modelin eğitilmesi:
history = model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=100,  
    validation_split=0.1,
    callbacks=callbacks
)

# En iyi modeli yükleme ve test verilerinde değerlendirme:
LSTM_EMD_best_model = load_model('LSTM_EMD_best_model.keras') 
loss, accuracy = LSTM_EMD_best_model.evaluate(X_test, y_test)
model.save("LSTM_EMD_best_model.keras")
print(f"LSTM Test Loss: {loss:.2f}")
print(f"LSTM Test Accuracy: {accuracy:.2f}")

# Eğitim sürecinin görselleştirilmesi:
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('LSTM Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Test verileri üzerinde tahminler:
y_pred = np.argmax(LSTM_EMD_best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Sınıflandırma raporu ve karmaşıklık matrisinin oluşturulması:
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print("LSTM Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("LSTM Confusion Matrix")
plt.tight_layout()
plt.show()

#     ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-23) 
# Test Loss: 0.25
# Test Accuracy: 0.91

# ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-1)
# Test Loss: 0.30
# Test Accuracy: 0.91