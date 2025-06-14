import numpy as np
import pandas as pd
import tensorflow as tf
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

matrix_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\Feature\feature_normalizasyon.csv"
label_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\label.xlsx"

matrix_df = pd.read_csv(matrix_file)
label_df = pd.read_excel(label_file)
merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")
X = merged_data.loc[:, "Minimum Value":"Relative Energy"].values 

scaler = StandardScaler()
X_reshaped = X.reshape(-1, 10)  # 2D hale getir
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 30, 10)  # Tekrar 3D hale getir

y = merged_data["Label"].values  
y = y.reshape(-1, 30)[:, 0]
y = to_categorical(y, num_classes=2) 

# Eğitim ve test için bölüyoruz (burada y one-hot encoded, ancak değerlendirme için integer'a çevireceğiz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential([
   Input(shape=(30, 10)), 
   Conv1D(256, kernel_size=3, activation='relu'),  
   BatchNormalization(),
   MaxPooling1D(pool_size=2),
   Dropout(0.25),
   Conv1D(128, kernel_size=3 , activation='relu'),
   BatchNormalization(),
   MaxPooling1D(pool_size=2),
   Dropout(0.25),
   Conv1D(64, kernel_size=3, activation='relu'),
   BatchNormalization(),
   MaxPooling1D(pool_size=2),
   Dropout(0.25),
   Flatten(),
   Dense(256, activation='relu'),
   Dense(256, activation='relu'),
   Dense(128, activation='relu'),
   Dense(128, activation='relu'),
   Dense(64, activation='relu'), 
   Dense(64, activation='relu'), 
   Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',                                                      
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('CNN_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=100,  
    validation_split=0.1,
    callbacks=callbacks
)

CNN_DWT_best_model = load_model('CNN_DWT_best_model.keras') 
loss, accuracy = CNN_DWT_best_model.evaluate(X_test, y_test)
model.save("CNN_DWT_best_model.keras")
print(f"CNN Test Loss: {loss:.2f}")
print(f"CNN Test Accuracy: {accuracy:.2f}")

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

# Tahminleri al ve integer etiketlere çevir
y_pred = np.argmax(CNN_DWT_best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Değerlendirme metrikleri için y_true kullanıyoruz
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("CNN Confusion Matrix")
plt.show()

# Yanlış tahminleri belirleme
false_positives = np.where((y_true == 0) & (y_pred == 1))[0]  # Gerçek 0, tahmin 1
false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]  # Gerçek 1, tahmin 0

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








































# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
# from tensorflow.keras.utils import to_categorical  
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# matrix_file=r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\Feature\feature_normalizasyon.csv"
# label_file=r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\label.xlsx"

# matrix_df=pd.read_csv(matrix_file)
# label_df=pd.read_excel(label_file)
# merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")
# X = merged_data.loc[:, "Minimum Value":"Relative Energy"].values 

# scaler = StandardScaler()
# X_reshaped = X.reshape(-1, 10)  # 2D hale getir
# X_scaled = scaler.fit_transform(X_reshaped)
# X = X_scaled.reshape(-1, 30, 10)  # Tekrar 3D hale getir


# # X = X.reshape(-1, 30, 10)
# y = merged_data["Label"].values  
# y = y.reshape(-1, 30)[:, 0]
# y = to_categorical(y, num_classes=2) 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# model = Sequential([
#    Input(shape=(30, 10)), 
#    Conv1D(256, kernel_size=3, activation='relu'),  
#    BatchNormalization(),
#    MaxPooling1D(pool_size=2),
#    Dropout(0.25),
#    Conv1D(128, kernel_size=3 , activation='relu'),
#    BatchNormalization(),
#    MaxPooling1D(pool_size=2),
#    Dropout(0.25),
#    Conv1D(64, kernel_size=3, activation='relu'),
#    BatchNormalization(),
#    MaxPooling1D(pool_size=2),
#    Dropout(0.25),
#    Flatten(),
#    Dense(256, activation='relu'),
#    Dense(128, activation='relu'),
#    Dense(64, activation='relu'),   
#    Dense(2, activation='softmax'),
   
#    ])

# # optimizer = Adam(learning_rate=0.01)                                              #öğrenme oranı
# # model.compile(optimizer=optimizer,                                                 #optimizer='adam',
# #               loss='categorical_crossentropy',                                     # One-hot encoding olduğu için 'categorical_crossentropy' binary_crossentropy
# #               metrics=['accuracy'])


# model.compile(optimizer='adam',                                                      
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # # optimizer = SGD(learning_rate=0.001, momentum=0.9)  # momentum=0.9
# # optimizer = RMSprop(learning_rate=0.01) # rho=0.9  , epsilon=1e-7
# # model.compile(optimizer=optimizer,                                                 
# #               loss='categorical_crossentropy',                                    
# #               metrics=['accuracy'])






# callbacks = [
#     # EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
#     ModelCheckpoint('first_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
# ]

# history = model.fit(
#     X_train, y_train,
#     epochs=100, 
#     batch_size=100,  
#     validation_split=0.1,
#     callbacks=callbacks
# )
# first_best_model = load_model('first_best_model.keras') 
# loss, accuracy = first_best_model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss:.2f}")
# print(f"Test Accuracy: {accuracy:.2f}")
# train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# plt.figure(figsize=(6, 4))
# plt.plot(train_accuracy, label='train_accuracy', color='blue') 
# plt.plot(val_accuracy, label='val_accuracy', color='red') 
# plt.title('Eğitim ve Doğrulama Doğruluğu')
# plt.xlabel('Epoch')
# plt.ylabel('Doğruluk')
# plt.legend()
# plt.tight_layout()
# plt.show()





# # learning rate | accuarcy| loss  
# #  0.0001           0.84    0.33
# #  0.1              0.49    0.70
# #  0.01             0.94    0.16  - hafif dalgalanmalar val_accuarcy train accuarcy tam takip ediyor.
# # değersiz          0.94          - ikisi de daha düz, val accuarcy değeri sonda hafif aşağı gidiyor.                    






















# # ###############

# # 962  0.02
# # 9 4 1         0.14 100
# # 952
# # 851 0.015 , 0.014, 0.013 ,  0.012(0.38 0.83  dalgalı 1e yakın)                           (0.015)    0.24  0.83   




# # 5 5 1               0.42   0.83    çoğunluk düz çıktı 
# # 353                 0.64   0.83 
# #  9 2 1              0.48 - 0.83   ;    0.33 - 0.83

# # optimizer = RMSprop(learning_rate=0.01)                                        # , rho=0.9
# # model.compile(optimizer=optimizer,                                                 
# #               loss='categorical_crossentropy',                                    
# #               metrics=['accuracy'])























# # # optimizer = SGD(learning_rate=0.001, momentum=0.9)  # momentum=0.9
# # optimizer = RMSprop(learning_rate=0.01) # rho=0.9  , epsilon=1e-7
# # model.compile(optimizer=optimizer,                                                 
# #               loss='categorical_crossentropy',                                    
# #               metrics=['accuracy'])



