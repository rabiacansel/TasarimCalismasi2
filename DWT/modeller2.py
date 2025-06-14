import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM, Bidirectional, GRU
from tensorflow.keras.utils import to_categorical  
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.regularizers import l2

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

matrix_df = pd.read_csv(matrix_file)
label_df = pd.read_excel(label_file)
merged_data = pd.merge(matrix_df, label_df, on="Subject", how="inner")
X = merged_data.loc[:, "Minimum Value":"Relative Energy"].values 

scaler = StandardScaler()
X_reshaped = X.reshape(-1, 10)  
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 30, 10)  

y_int = merged_data["Label"].values  
y_int = y_int.reshape(-1, 30)[:, 0]
y = to_categorical(y_int, num_classes=2)

def KNN():
    X_flat = X.reshape(X.shape[0], -1)
    y_int_local = np.argmax(y, axis=1)
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_flat, y_int_local, test_size=0.1, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)
    y_pred = knn.predict(X_test_knn)
    
    acc = accuracy_score(y_test_knn, y_pred)
    print(f"KNN Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_knn, y_pred))
    
    cm = confusion_matrix(y_test_knn, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("KNN Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_knn == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_knn == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_knn)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_knn)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def SVM():
    X_flat = X.reshape(X.shape[0], -1)
    y_int_local = np.argmax(y, axis=1)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_flat, y_int_local, test_size=0.1, random_state=42)
    
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train_svm, y_train_svm)
    y_pred = svm_model.predict(X_test_svm)
    
    acc = accuracy_score(y_test_svm, y_pred)
    print(f"SVM Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_svm, y_pred))
    
    cm = confusion_matrix(y_test_svm, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("SVM Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_svm == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_svm == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_svm)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_svm)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def RF():
    X_flat = X.reshape(X.shape[0], -1)
    y_int_local = np.argmax(y, axis=1)
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_flat, y_int_local, test_size=0.1, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    y_pred = rf.predict(X_test_rf)
    
    acc = accuracy_score(y_test_rf, y_pred)
    print(f"RF Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_rf, y_pred))
    
    cm = confusion_matrix(y_test_rf, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("RF Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_rf == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_rf == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_rf)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_rf)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def BİLSTM():
    X_train_bilstm, X_test_bilstm, y_train_bilstm, y_test_bilstm = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = Sequential([
        Input(shape=(30, 10)),
        Bidirectional(LSTM(256, activation='tanh', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)),
        Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('BiLSTM_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-1)
    ]
    
    history = model.fit(X_train_bilstm, y_train_bilstm, epochs=100, batch_size=100,
                        validation_split=0.1, callbacks=callbacks)
    
    BiLSTM_DWT_best_model = load_model('BiLSTM_DWT_best_model.keras')
    loss, accuracy = BiLSTM_DWT_best_model.evaluate(X_test_bilstm, y_test_bilstm)
    model.save("BiLSTM_DWT_best_model.keras")
    print(f"Bi-LSTM Test Loss: {loss:.2f}")
    print(f"Bi-LSTM Test Accuracy: {accuracy:.2f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Bi-LSTM Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    y_pred = np.argmax(BiLSTM_DWT_best_model.predict(X_test_bilstm), axis=1)
    y_true = np.argmax(y_test_bilstm, axis=1)
    
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Bi-LSTM Confusion Matrix")
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


üdef GRU_model():
    X_train_gru, X_test_gru, y_train_gru, y_test_gru = train_test_split(X, y, test_size=0.1, random_state=42)
    model = Sequential([
    Input(shape=(30, 10)),

    GRU(256, return_sequences=True, activation='tanh', recurrent_dropout=0.2),
    BatchNormalization(),
    Dropout(0.3),

    GRU(128, activation='tanh', recurrent_dropout=0.2),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  
    callbacks = [
        ModelCheckpoint('GRU_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]
    
    history = model.fit(X_train_gru, y_train_gru, epochs=100, batch_size=100, validation_split=0.1, callbacks=callbacks)
    GRU_DWT_best_model = load_model('GRU_DWT_best_model.keras')
    loss, accuracy = GRU_DWT_best_model.evaluate(X_test_gru, y_test_gru)
    model.save("GRU_DWT_best_model.keras")
    print(f"GRU Test Loss: {loss:.2f}")
    print(f"GRU Test Accuracy: {accuracy:.2f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='train_accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='red')
    plt.title('GRU Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    y_pred = np.argmax(GRU_DWT_best_model.predict(X_test_gru), axis=1)
    y_true = np.argmax(y_test_gru, axis=1)
    
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("GRU Confusion Matrix")
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
#KNN()
#SVM()
#RF()
#BİLSTM()
#GRU_model()
