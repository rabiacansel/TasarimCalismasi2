import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Orijinal dosya yolu
input_file_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_feature\EMD_feature.csv"

# Yeni dosya yolları
output_csv_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_feature\EMD_feature_normalized.csv"

# Veriyi oku
df = pd.read_csv(input_file_path)

# Normalizasyon yapılacak sütunlar
columns_to_normalize = ["Minimum Value", "Maximum Value", "Median", "Mean", "Energy", "Standard Deviation"]

# Normalizasyon işlemi
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Yeni CSV dosyasına kaydet
df.to_csv(output_csv_path, index=False)


print("Normalizasyon tamamlandı. Veriler CSV dosyasına kaydedildi.")
