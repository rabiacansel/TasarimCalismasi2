import pandas as pd
import numpy as np
import ast
import os
import glob
from scipy.stats import skew

def compute_entropy(coeffs):
    """
    Verilen katsayıların normalize edilmiş mutlak değerleri üzerinden Shannon Entropy hesaplar.
    """
    coeffs = np.array(coeffs)
    abs_coeffs = np.abs(coeffs)
    total = abs_coeffs.sum()
    if total == 0:
        return 0
    probs = abs_coeffs / total
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log2(probs))
    return ent

# Verilerin bulunduğu dizin (DWT klasörü)
data_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\DWT"
# Dizin altındaki tüm CSV dosyalarını al (tüm 54 deneğe ait 289 pencere dosyası)
file_paths = glob.glob(os.path.join(data_dir, "*.csv"))

print(f"Toplam {len(file_paths)} dosya bulunuyor.")

# Tüm dosyalardan elde edilecek sonuçların tutulacağı liste
results = []

for i, file_path in enumerate(file_paths):
    print(f"[{i+1}/{len(file_paths)}] İşleniyor: {file_path}")
    
    # Dosya adından Subject (örneğin S1S1) ve Window (örneğin window_0) bilgisini çıkarıyoruz
    file_name = os.path.basename(file_path)
    parts = file_name.split("_")
    subject_info = parts[1]                # Örneğin "S1S1"
    window_info = parts[-1].split(".")[0]    # Örneğin "window_0"
    
    # Dosyayı oku
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Dosya okunamadı: {file_path} - Hata: {e}")
        continue
    
    features_list = []
    
    # Her satırdaki "Coefficients" sütunundan özellikleri hesapla
    for index, row in df.iterrows():
        try:
            coeffs = ast.literal_eval(row["Coefficients"])
            coeffs = np.array(coeffs, dtype=float)
            if coeffs.size > 0:
                min_val    = np.min(coeffs)
                max_val    = np.max(coeffs)
                median_val = np.median(coeffs)
                mean_val   = np.mean(coeffs)
                energy_val = np.sum(coeffs**2)
                std_val    = np.std(coeffs)
                var_val    = np.var(coeffs)
                skew_val   = skew(coeffs)
                ent_val    = compute_entropy(coeffs)
            else:
                min_val = max_val = median_val = mean_val = energy_val = std_val = var_val = skew_val = ent_val = np.nan
        except Exception as e:
            print(f"Hata oluştu! Dosya: {file_name}, Satır: {index}, Hata: {e}")
            min_val = max_val = median_val = mean_val = energy_val = std_val = var_val = skew_val = ent_val = np.nan
        
        features_list.append({
            "Subject": subject_info,
            "Window": window_info,
            "Channel": row["Channel"],
            "Level": row["Level"],
            "Minimum Değer": min_val,
            "Maksimum Değer": max_val,
            "Medyan": median_val,
            "Ortalama (Mean)": mean_val,
            "Enerji (Energy)": energy_val,
            "Standart Sapma (Standard Deviation)": std_val,
            "Varyans (Variance)": var_val,
            "Çarpıklık (Skewness)": skew_val,
            "Entropy": ent_val
        })
    
    # DataFrame oluştur
    features_df = pd.DataFrame(features_list)
    
    # Aynı kanala ait satırlarda "Göreceli Enerji (Relative Energy)" hesapla
    features_df["Göreceli Enerji (Relative Energy)"] = np.nan
    for channel, group in features_df.groupby("Channel"):
        total_energy = group["Enerji (Energy)"].sum()
        if total_energy != 0:
            features_df.loc[group.index, "Göreceli Enerji (Relative Energy)"] = group["Enerji (Energy)"] / total_energy
        else:
            features_df.loc[group.index, "Göreceli Enerji (Relative Energy)"] = np.nan
    
    # İstenen sütun sırası: Meta (Subject, Window, Channel, Level) + 10 özellik
    features_df = features_df[["Subject", "Window", "Channel", "Level", 
                               "Minimum Değer", "Maksimum Değer", "Medyan", "Ortalama (Mean)",
                               "Enerji (Energy)", "Standart Sapma (Standard Deviation)", "Varyans (Variance)",
                               "Çarpıklık (Skewness)", "Entropy", "Göreceli Enerji (Relative Energy)"]]
    
    # Dosyaya ait özellik matrisini sonuçlara ekle
    results.append(features_df)
    
    # Matrisler arasında 1 boş satır eklemek için aynı sütunlarda boş bir satır ekle
    blank_row = pd.DataFrame([[""] * len(features_df.columns)], columns=features_df.columns)
    results.append(blank_row)

# Son ek boş satırı kaldır
if results:
    results = results[:-1]

# Tüm sonuçları alt alta ekleyip tek bir DataFrame oluştur
final_df = pd.concat(results, ignore_index=True)

# Çıktıyı CSV dosyası olarak kaydediyoruz
output_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\Feature\feature.csv"
final_df.to_csv(output_path, index=False)

print(f"Özellikler CSV dosyası oluşturuldu: {output_path}")
