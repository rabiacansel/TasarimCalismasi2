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
# Dizin altındaki tüm CSV dosyalarını al (54 deneğe ait 289 pencere dosyası gibi)
file_paths = glob.glob(os.path.join(data_dir, "*.csv"))

print(f"Toplam {len(file_paths)} dosya bulunuyor.")

# Tüm dosyalardan elde edilecek sonuçların tutulacağı liste
results = []

for i, file_path in enumerate(file_paths):
    print(f"[{i+1}/{len(file_paths)}] İşleniyor: {file_path}")
    
    # Dosya adından Subject (ör. S1S1) ve Window (ör. window_0) bilgisini çıkar
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
            "Minimum Value": min_val,
            "Maximum Value": max_val,
            "Median": median_val,
            "Mean": mean_val,
            "Energy": energy_val,
            "Standard Deviation": std_val,
            "Variance": var_val,
            "Skewness": skew_val,
            "Entropy": ent_val
        })
    
    # Oluşturulan listeyi DataFrame'e çeviriyoruz
    features_df = pd.DataFrame(features_list)
    
    # Aynı kanala ait satırlarda "Relative Energy" hesapla
    features_df["Relative Energy"] = np.nan
    for channel, group in features_df.groupby("Channel"):
        total_energy = group["Energy"].sum()
        if total_energy != 0:
            features_df.loc[group.index, "Relative Energy"] = group["Energy"] / total_energy
        else:
            features_df.loc[group.index, "Relative Energy"] = np.nan
    
    # İstenen sütun sırası: Meta (Subject, Window, Channel, Level) + 10 özellik
    features_df = features_df[["Subject", "Window", "Channel", "Level", 
                               "Minimum Value", "Maximum Value", "Median", "Mean",
                               "Energy", "Standard Deviation", "Variance",
                               "Skewness", "Entropy", "Relative Energy"]]
    
    # Bu dosyaya ait özellik matrisini sonuçlara ekle
    results.append(features_df)
    
    # Matrisler arasında 1 boş satır eklemek için aynı sütunlarda boş bir satır ekliyoruz
    blank_row = pd.DataFrame([[""] * len(features_df.columns)], columns=features_df.columns)
    results.append(blank_row)

# Son ek boş satırı kaldırıyoruz
if results:
    results = results[:-1]

# Tüm sonuçları alt alta ekleyip tek DataFrame oluşturuyoruz
final_df = pd.concat(results, ignore_index=True)

# ----------------------------------------------
# BLOK BAZINDA -1 ile 1 Arasında Normalizasyon Uygulama
# ----------------------------------------------
# Normalizasyon uygulanacak sayısal sütunlar:
numeric_cols = ["Minimum Value", "Maximum Value", "Median", "Mean", "Energy", 
                "Standard Deviation", "Variance", "Skewness", "Entropy", "Relative Energy"]

# final_df içerisindeki boş satırları atlayarak, blokları belirleyelim.
block_indices = []
current_block = []

for idx, row in final_df.iterrows():
    # Boş satırları tespit etmek için "Subject" sütununa bakıyoruz (boş ise blok ayracı)
    if row["Subject"] == "":
        if current_block:
            block_indices.append(current_block)
            current_block = []
    else:
        current_block.append(idx)
if current_block:
    block_indices.append(current_block)

# Her blok için, seçilen numeric sütunları -1 ile 1 aralığına normalize edelim
for block in block_indices:
    block_data = final_df.loc[block, numeric_cols].astype(float)
    for col in numeric_cols:
        col_min = block_data[col].min()
        col_max = block_data[col].max()
        if col_max != col_min:
            normalized = 2 * (block_data[col] - col_min) / (col_max - col_min) - 1
        else:
            normalized = 0  # Sabit değer durumunda 0 alalım
        final_df.loc[block, col] = normalized

# ----------------------------------------------
# Normalizasyon sonrası sonuçları CSV olarak kaydediyoruz
output_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\Feature\feature_normalizasyon.csv"
final_df.to_csv(output_path, index=False)

print(f"Özellikler normalizasyon sonrası CSV dosyası oluşturuldu: {output_path}")
