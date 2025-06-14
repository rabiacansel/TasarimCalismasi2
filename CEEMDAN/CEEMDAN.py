import os
import glob
import pandas as pd
from PyEMD import CEEMDAN

# Girdi veri klasör yolu
input_folder = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\OSW"

# Çıktı veri klasör yolu
output_folder = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\CEEMDAN"

# Eğer çıktı klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# İşlenecek kanallar
channels = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]

# Tüm CSV dosyalarını listele
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Her CSV dosyası için işle
for file in csv_files:
    df = pd.read_csv(file)
    file_name = os.path.basename(file)
    file_base = os.path.splitext(file_name)[0]  # Örn: S1S1_window_0

    all_imfs = []

    for channel in channels:
        signal = df[channel].values
        ceemdan = CEEMDAN()
        imfs = ceemdan.ceemdan(signal)
        
        imf_df = pd.DataFrame(imfs.T)
        imf_df.columns = [f"{channel}_IMF_{i+1}" for i in range(imf_df.shape [1])]
        all_imfs.append(imf_df)

    result_df = pd.concat(all_imfs, axis=1)

    new_filename = os.path.join(output_folder, f"{file_base}_ceemdan.csv")
    result_df.to_csv(new_filename, index=False)

    print(f"{new_filename} kaydedildi.")
