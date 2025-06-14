import os
import pandas as pd
import json
import numpy as np                      # ← eklendi
import matplotlib.pyplot as plt
from PyEMD import EMD

# Girdi ve çıktı klasörleri
input_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\DWT\DWT_S1S1_window_0.csv"
output_img_folder = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_image"
os.makedirs(output_img_folder, exist_ok=True)

# CSV'den EEG.Pz kanalını oku ve sinyali düz listeye çevir
df = pd.read_csv(input_file, header=None, names=['Channel', 'Index', 'Values'])
df_pz = df[df['Channel'] == 'EEG.Pz']
signal = []
for val in df_pz['Values']:
    try:
        val_clean = val.strip().replace("'", '"')
        signal.extend(json.loads(val_clean))
    except Exception:
        continue

# → Burada listeden NumPy dizisine dönüştürüyoruz:
signal = np.asarray(signal, dtype=float)

# Orijinal sinyal
plt.figure(figsize=(10, 3))
plt.plot(signal)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "EMD_original.png"))
plt.close()

# EMD ile IMF ayrıştırması
emd = EMD()
imfs = emd.emd(signal)                  # Artık hata vermeyecek

# Her bir IMF için grafik
for idx, imf in enumerate(imfs, start=1):
    plt.figure(figsize=(10, 3))
    plt.plot(imf)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, f"EMD_IMF_{idx}.png"))
    plt.close()

# IMF’lerin toplamından yeniden oluşturulan sinyal
reconstructed = imfs.sum(axis=0)
plt.figure(figsize=(10, 3))
plt.plot(reconstructed)
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "EMD_imftoplamları.png"))
plt.close()

print("EMD için görseller (orijinal, IMF’ler, rekonstrüksiyon) oluşturuldu.")





















# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # 1. Dosyayı oku
# file_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD\S1S1_window_0_emd.csv"
# df = pd.read_csv(file_path)

# # 2. Hangi kanalın en çok peak yaptığına bakalım
# peak_counts = {}

# for column in df.columns:
#     signal = df[column].values
#     peaks, _ = find_peaks(signal)
#     peak_counts[column] = len(peaks)

# # En çok peak yapan kanalı bul
# max_peak_channel = max(peak_counts, key=peak_counts.get)
# print(f"En çok peak yapan kanal: {max_peak_channel} ({peak_counts[max_peak_channel]} peak)")

# related_columns = [col for col in df.columns if max_peak_channel.split('_')[0] in col]

# # Eğer böyle bir yapı yoksa, yani sadece kanal adı varsa direkt o kolonu kullanacağız
# if len(related_columns) == 0:
#     related_columns = [max_peak_channel]

# # Grafik çizimi
# plt.figure(figsize=(15, 3 * len(related_columns)))

# for idx, col in enumerate(related_columns):
#     imf = df[col].values
#     plt.subplot(len(related_columns), 1, idx + 1)
#     plt.plot(imf)
#     plt.title(f"{col}")
#     plt.xlabel('Örnek')
#     plt.ylabel('Genlik')
#     plt.tight_layout()

# plt.show()




















# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# import os

# # Dosya yolları
# file_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD\S1S1_window_0_emd.csv"
# save_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_image"

# # Klasör yoksa oluştur
# os.makedirs(save_dir, exist_ok=True)

# # CSV'yi oku
# df = pd.read_csv(file_path)

# # En çok peak yapan kanalı bul
# peak_counts = {}
# for column in df.columns:
#     signal = df[column].values
#     peaks, _ = find_peaks(signal)
#     peak_counts[column] = len(peaks)

# max_peak_channel = max(peak_counts, key=peak_counts.get)
# print(f"En çok peak yapan kanal: {max_peak_channel} ({peak_counts[max_peak_channel]} peak)")

# # Aynı kanala ait IMF bileşenlerini seç
# related_columns = [col for col in df.columns if max_peak_channel.split('_')[0] in col]
# if len(related_columns) == 0:
#     related_columns = [max_peak_channel]


# # Her IMF bileşenini ayrı ayrı görsel olarak kaydet
# for col in related_columns:
#     imf = df[col].values
#     plt.figure(figsize=(10, 4))
#     plt.plot(imf)
#     plt.tight_layout()
    
#     # Kaydet
#     save_path = os.path.join(save_dir, f"{col}.png")
#     plt.savefig(save_path)
#     plt.close()  # Belleği temizlemek için figürü kapat

