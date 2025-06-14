import os
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import json

input_file = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\DWT\DWT_S1S1_window_0.csv"
output_img_folder = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\dwt_image2"
os.makedirs(output_img_folder, exist_ok=True)

df = pd.read_csv(input_file, header=None, names=['Channel', 'Index', 'Values'])
df_pz = df[df['Channel'] == 'EEG.Pz']

signal = []
for val in df_pz['Values']:
    try:
        val_clean = val.strip().replace("'", '"')
        signal.extend(json.loads(val_clean))
    except Exception:
        continue

plt.figure(figsize=(10, 3))
plt.plot(signal)
plt.title("Level 0 - Orijinal Sinyal (EEG.Pz)")
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, "level0.png"))
plt.close()

# DWT işlemi ve hem ln hem hn görselleri
wavelet = 'db5'
level = 5
approx = signal

for i in range(1, level + 1):
    coeffs = pywt.wavedec(approx, wavelet, level=1)
    approx = coeffs[0]  # ln (approximation)
    detail = coeffs[1]  # hn (detail)

    plt.figure(figsize=(10, 3))
    plt.plot(approx)
    plt.title(f"Level {i} - A{i}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, f"level_ln{i}.png"))
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(detail)
    plt.title(f"Level {i} - D{i}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, f"level_hn{i}.png"))
    plt.close()

print("EEG.Pz kanalı için DWT ln ve hn görselleri oluşturuldu.")


