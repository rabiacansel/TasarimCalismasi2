import os
import glob
import pandas as pd
import re

# Özellik hesaplama fonksiyonları
def compute_energy(series: pd.Series) -> float:
    """
    Enerji: değerlerin karelerinin toplamı
    """
    return (series ** 2).sum()

# CSV dosyalarının bulunduğu klasör yolu
data_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD"
# Çıktı dosyalarının kaydedileceği klasör
output_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\EMD_feature"
os.makedirs(output_dir, exist_ok=True)

feature_rows = []

# Sadece *_emd.csv dosyalarını tara
for file_path in glob.glob(os.path.join(data_dir, "*_emd.csv")):
    df = pd.read_csv(file_path)
    base_name = os.path.basename(file_path)
    parts = base_name.split("_")
    subject = parts[0]  # Örn: "S1S1"

    # "window" kelimesinden sonraki sayıyı al
    try:
        if "window" in parts:
            idx = parts.index("window")
            window = int(parts[idx + 1])
        else:
            window = None
    except (ValueError, IndexError):
        window = None

    # Her sütun adı için kanal & IMF bilgisi al, sadece IMFn<=3 olanları işle
    for col in df.columns:
        m = re.match(r"(.*)_IMF_(\d+)$", col)
        if not m:
            continue
        channel, imf_num = m.group(1), int(m.group(2))
        if imf_num > 3:
            continue

        series = df[col]
        feature_rows.append({
            'Subject': subject,
            'Window': window,
            'Channel': channel,
            'IMF': imf_num,
            'Minimum Value': series.min(),
            'Maximum Value': series.max(),
            'Median': series.median(),
            'Mean': series.mean(),
            'Energy': compute_energy(series),
            'Standard Deviation': series.std()
        })

# DataFrame oluştur ve sütun sırasını düzenle
features_df = pd.DataFrame(feature_rows, columns=[
    'Subject', 'Window', 'Channel', 'IMF',
    'Minimum Value', 'Maximum Value', 'Median', 'Mean',
    'Energy', 'Standard Deviation'
])

# Sonuçları CSV olarak kaydet
out_csv = os.path.join(output_dir, "EMD_feature.csv")
features_df.to_csv(out_csv, index=False)

print(f"Özellikler kaydedildi: {out_csv}")
