import os
import pandas as pd
import pywt

# Klasör yolu
folder_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\4_ATAR"
output_folder = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\OSW"  # çıkış klasör

# Pencereleme parametreleri
window_size = 384
sliding_window = 32

# Çıkış klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Klasördeki tüm CSV dosyalarını al
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in files:
    file_path = os.path.join(folder_path, file)
    
    # CSV dosyasını oku
    df = pd.read_csv(file_path)
    
    # Pencereleme işlemi
    windows = []
    start = 0
    while start + window_size <= len(df):  # Son pencereyi de dahil et
        window = df.iloc[start:start + window_size]
        windows.append(window)
        start += sliding_window  # Kaydırma adımı kadar ilerle

    # Her pencereyi ayrı CSV olarak kaydet
    for i, window in enumerate(windows):
        output_file = os.path.join(output_folder, f"{file[:-4]}_window_{i}.csv")
        window.to_csv(output_file, index=False)

    print(f"{file} işlendi, {len(windows)} pencere oluşturuldu.")

print("Tüm dosyalar işlendi ve kaydedildi.")


# DWT işlemini gerçekleştirmek için tanımlanan fonksiyon
def perform_dwt(data, wavelet='db5', level=5):
    dwt_results = {}
    for column in data.columns:
        # Her kanal için 5 seviyeli DWT hesapla
        coeffs = pywt.wavedec(data[column], wavelet, level=level)
        dwt_results[column] = coeffs
    return dwt_results

# DWT işlemi için girdi ve çıktı klasörlerini tanımlama
input_dir = output_folder  # İlk kodun çıkış klasörü (pencere CSV'leri burada)
output_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Tasarım Çalışması 2\veriler\DWT"  # DWT sonuçlarının kaydedileceği klasör

# Çıkış klasörünü oluşturma
os.makedirs(output_dir, exist_ok=True)

# Tüm CSV dosyalarını sırayla işleme
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # CSV dosyasını okuma
        data = pd.read_csv(file_path)
        
        # DWT işlemi uygulama
        dwt_results = perform_dwt(data, wavelet='db5', level=5)
        
        # DWT sonuçlarını saklamak için satır listesi oluşturma
        rows = []
        for channel, coeffs in dwt_results.items():
            for level, coeff in enumerate(coeffs):
                rows.append({
                    "Channel": channel,
                    "Level": level,
                    "Coefficients": coeff.tolist()
                })
        # DataFrame oluşturma ve CSV olarak kaydetme
        df_dwt = pd.DataFrame(rows)
        output_file_path = os.path.join(output_dir, f'DWT_{filename[:-4]}.csv')
        df_dwt.to_csv(output_file_path, index=False)
                
        print(f"{filename} dosyasının DWT işlemi tamamlandı ve {output_file_path} dosyasına kaydedildi.")

print("DWT işlemi tamamlandı. Sonuçlar 'DWT_ATAR' klasörüne kaydedildi.")
