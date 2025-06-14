import os
import pandas as pd
import pywt

window_size = 384
sliding_window = 32
os.makedirs(output_folder, exist_ok=True)
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in files:
    file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        windows = []
    start = 0
    while start + window_size <= len(df): 
        window = df.iloc[start:start + window_size]
        windows.append(window)
        start += sliding_window  

for i, window in enumerate(windows):
        output_file = os.path.join(output_folder, f"{file[:-4]}_window_{i}.csv")
        window.to_csv(output_file, index=False)

def perform_dwt(data, wavelet='db5', level=5):
    dwt_results = {}
    for column in data.columns:
        coeffs = pywt.wavedec(data[column], wavelet, level=level)
        dwt_results[column] = coeffs
    return dwt_results

input_dir = output_folder 
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        data = pd.read_csv(file_path)
        dwt_results = perform_dwt(data, wavelet='db5', level=5)

        rows = []
        for channel, coeffs in dwt_results.items():
            for level, coeff in enumerate(coeffs):
                rows.append({
                    "Channel": channel,
                    "Level": level,
                    "Coefficients": coeff.tolist()
                })
        df_dwt = pd.DataFrame(rows)
        output_file_path = os.path.join(output_dir, f'DWT_{filename[:-4]}.csv')
        df_dwt.to_csv(output_file_path, index=False)
                
