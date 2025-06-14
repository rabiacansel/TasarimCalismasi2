import os
import pandas as pd

window_size = 384
sliding_window = 32

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in files:
    file_path = os.path.join(folder_path, file)    
    df = pd.read_csv(file_path)
    
    windows = []
    for start in range(0, len(df) - window_size + 1, sliding_window):
        window = df.iloc[start:start + window_size]
        windows.append(window)
    
        for i, window in enumerate(windows):
        output_file = os.path.join(output_folder, f"{file[:-4]}_window_{i}.csv")
        window.to_csv(output_file, index=False)

    print(f"{file} işlendi, {len(windows)} pencere oluşturuldu.")
