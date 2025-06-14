import os
import glob
import pandas as pd
from PyEMD import EMD

os.makedirs(output_folder, exist_ok=True)
channels = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for file in csv_files:
    df = pd.read_csv(file)
    file_name = os.path.basename(file)
    file_base = os.path.splitext(file_name)[0]  # Örn: S1S1_window_0

    all_imfs = []  

    for channel in channels:
        signal = df[channel].values
        emd = EMD()
        imfs = emd.emd(signal) 

        imf_df = pd.DataFrame(imfs.T)  
        
        imf_df.columns = [f"{channel}_IMF_{i+1}" for i in range(imf_df.shape[1])]
        
        all_imfs.append(imf_df)
    
    result_df = pd.concat(all_imfs, axis=1)

    new_filename = os.path.join(output_folder, f"{file_base}_emd.csv")
    
    result_df.to_csv(new_filename, index=False)

    print(f"{new_filename} kaydedildi.")
