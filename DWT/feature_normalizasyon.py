import pandas as pd
import numpy as np
import ast
import os
import glob
from scipy.stats import skew

def compute_entropy(coeffs):
    coeffs = np.array(coeffs)
    abs_coeffs = np.abs(coeffs)
    total = abs_coeffs.sum()
    if total == 0:
        return 0
    probs = abs_coeffs / total
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log2(probs))
    return ent
file_paths = glob.glob(os.path.join(data_dir, "*.csv"))

results = []

for i, file_path in enumerate(file_paths):
    print(f"[{i+1}/{len(file_paths)}] İşleniyor: {file_path}")
    
    file_name = os.path.basename(file_path)
    parts = file_name.split("_")
    subject_info = parts[1]                
    window_info = parts[-1].split(".")[0]   ü
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Dosya okunamadı: {file_path} - Hata: {e}")
        continue
    
    features_list = []
    
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
    
    features_df = pd.DataFrame(features_list)
    
    features_df["Relative Energy"] = np.nan
    for channel, group in features_df.groupby("Channel"):
        total_energy = group["Energy"].sum()
        if total_energy != 0:
            features_df.loc[group.index, "Relative Energy"] = group["Energy"] / total_energy
        else:
            features_df.loc[group.index, "Relative Energy"] = np.nan
    
    features_df = features_df[["Subject", "Window", "Channel", "Level", 
                               "Minimum Value", "Maximum Value", "Median", "Mean",
                               "Energy", "Standard Deviation", "Variance",
                               "Skewness", "Entropy", "Relative Energy"]]
    
    results.append(features_df)
    
    blank_row = pd.DataFrame([[""] * len(features_df.columns)], columns=features_df.columns)
    results.append(blank_row)

if results:
    results = results[:-1]

final_df = pd.concat(results, ignore_index=True)

numeric_cols = ["Minimum Value", "Maximum Value", "Median", "Mean", "Energy", 
                "Standard Deviation", "Variance", "Skewness", "Entropy", "Relative Energy"]

block_indices = []
current_block = []

for idx, row in final_df.iterrows():
    if row["Subject"] == "":
        if current_block:
            block_indices.append(current_block)
            current_block = []
    else:
        current_block.append(idx)
if current_block:
    block_indices.append(current_block)

for block in block_indices:
    block_data = final_df.loc[block, numeric_cols].astype(float)
    for col in numeric_cols:
        col_min = block_data[col].min()
        col_max = block_data[col].max()
        if col_max != col_min:
            normalized = 2 * (block_data[col] - col_min) / (col_max - col_min) - 1
        else:
            normalized = 0  
        final_df.loc[block, col] = normalized

final_df.to_csv(output_path, index=False)

print(f"CSV dosyası oluşturuldu: {output_path}")
