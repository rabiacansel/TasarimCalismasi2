import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(input_file_path)

columns_to_normalize = ["Minimum Value", "Maximum Value", "Median", "Mean", "Energy", "Standard Deviation"]
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.to_csv(output_csv_path, index=False)
