import os
import pandas as pd

# Path to the directory containing the datasets
directory_path = "/home/tjbouchard/dgnn/cgcnndefect/examples/OxideMLpaper1/cgcnn"

# List of percentages
percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# Expected Oxygen-to-Metal ratio
expected_oxygen = 795
expected_metal = 686
expected_ratio = expected_oxygen / expected_metal

# Iterate through each file and verify the ratio
for pct in percentages:
    file_path = os.path.join(directory_path, f"id_prop.csv.{pct}pct")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, names=["id", "value"])
        oxygen_count = data[data["id"].str.contains("-O")].shape[0]
        metal_count = data[~data["id"].str.contains("-O")].shape[0]
        
        if metal_count > 0:  # Avoid division by zero
            actual_ratio = oxygen_count / metal_count
        else:
            actual_ratio = 0  # In case of no metal data, ratio is undefined
        
        print(f"{pct}% Dataset: Oxygen = {oxygen_count}, Metal = {metal_count}, Ratio = {actual_ratio:.4f}")
        
        if abs(actual_ratio - expected_ratio) < 1e-2:
            print(f"Ratio is correct for {pct}% dataset.")
        else:
            print(f"Ratio is incorrect for {pct}% dataset.")
    else:
        print(f"File not found: {file_path}")
