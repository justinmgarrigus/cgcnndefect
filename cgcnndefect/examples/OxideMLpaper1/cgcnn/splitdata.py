import pandas as pd
import random
import os

# Path to the input file
file_path = "/home/tjbouchard/dgnn/cgcnndefect/examples/OxideMLpaper1/pretrain/id_prop.csv.pretrain"

# Load the dataset
data = pd.read_csv(file_path, header=None, names=["id", "value"])

# Identify Oxygen and Metal entries
oxygen_data = data[data["id"].str.contains("-O")]
metal_data = data[~data["id"].str.contains("-O")]

# Total counts of Oxygen and Metal
total_oxygen = len(oxygen_data)
total_metal = len(metal_data)

# Validate ratio

# Define percentages
percentages = [10,20,30,40,50,60,70,80,90]

# Output directory (same as input file directory)
output_dir = os.path.dirname(file_path)

# Split and save datasets
for pct in percentages:
    oxygen_count = round(total_oxygen * (pct / 100))
    metal_count = round(total_metal * (pct / 100))
    
    # Randomly sample Oxygen and Metal entries
    sampled_oxygen = oxygen_data.sample(oxygen_count, random_state=2)
    sampled_metal = metal_data.sample(metal_count, random_state=2)
    
    # Combine and shuffle the dataset
    subset = pd.concat([sampled_oxygen, sampled_metal]).sample(frac=1, random_state=42)
    
    # Save to file
    output_file = os.path.join(output_dir, f"id_prop.csv.{pct}pct")
    subset.to_csv(output_file, index=False, header=False)
    print(f"Saved {pct}% dataset to {output_file}")
