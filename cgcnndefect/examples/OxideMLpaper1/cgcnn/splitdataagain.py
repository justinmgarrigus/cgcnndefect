import os
import pandas as pd
import numpy as np

# Define file paths and settings
base_path = "~/dgnn/cgcnndefect/examples/OxideMLpaper1/cgcnn"
oxygen_file = os.path.join(base_path, "id_prop.csv.oxygen")
metal_file = os.path.join(base_path, "id_prop.csv.metal")
output_template = "id_prop.csv.{category}{pct}pct"

# Define categories and percentages
categories = ["oxygen", "metal"]
percentages = range(10, 101, 10)

def split_data(file_path, category):
    # Load the data
    data = pd.read_csv(file_path, header=None)

    # Shuffle data for randomness
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    for pct in percentages:
        # Calculate the number of rows for the current percentage
        rows = int(len(data) * (pct / 100.0))

        # Slice the data
        subset = data.iloc[:rows]

        # Define output file path
        output_file = os.path.join(base_path, output_template.format(category=category, pct=pct))

        # Save the subset to the output file
        subset.to_csv(output_file, index=False, header=False)

# Split both datasets
split_data(oxygen_file, "oxygen")
split_data(metal_file, "metal")

print("Data splitting complete. Files saved in 10% to 100% subsets for oxygen and metal.")
