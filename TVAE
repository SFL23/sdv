# Import necessary modules
import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import TVAE

# Load demo data
data = load_tabular_demo('student_placements')

# Create an instance of TVAE model
model = TVAE()

# Fit the model to the data
model.fit(data)

# Generate synthetic data
num_rows = 200
new_data = model.sample(num_rows=num_rows)

# Save the synthetic data to a CSV file
new_data.to_csv('synthetic_data.csv', index=False)

# Display the first 5 samples of the synthetic data
print(new_data.head())
