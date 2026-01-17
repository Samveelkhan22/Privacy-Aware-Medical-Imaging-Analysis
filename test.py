import pandas as pd
import random
import numpy as np

# Read the location data from an Excel file
location_data = pd.read_excel(r'D:\Users\J.I Traders\Desktop\Projects\python_epsilon\Citiesp.xlsx')

# Split the 'Geographic Area' column into city and state if necessary
if 'Geographic Area' in location_data.columns:
    location_data[['City', 'State']] = location_data['Geographic Area'].str.split(',', expand=True)
else:
    # Assuming 'City' and 'State' are separate columns in the Excel file
    location_data['City'] = location_data['City'].str.strip()
    location_data['State'] = location_data['State'].str.strip()

# View the location data (optional)
print(location_data.head())

# Sample patient data
patient_data = [
    {'patient_id': 1, 'value': None, 'location': None},
    {'patient_id': 2, 'value': None, 'location': None},
    # Add more patient records as needed
]

# Fill missing values and assign random locations (city and state)
for patient in patient_data:
    # Fill synthetic 'value' if missing
    if patient['value'] is None:
        patient['value'] = random.randint(50, 100)
    
    # Assign random city and state
    random_location = location_data.sample(n=1).iloc[0]
    patient['location'] = f"{random_location['City'].strip()}, {random_location['State'].strip()}"

# Write the 'fill.txt' file with synthetic data
with open('fill.txt', 'w') as f:
    for patient in patient_data:
        f.write(f"{patient['patient_id']}, {patient['value']}, {patient['location']}\n")

print("fill.txt created with synthetic data.")

def laplace_mechanism(value, sensitivity, epsilon):
    # Apply Laplace noise
    noise = np.random.laplace(0, sensitivity / epsilon)
    return value + noise

# Assume sensitivity is 1 for location selection
epsilon = 0.1  # You can change this value as needed

# Apply noise to locations
noisy_patient_data = []

for patient in patient_data:
    # Select a new random location based on Laplace mechanism
    noisy_location_index = int(laplace_mechanism(random.randint(0, len(location_data)-1), 1, epsilon))
    noisy_location_index = max(0, min(len(location_data)-1, noisy_location_index))  # Ensure index is within bounds
    
    noisy_location = location_data.iloc[noisy_location_index]
    noisy_patient_data.append({
        'patient_id': patient['patient_id'],
        'value': patient['value'],
        'location': f"{noisy_location['City'].strip()}, {noisy_location['State'].strip()}"
    })

# Write the 'noise.txt' file with noisy data
with open('noise.txt', 'w') as f:
    for patient in noisy_patient_data:
        f.write(f"{patient['patient_id']}, {patient['value']}, {patient['location']}\n")

print("noise.txt created with noisy data.")
