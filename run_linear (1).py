import pickle
import numpy as np
import pandas as pd
import warnings
# to ignore warnings
warnings.filterwarnings("ignore")
try:
    with open("linear.pkl", "rb") as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
except Exception as e:
    print(f"Error loading linear.pkl: {e}")
    exit()
#  Full 26 features for Scaler 
all_26_features = [
    'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'year_built',
    'property_age', 'renovation_status', 'lot_size', 'distance_city_km',
    'neighborhood_score', 'crime_rate', 'school_rating', 'hospital_km',
    'shopping_km', 'transport_score', 'construction_quality',
    'energy_efficiency', 'water_supply', 'electricity_supply',
    'internet_score', 'green_space', 'flood_risk', 'noise_level',
    'property_type_House', 'property_type_Villa', 'parking_1.0'
]
#  15 features for linear regression( 15 is the number of features selected on which our model is trained)
feature_names = [
    'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'renovation_status',
    'distance_city_km', 'neighborhood_score', 'crime_rate', 'shopping_km',
    'transport_score', 'construction_quality', 'water_supply',
    'green_space', 'property_type_Villa', 'parking_1.0'
]
# user input
print(f" House Price Predictor ")
print(f"Please enter the {len(feature_names)} values separated by spaces.")
print("Keywords allowed: 'Villa', 'House', 'Yes' (for parking,rennovation_status), 'No' (for parking,rennovation_status)")
values = input("\nEnter values: ")
input_list = values.split()
if len(input_list) != 15:
    print(f"\n Expected 15 values, but you entered {len(input_list)}")
else:
    try:
        processed_inputs = []
        for i, val in enumerate(input_list):
            clean_val = val.lower().strip()
            
            # encoding for categorical columns
            if clean_val in ['villa', 'yes']:
                processed_inputs.append(1.0)
            elif clean_val in ['house', 'no']:
                processed_inputs.append(0.0)
            else:
                processed_inputs.append(float(clean_val))
        # Create a dummy row of 26 zeros to satisfy the Scaler
        full_row = np.zeros((1, 26))
        for i, name in enumerate(feature_names):
            idx = all_26_features.index(name)
            full_row[0, idx] = processed_inputs[i]
        full_df = pd.DataFrame(full_row, columns=all_26_features)
        #  Scaling the full 26-feature row
        scaled_full = scaler.transform(full_df)
        # model trained only on selected features so extract only those in correct order
        model_indices = [all_26_features.index(n) for n in feature_names]
        scaled_for_model = scaled_full[:, model_indices]
        
        prediction = model.predict(scaled_for_model)
        
        print(f" PREDICTED PRICE: {prediction[0]:,.2f} Rupees")

    except ValueError as v:
        print(f" Data Error: Could not convert input to a number. ({v})")
    except Exception as e:
        print(f" Unexpected Error: {e}")