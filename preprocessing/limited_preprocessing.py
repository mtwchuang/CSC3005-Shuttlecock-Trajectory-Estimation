import pandas as pd
import numpy as np
import os
import math

def preprocess_limited_data(input_file, output_file):
    """Preprocesses limited data for shuttlecock trajectory prediction."""
    print("\n🔄 Loading cleaned normal dataset...")
    data = pd.read_csv(input_file)
    print("✅ Dataset loaded.")
    # Keep only shuttlecock position columns (X, Y, Z),  observation number, and observation group number
    data = data[['SHUTTLECOCK POSITION IN AIR(X) metres',
                    'SHUTTLECOCK POSITION IN AIR(Y) metres',
                    'SHUTTLECOCK POSITION IN AIR(Z) metres',
                    'OBSERVATION GROUP NUMBER',
                    'OBSERVATION NUMBER']]
    print("✅ Filtered only limited shuttlecock features.")
    
    def assign_human_position(group):
        """Assigns human player position based on nearest round down of first shuttlecock observation"""
        group['HUMAN PLAYER POSITION(X) metres'] = math.floor(group.iloc[0]['SHUTTLECOCK POSITION IN AIR(X) metres'])     # Floor function, all shuttlecock first x are higher than human starting x
        group['HUMAN PLAYER POSITION(Y) metres'] = round(group.iloc[0]['SHUTTLECOCK POSITION IN AIR(Y) metres'])           # Round function, account for shuttlecocks first y that are slightly lower than human starting y
        return group
    
    print("\n🔄 Assigning human player X-Y-Z positions...")
    data = data.groupby('OBSERVATION GROUP NUMBER').apply(assign_human_position)
    # Ensures each row is assigned an observation group number in the csv file
    data = data.reset_index(drop=True)
    # Extract first Z-position in each group, assign Z = 1.6 if first Z < 2.24, else Z = 2.6
    first_z_values = data.groupby('OBSERVATION GROUP NUMBER')['SHUTTLECOCK POSITION IN AIR(Z) metres'].first()
    z_mapping = {group: (1.6 if z < 2.25 else 2.6) for group, z in first_z_values.items()}
    data['HUMAN PLAYER POSITION(Z) metres'] = data['OBSERVATION GROUP NUMBER'].map(z_mapping)
    print("✅ Human player X-Y-Z-position assigned.")

    def enrich_features(group):
        """Enriches dataset with additional features."""
        # Calculate distances between initial shuttlecock and human position
        x_displacement = group.iloc[0]['SHUTTLECOCK POSITION IN AIR(X) metres'] - group.iloc[0]['HUMAN PLAYER POSITION(X) metres']
        y_displacement = group.iloc[0]['SHUTTLECOCK POSITION IN AIR(Y) metres'] - group.iloc[0]['HUMAN PLAYER POSITION(Y) metres'] 
        z_displacement = group.iloc[0]['SHUTTLECOCK POSITION IN AIR(Z) metres'] - group.iloc[0]['HUMAN PLAYER POSITION(Z) metres']

        # Derive initial firing angle, TOA Trigonometry (θ = tan^-1(initial shuttlecock height (Z)/initial shuttlecock length travelled (X)))
        fir_angle_rad = math.atan(z_displacement/x_displacement)
        fir_angle_deg = math.degrees(fir_angle_rad)
        # Assign firing angle rounded to closest 10 to observation group
        group['INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)'] = round(fir_angle_deg, -1)

        # Derive inital slant angle, TOA Trigonometry (θ = tan^-1(initial shuttlecock breath (Y)/initial shuttlecock length (X)))
        slant_angle_rad = math.atan(y_displacement/x_displacement)
        slant_angle_deg = math.degrees(slant_angle_rad)
        # Assign slant angle rounded to closest integer to observation group
        group['INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)'] = round(slant_angle_deg)

        # Calculate the intial velocity, displacement divided by the time elapsed in the first observation step (10ms)
        time_elapsed = 0.01
        displacement = math.sqrt(x_displacement**2 + y_displacement**2 + z_displacement**2)
        # Assign initial velocity rounded to closest 10 to observation group
        group['INITIAL VELOCITY OF SHUTTLECOCK(m/s)'] = round(displacement/time_elapsed, -1)
        return group
    
    print("\n🔄 Enriching data features...")
    data = data.groupby('OBSERVATION GROUP NUMBER').apply(enrich_features)
    # Ensures each row is assigned an initial velocity, slant angle, firing angle in the csv file
    data = data.reset_index(drop=True)
    print("\n✅ Derived inital firing angle, slant angle, velocity.")

    # Save cleaned limited data
    data.to_csv(output_file, index=False)
    print(f"\n🎉 Limited Data Preprocessing complete! Saved to {output_file}")

