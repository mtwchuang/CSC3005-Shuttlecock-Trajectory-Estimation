import pandas as pd
import numpy as np
import os

def preprocess_normal_data(input_file1, input_file2, output_file):
    """Preprocesses normal data for shuttlecock trajectory prediction."""
    print("\n🔄 Loading datasets...")
    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)
    print("✅ Datasets loaded.")

    # Tag data sources
    df1['SOURCE'] = 'SERVING'
    df2['SOURCE'] = 'RALLYING'

    # Combine datasets
    data = pd.concat([df1, df2], ignore_index=True)
    print("✅ Datasets loaded and combined.")
    
    # Initialize variables
    observation_group_num = 0
    observation_num = 0
    is_group = False

    print("\n🔄 Assigning observation group numbers...")
    for index, row in data.iterrows():
        # Row is blank, is not part of any observation
        if row['HUMAN PLAYER POSITION(X) metres'] == 0:
            if is_group:
                is_group = False
            # Remove blank row after handling it
            data.drop(index, inplace=True)
        # Row is part of an observation
        elif row['HUMAN PLAYER POSITION(X) metres'] == 4:
            # If row is known to be part of a group, increment observation number
            if is_group:
                observation_num += 1
            # If row is not known to be part of a group, set isGroup to true, increment observation group number, and set observation number to 1
            else:
                is_group = True
                observation_group_num += 1
                observation_num = 1
            # Label observation group number and observation number to data
            data.at[index, 'OBSERVATION NUMBER'] = observation_num
            data.at[index, 'OBSERVATION GROUP NUMBER'] = observation_group_num
    print(f"✅ Assigned {data['OBSERVATION GROUP NUMBER'].nunique()} unique observation groups.")

    # Assign human player positions Z, 1.6 for serving data, 2.6 for the rest
    data['HUMAN PLAYER POSITION(Z) metres'] = data['SOURCE'].map({'SERVING': 1.6, 'RALLYING': 2.6})
    data = data.drop(columns=['SOURCE'])
    print("✅ Human player Z-position assigned.")

    # Renaming data columns
    data = data.rename(columns={'SHUTTLECOCK POSITIION IN AIR(X ) metres': 'SHUTTLECOCK POSITION IN AIR(X) metres'})
    data = data.rename(columns={'SHUTTLECOCK POSITIION IN AIR(Y) metres': 'SHUTTLECOCK POSITION IN AIR(Y) metres'})
    data = data.rename(columns={'SHUTTLECOCK POSITIION IN AIR(Z) metres': 'SHUTTLECOCK POSITION IN AIR(Z) metres'})
    data = data.rename(columns={'SHUTTELCOCK SLANT ANGLE TO SIDELINE(DEGREE)': 'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)'})
    data = data.rename(columns={'INITIAL SHUTTELCOCK FIRING ANGLE (DEGREE)': 'INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)'})

    # Ensure the 'data/cleaned' directory exists, create if it doesn't
    os.makedirs('data/cleaned', exist_ok=True)
    # Save cleaned data
    data.to_csv(output_file, index=False)
    print(f"\n🎉 Normal Data Preprocessing complete! Saved to {output_file}")