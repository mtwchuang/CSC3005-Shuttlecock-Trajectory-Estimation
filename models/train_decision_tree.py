import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os

# data mining and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import learning_curve
import joblib

# visualization
import matplotlib.pyplot as plt

def train_observation_sequence_predictor(filepath, best_depth, outpath):
    data = pd.read_csv(filepath)
    # Drop all rows except last observation sequence, of each observation group
    data = data.drop_duplicates(subset='OBSERVATION GROUP NUMBER', keep='last')
    # Define X features
    x_feature = data[['HUMAN PLAYER POSITION(X) metres', 
        'HUMAN PLAYER POSITION(Y) metres', 
        'HUMAN PLAYER POSITION(Z) metres', 
        'INITIAL VELOCITY OF SHUTTLECOCK(m/s)', 
        'INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)', 
        'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)']]
    # Define Y feature
    y_feature = data['OBSERVATION NUMBER']
    # Train-Test splits
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_feature, test_size=0.2, random_state=42)

    # Train and fit the Decision Tree model using x and y training sets
    decision_tree_model = DecisionTreeClassifier(max_depth=best_depth)
    decision_tree_model = decision_tree_model.fit(x_train, y_train)

    # Ensure the 'trained_models' directory exists
    os.makedirs('models/trained_models', exist_ok=True)
    # Save the trained model
    joblib.dump(decision_tree_model, outpath)
