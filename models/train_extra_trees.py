import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import os

def train_extra_tree_model(filepath, best_depth, best_n_estimators, outpath):
    """Reusable function to train an Extra Trees model."""
    data = pd.read_csv(filepath)
    # Define X features
    x_features = data[['HUMAN PLAYER POSITION(X) metres', 
        'HUMAN PLAYER POSITION(Y) metres', 
        'HUMAN PLAYER POSITION(Z) metres', 
        'INITIAL VELOCITY OF SHUTTLECOCK(m/s)', 
        'INITIAL SHUTTLECOCK FIRING ANGLE(DEGREE)', 
        'INITIAL SHUTTLECOCK SLANT ANGLE TO SIDELINE(DEGREE)',
        'OBSERVATION NUMBER']]
    # Define Y features
    y_features = data[['SHUTTLECOCK POSITION IN AIR(X) metres', 
        'SHUTTLECOCK POSITION IN AIR(Y) metres', 
        'SHUTTLECOCK POSITION IN AIR(Z) metres']]
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_features, test_size=0.2, random_state=42)
    # Train final model with best hyperparameters from Jupyter Notebook
    best_rf_model = ExtraTreesRegressor(n_estimators=best_n_estimators, max_depth=best_depth, random_state=42)
    best_rf_model.fit(x_train, y_train)
    # Save the trained model
    os.makedirs('models/trained_models', exist_ok=True)
    joblib.dump(best_rf_model, outpath)