import models.train_all_models as mtm
import preprocessing.all_preprocessing as pp

# Run this script before running any of the other notebooks
if __name__ == "__main__":
    """Set up the data and train the models"""
    pp.preprocess_data()    # Will preprocess data and save to 'data/cleaned' directory, Warning: This will take a while (45 - 60 minutes)
    mtm.train_models()      # Will train all the models and save to 'models/trained' directory, Warning: This will take a while (5 - 10 minutes)