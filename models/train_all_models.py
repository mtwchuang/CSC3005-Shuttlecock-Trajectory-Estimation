import train_decision_tree as tdt
import train_random_forest as trf
import train_extra_trees as tet
import train_lstm_rnn as tlr

# Constants
FILEPATH_NORMAL = "data/cleaned/normal_data.csv"
FILEPATH_LIMITED = "data/cleaned/limited_data.csv"

BEST_DT_DEPTH = 4
DT_OUTPATH = "models/trained_models/decision_tree_model.pkl"

BEST_RF_DEPTH = 15
BEST_RF_NESTIMATORS = 200
RF_OUTPATH_NORMAL = "models/trained_models/random_forest_model_normal.pkl"
RF_OUTPATH_LIMITED = "models/trained_models/random_forest_model_limited.pkl"

BEST_ET_DEPTH = 20
BEST_ET_NESTIMATORS = 200
ET_OUTPATH_NORMAL = "models/trained_models/extra_trees_model_normal.pkl"
ET_OUTPATH_LIMITED = "models/trained_models/extra_trees_model_limited.pkl"

LSTM_UNITS_1_NORMAL, LSTM_UNITS_2_NORMAL = 128, 128
LSTM_UNITS_1_LIMITED, LSTM_UNITS_2_LIMITED = 256, 128
LSTM_BATCH_SIZE = 32
LSTM_OUTPATH_NORMAL = "models/trained_models/lstm_rnn_model_normal.keras"
LSTM_OUTPATH_LIMITED = "models/trained_models/lstm_rnn_model_limited.keras"

def train_models():
    # Train Decision Tree model for predicting observation sequences
    print("\n🔄 Training Decision Trees model using Normal Data...")
    tdt.train_observation_sequence_predictor(FILEPATH_NORMAL, BEST_DT_DEPTH, DT_OUTPATH)
    print("\n✅ Decision Tree model for predicting number of sequences saved!")
    # Train Random Forest Regressor for predicting shuttlecock positions 
    print("\n🔄 Training Random Forest Regressor model using Normal Data...")
    trf.train_random_forest_model(FILEPATH_NORMAL, BEST_RF_DEPTH, BEST_RF_NESTIMATORS, RF_OUTPATH_NORMAL)
    print("\n✅ Random Forest model for Normal Data saved!")
    print("\n🔄 Training Random Forest model for Limited Data...")
    trf.train_random_forest_model(FILEPATH_LIMITED, BEST_RF_DEPTH, BEST_RF_NESTIMATORS, RF_OUTPATH_LIMITED)
    print("\n✅ Random Forest model for Limited Data saved!")
    # Train Extra Tree Regressor for predicting shuttlecock positions
    print("\n🔄 Training Extra Trees model for Normal Data...")
    tet.train_extra_tree_model(FILEPATH_NORMAL, BEST_ET_DEPTH, BEST_ET_NESTIMATORS, ET_OUTPATH_NORMAL)
    print("\n✅ Extra Trees model for Normal Data saved!")
    print("\n🔄 Training Extra Trees model for Limited Data...")
    tet.train_extra_tree_model(FILEPATH_LIMITED, BEST_ET_DEPTH, BEST_ET_NESTIMATORS, ET_OUTPATH_LIMITED)
    print("\n✅ Extra Trees model for Limited Data saved!")
    # Train LSTM RNN models for predicting shuttlecock positions
    tlr.train_lstm_rnn_model(FILEPATH_NORMAL, LSTM_UNITS_1_NORMAL, LSTM_UNITS_2_NORMAL, LSTM_BATCH_SIZE, LSTM_OUTPATH_NORMAL)
    tlr.train_lstm_rnn_model(FILEPATH_LIMITED, LSTM_UNITS_1_LIMITED, LSTM_UNITS_2_LIMITED, LSTM_BATCH_SIZE, LSTM_OUTPATH_LIMITED)




