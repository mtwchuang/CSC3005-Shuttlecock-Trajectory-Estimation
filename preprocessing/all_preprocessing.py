import normal_preprocessing as npp
import limited_preprocessing as lpp

# Constants
RAW_SERVING_FILE = 'data/raw/badmintondata.csv'
RAW_RALLYING_FILE = 'data/raw/badmintondata2.csv'

NORMAL_OUTPUT = 'data/cleaned/normal_data.csv'
LIMITED_OUTPUT = "data/cleaned/limited_data.csv"

def preprocess_data():
    # Preprocess normal data
    npp.preprocess_normal_data(RAW_SERVING_FILE, RAW_RALLYING_FILE, NORMAL_OUTPUT)
    lpp.preprocess_limited_data(NORMAL_OUTPUT, LIMITED_OUTPUT)
