# Configuration settings for Plant Disease Detection System

# Sensitivity level for detection
SENSITIVITY_LEVEL = 'high'

# Threshold for disease probability
DISEASE_PROBABILITY_THRESHOLD = 0.7

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'username',
    'password': 'password',
    'database': 'plant_disease_db'
}

# Model configuration
MODEL_PATH = 'models/disease_model.h5'
