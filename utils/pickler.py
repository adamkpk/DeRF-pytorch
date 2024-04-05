import pickle
from utils.loader import load_data
from config import DATASET_NAME

training_dataset, near, far = load_data('train')
testing_dataset, _, _ = load_data('test')

with open(f'./../data/{DATASET_NAME}_data.pkl', 'wb') as f:
    pickle.dump((training_dataset, testing_dataset, near, far), f)
