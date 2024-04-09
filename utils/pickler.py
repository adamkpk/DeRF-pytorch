import pickle
from utils.loader import load_data
from config import DATASET_NAME, DATASET_TYPE

training_dataset, near, far = load_data('train')
testing_dataset, _, _ = load_data('test')

with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'wb') as f:
    pickle.dump((training_dataset, testing_dataset, near, far), f)
