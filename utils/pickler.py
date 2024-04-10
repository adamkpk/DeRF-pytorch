import pickle
from utils.loader import load_data, compute_bounding_box
from config import DATASET_NAME, DATASET_TYPE

training_dataset, near, far = load_data('train')
testing_dataset, _, _ = load_data('test')

# Note: NeRF does not make use of the bounding_box, only DeRF, and only for initializing the Voronoi head positions
bounding_box = compute_bounding_box(training_dataset[:, :6], near, far)

with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'wb') as f:
    pickle.dump((training_dataset, testing_dataset, near, far, bounding_box), f)
