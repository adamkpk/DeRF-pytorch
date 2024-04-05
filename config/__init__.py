# TUNING

# Options: blender, llff
DATASET_NAME = 'llff'

# Options: 1, 2
TRAINING_ACCELERATION = 2

DATASET_SIZE_DICT = {
    'blender': tuple([400, 400]),
    'llff': tuple([504, 378])
}

DATASET_TEST_SIZE = {
    'blender': 200,
    'llff': 20
}

DATASET_EPOCHS = {
    'blender': 16,
    'llff': 30
}

DATASET_MILESTONES = {
    'blender': [2, 4, 8],
    'llff': [10, 20]
}
