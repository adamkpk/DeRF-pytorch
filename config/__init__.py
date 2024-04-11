# TUNING ---------------------------------------------------------------------------------------------------------------

# Options: cpu, cuda   (others not guaranteed)
DEVICE = 'cuda'

# Options: blender, llff
DATASET_NAME = 'blender'

# Options: blender -> lego     llff -> fern, flower
DATASET_TYPE = 'lego'

# Options: 1, 2
TRAINING_ACCELERATION = 1

# Options: True -> tests all epochs, False -> tests only the last epoch
TEST_ALL_EPOCHS = False

HEAD_COUNT = 32

# CONSTANTS ------------------------------------------------------------------------------------------------------------

BINS_COARSE = 64
BINS_FINE = 192

HIDDEN_UNITS = {
    'full': 256,
    'head': 64
}

DATASET_SIZE_DICT = {
    'blender': tuple([400, 400]),
    'llff': tuple([504, 378])
}

DATASET_TEST_SIZE = {
    'blender': {
        'lego': 200
    },
    'llff': {
        'fern': 20,
        'flower': 34
    }
}

DATASET_EPOCHS = {
    'blender': 16,
    'llff': 30
}

DATASET_MILESTONES = {
    'blender': [2, 4, 8],
    'llff': [10, 20]
}

# Note: No scheduler is used when coarse training, so no milestones are defined for coarse
DATASET_EPOCHS_COARSE = {
    'blender': 1,
    'llff': 10
}

DATASET_WHITEBG_EQUALIZATION = {
    'blender': True,
    'llff': False
}
