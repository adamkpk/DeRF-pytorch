# TUNING ---------------------------------------------------------------------------------------------------------------

# Options: blender, llff
DATASET_NAME = 'blender'

# Options: blender -> lego     llff -> fern, flower
DATASET_TYPE = 'lego'

# Options: 1, 2
TRAINING_ACCELERATION = 1

# CONSTANTS ------------------------------------------------------------------------------------------------------------

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
