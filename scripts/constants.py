# Max number of moves for a generated beta/move sequence
max_moves = 16

# None: no WandB logging
# CNN: CNN logging
# LSTM: LSTM logging
wandb_mode = None

# List of all model types which will be run on program execution
active_models = [
    "LSTM",
    "LSTM_RANDOM",
    "CNN"
]

# Mappings for each of the grading systems
grade_map = {
    2: {'font_scale': '6B', 'v_scale': 4},
    3: {'font_scale': '6B+', 'v_scale': 4},
    4: {'font_scale': '6C', 'v_scale': 5},
    5: {'font_scale': '6C+', 'v_scale': 5},
    6: {'font_scale': '7A', 'v_scale': 6},
    7: {'font_scale': '7A+', 'v_scale': 7},
    8: {'font_scale': '7B', 'v_scale': 8},
    9: {'font_scale': '7B+', 'v_scale': 8},
    10: {'font_scale': '7C', 'v_scale': 9},
    11: {'font_scale': '7C+', 'v_scale': 10},
    12: {'font_scale': '8A', 'v_scale': 11},
    13: {'font_scale': '8A+', 'v_scale': 12},
    14: {'font_scale': '8B', 'v_scale': 13},
    15: {'font_scale': '8B+', 'v_scale': 14}
}

# hyperparameters_lstm = dict(
#     epochs = 183,
#     batch_size = 16,
#     nodes_1 = 46,
#     nodes_2 = 20,
# )

hyperparameters_lstm = dict(
    epochs = 58,
    batch_size = 157,
    nodes_1 = 59,
    nodes_2 = 17,
)

