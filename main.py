import json
from scripts.preprocess import *
from scripts.run_models import *

with open('data/hold_positions.json')as hold_file, open('data/moonboard_problems.json') as problems_file:
    hold_positions = json.load(hold_file)
    problems = json.load(problems_file)

run_lstm(*preprocess_lstm(problems, hold_positions))
# run_cnn(*preprocess_cnn(problems))