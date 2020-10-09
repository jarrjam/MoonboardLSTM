import json
from scripts.preprocess import *

with open('data/hold_positions.json')as hold_file, open('data/moonboard_problems.json') as problems_file:
    hold_positions = json.load(hold_file)
    problems = json.load(problems_file)

preprocess(problems, hold_positions, 'lstm')
    