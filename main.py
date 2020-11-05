import json
from scripts.constants import active_models
from scripts.run_models import run

with open('data/hold_positions.json') as hold_file, open('data/moonboard_problems.json') as problems_file:
    hold_positions = json.load(hold_file)
    problems = json.load(problems_file)

for model_type in active_models:
    run(problems, hold_positions, model_type)
