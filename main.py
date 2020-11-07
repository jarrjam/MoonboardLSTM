import json
from scripts.constants import active_models
from scripts.run_models import run
from scripts.log import log_output
from scripts.metrics import final_results_table

with open('data/hold_positions.json') as hold_file, open('data/moonboard_problems.json') as problems_file:
    hold_positions = json.load(hold_file)
    problems = json.load(problems_file)

results = []

for model_type in active_models:
    results.append(run(problems, hold_positions, model_type))

log_output("ALL", "Final results from experiments:\n\n" + final_results_table(results))