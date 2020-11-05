from . import constants
import numpy as np
from numpy.random import shuffle
import pandas as pd
import math
import copy
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def convert_hold_orientation_to_angle(hold_pos):
    new_hold_pos = {}

    for key in hold_pos:
        pos = hold_pos[key]['position']
        j = ord(pos[0]) - 65
        i = int(pos[1:]) - 1

        orientation = None

        if hold_pos[key]['orientation'] == "NE":
            orientation = 1/8
        elif hold_pos[key]['orientation'] == "E":
            orientation = 2/8
        elif hold_pos[key]['orientation'] == "SE":
            orientation = 3/8
        elif hold_pos[key]['orientation'] == "S":
            orientation = 4/8
        elif hold_pos[key]['orientation'] == "SW":
            orientation = 5/8
        elif hold_pos[key]['orientation'] == "W":
            orientation = 6/8
        elif hold_pos[key]['orientation'] == "NW":
            orientation = 7/8
        elif hold_pos[key]['orientation'] == "N":
            orientation = 0

        new_hold_pos[(j, i)] = orientation

    return new_hold_pos


# Converts grades from scraped data grading system (which is based off Font grade system)
# to V-scale grade system
def convert_grades(problems):
    for key in problems:
        int_grade = problems[key]['grade']
        problems[key]['grade'] = constants.grade_map[int_grade]['v_scale']

    return problems


# Estimates hold difficulty based on the most frequent grade the hold appears in
def calculate_hold_difficulty(problems):
    grade_freq = np.array([0 for i in range(18)], dtype=float)
    hold_freq = {}

    for problem_id in problems:
        grade = problems[problem_id]['grade']
        grade_freq[grade] += 1

        for section in ['start', 'mid', 'end']:
            for hold_coord in problems[problem_id][section]:
                # Initialises hold frequency array for that hold if it does not already exist
                if hold_coord not in hold_freq:
                    hold_freq[hold_coord] = [0 for i in range(18)]

                hold_freq[hold_coord][grade] += 1

    hold_avg_grade = {}
    # freq_weight = [1.0 if grade_freq[i] > 30 else 0.6 for i in range(18)]
    freq_weight = [1.0 for i in range(18)]

    for hold in hold_freq:
        # Scales hold_freq down to [0, 1]
        hold_freq[hold] = np.divide(hold_freq[hold],  grade_freq, out=np.zeros(
            18, dtype=float), where=grade_freq != 0)

        # Multiples the frequency of each hold for a particular climbing grade by its associated weight
        # At the moment, the weight for each grade is 1. Past experiments with different weights have not boosted performacne
        hold_freq[hold] = np.multiply(hold_freq[hold], freq_weight)

        # Threshold is set to be the median frequency for that hold
        threshold = np.sum(hold_freq[hold]) / 2

        # TODO: Comment this section properly
        curr_sum = 0.0
        for i, freq in enumerate(hold_freq[hold]):
            curr_sum += freq

            if curr_sum >= threshold:
                hold_avg_grade[hold] = (
                    float(i) + 1-((curr_sum - threshold)/freq))/17
                break

    return hold_avg_grade


# Function which calculates movements to closest holds
# Move distances and angles are calculated by looking at the coords of the 2nd most recent move.
# This is done to be more reminiscent of actual climbing, by having one hand at a time move and with each hand moving
# one after the other (so every 2nd move).
def move_to_closest_holds(holds, beta):
    # In order to simplify things, we only move to a hold once
    used_hold_dict = {}

    while len(used_hold_dict) < len(holds):
        # Keeps track of the last 2 moves and the coord for each of the holds for that move
        # Each of these moves were used by a different hand
        last_move_for_hand_1 = beta[-1]
        last_coord_1 = last_move_for_hand_1[0]

        last_move_for_hand_2 = beta[-2]
        last_coord_2 = last_move_for_hand_2[0]

        # Keeps track of the current closest hold
        min_hold = None
        min_distance = None
        min_angle = None

        for i in range(len(holds)):
            # curr_hold = coord_to_key(holds[i])
            coords = holds[i]

            if coords not in used_hold_dict:
                # Horizontal distance is weighted less than vertical distance as it is usually much easier to move horizontally
                weighted_distance = math.sqrt(
                    pow(0.7 * (coords[0] - last_coord_2[0]), 2) + pow(coords[1] - last_coord_2[1], 2))
                raw_distance = math.sqrt(
                    pow(coords[0] - last_coord_2[0], 2) + pow(coords[1] - last_coord_2[1], 2))
                curr_angle = 0

                # Calculates angle of move. Angle of move is calculated in a way that makes sure the angle does not exceed 180
                if last_coord_2[1] <= coords[1]:
                    if last_coord_2[0] <= coords[0]:
                        curr_angle = math.degrees(
                            math.asin(abs(coords[0] - last_coord_2[0])/raw_distance))
                    else:
                        curr_angle = 90 - \
                            math.degrees(
                                math.acos(abs(coords[0] - last_coord_2[0])/raw_distance))
                elif last_coord_2[0] <= coords[0]:
                    curr_angle = 180 - \
                        math.degrees(
                            math.acos(abs(coords[0] - last_coord_2[0])/raw_distance))
                else:
                    curr_angle = 90 + \
                        math.degrees(
                            math.asin(abs(coords[0] - last_coord_2[0])/raw_distance))

                # Angle scaled between 0 and 1
                curr_angle /= 180

                if (min_hold is None or weighted_distance < min_distance):
                    min_distance = weighted_distance
                    min_hold = coords
                    min_angle = curr_angle

        # Temp fix to deal with climbs that have duplicate holds
        if min_hold is None:
            return beta

        beta.append((min_hold, min_distance, min_angle))
        used_hold_dict[min_hold] = True

    return beta


# Modified version of move function that is used when you need to get both hands on final hold
def move_to_match_on_final_hold(coords, beta):
    for i in range(2):
        last_move_for_hand = beta[-2]
        last_coord = last_move_for_hand[0]

        weighted_distance = math.sqrt(
            pow(0.7*(coords[0] - last_coord[0]), 2) + pow(coords[1] - last_coord[1], 2))
        raw_distance = math.sqrt(
            pow(coords[0] - last_coord[0], 2) + pow(coords[1] - last_coord[1], 2))
        curr_angle = 0

        # Calculates angle of move. Angle of move is calculated in a way that makes sure the angle does not exceed 180
        if last_coord[1] <= coords[1]:
            if last_coord[0] <= coords[0]:
                curr_angle = math.degrees(
                    math.asin(abs(coords[0] - last_coord[0])/raw_distance))
            else:
                curr_angle = 90 - \
                    math.degrees(
                        math.acos(abs(coords[0] - last_coord[0])/raw_distance))
        elif last_coord[0] <= coords[0]:
            curr_angle = 180 - \
                math.degrees(
                    math.acos(abs(coords[0] - last_coord[0])/raw_distance))
        else:
            curr_angle = 90 + \
                math.degrees(
                    math.asin(abs(coords[0] - last_coord[0])/raw_distance))

        # Angle scaled between 0 and 1
        curr_angle /= 180
        beta.append(
            (coords, weighted_distance, curr_angle))

    return beta


# Creates sequences of moves (betas) for each Moonboard problem
# Move sequences are generated by moving each hand to the closest hold until you eventually reach the top
# Individual moves are in the form (hold_key, weighted_distance_to_move_to_hold, angle_to_move_to_hold)
def generate_betas(problems, random_beta=False):
    betas = {}

    for key in problems:
        problem = problems[key]
        beta = []

        # if len(problem['start']) > 0:
        # Generate start moves
        if len(problem['start']) == 2:
            # Non-match start - means that both your hands start on different holds for the first move
            if problem['start'][1] == problem['start'][1][1]:
                # If both holds are on the same row, then we will first append the left-most hold
                if problem['start'][0][0] < problem['start'][1][0]:
                    beta.append((problem['start'][0], 0, 0))
                    beta.append((problem['start'][1], 0, 0))
                else:
                    beta.append((problem['start'][1], 0, 0))
                    beta.append((problem['start'][0], 0, 0))
            # Otherwise, append the lowest hold first
            elif problem['start'][0][1] < problem['start'][1][1]:
                beta.append((problem['start'][0], 0, 0))
                beta.append((problem['start'][1], 0, 0))
            else:
                beta.append((problem['start'][1], 0, 0))
                beta.append((problem['start'][0], 0, 0))
        else:
            # Match start hold - means that both hands are on the same start hold
            beta.append((problem['start'][0], 0, 0))
            beta.append((problem['start'][0], 0, 0))

        # Generate mid section moves
        if random_beta:
            beta = generate_random_moves(problem['mid'], beta)
        else:
            beta = move_to_closest_holds(problem['mid'], beta)

        # Generate end moves
        # If there are two end holds, then beta can be calculated normally since there is no violation of the
        # 'move to a hold once' rule
        if len(problem['end']) == 2:
            beta = move_to_closest_holds(problem['end'], beta)
        # Otherwise we will need to calculate moves differently since we would need to move both hands to the same hold
        else:
            beta = move_to_match_on_final_hold(problem['end'][0], beta)

        betas[key] = beta

    return betas


def generate_random_moves(holds, beta):
    shuffle(holds)

    for hold in holds:
        last_move_for_hand_1 = beta[-1]
        last_coord_1 = last_move_for_hand_1[0]

        last_move_for_hand_2 = beta[-2]
        last_coord_2 = last_move_for_hand_2[0]

        # Horizontal distance is weighted less than vertical distance as it is usually much easier to move horizontally
        weighted_distance = math.sqrt(
            pow(0.7 * (hold[0] - last_coord_2[0]), 2) + pow(hold[1] - last_coord_2[1], 2))
        raw_distance = math.sqrt(
            pow(hold[0] - last_coord_2[0], 2) + pow(hold[1] - last_coord_2[1], 2))
        
        curr_angle = 0

        # Calculates angle of move. Angle of move is calculated in a way that makes sure the angle does not exceed 180
        if last_coord_2[1] <= hold[1]:
            if last_coord_2[0] <= hold[0]:
                curr_angle = math.degrees(
                    math.asin(abs(hold[0] - last_coord_2[0])/raw_distance))
            else:
                curr_angle = 90 - \
                    math.degrees(
                        math.acos(abs(hold[0] - last_coord_2[0])/raw_distance))
        elif last_coord_2[0] <= hold[0]:
            curr_angle = 180 - \
                math.degrees(
                    math.acos(abs(hold[0] - last_coord_2[0])/raw_distance))
        else:
            curr_angle = 90 + \
                math.degrees(
                    math.asin(abs(hold[0] - last_coord_2[0])/raw_distance))
        
        # Angle scaled between 0 and 1
        curr_angle /= 180
        beta.append((hold, weighted_distance, curr_angle))

    return beta

def scale_values(dataset):
    for i, problem in enumerate(dataset):
        for j, move in enumerate(problem[1]):
            # new_move = [move[0] / max_moves, move[1], move[2]]
            scaled_move = [move[0] / constants.max_moves,
                           move[1], move[2], move[3]]
            dataset[i][1][j] = scaled_move

    return dataset


# Adds padding to move sequences so that all move sequence arrays are always of length 16
def pad_dataset(dataset):
    for i, problem in enumerate(dataset):
        if len(problem[1]) < constants.max_moves:
            padding = np.zeros((constants.max_moves - len(problem[1]), 4))
            dataset[i][1] = np.concatenate((padding, problem[1]), axis=0)

    return dataset


# Provides ordinal encoding for labels
def ordinal_encode(grades):
    encoded = []

    for grade in grades:
        num_ones = grade - 4
        encoded_grade = [0 for i in range(10)]

        for i in range(num_ones):
            encoded_grade[i] = 1

        encoded.append(encoded_grade)

    return encoded


def upsample_dataset(dataset, num_samples_per_grade):
    # Stores arrays of upsampled cases, with an array for each climbing grade
    upsampled_classes = []

    # Range represents the range of possible grades
    for i in range(4, 15):
        df_class = dataset[dataset.grade == i]

        upsampled_classes.append(resample(df_class, replace=True,
                                          n_samples=num_samples_per_grade, random_state=10))

    dataset_upsampled = upsampled_classes[0]

    for i in range(1, len(upsampled_classes)):
        dataset_upsampled = pd.concat(
            [dataset_upsampled, upsampled_classes[i]], axis=0)

    return dataset_upsampled


def upsample_and_split(dataset, x_col_name):
    train, test = train_test_split(
        dataset, test_size=0.2, random_state=10)
    train, val = train_test_split(train, test_size=0.25, random_state=10)

    max_grade_frequency = train['grade'].value_counts().max()
    train_upsampled = upsample_dataset(
        train, num_samples_per_grade=max_grade_frequency)

    x_train = np.stack(np.array(train_upsampled[x_col_name]), axis=0)
    y_train = np.array(ordinal_encode(train_upsampled['grade']))

    x_val = np.stack(np.array(val[x_col_name]), axis=0)
    y_val = np.array(ordinal_encode(val['grade']))

    x_test = np.stack(np.array(test[x_col_name]), axis=0)

    # Don't ordinal encode y_test labels, otherwise they won't work properly with evaluation metrics
    y_test = test['grade']

    return x_train, y_train, x_val, y_val, x_test, y_test


# Uses problems, generated betas for each problem, and hold information to create a dataset that can be passed into the LSTM
# Each problem is represented by an array in the following format [problem ID, beta, grade]
def create_lstm_dataset(problems, betas, hold_avg_grade, hold_positions):
    dataset = []

    for problem_id in problems:
        if problem_id in betas:
            processed_problem = []
            processed_problem.append(problem_id)

            beta = betas[problem_id]

            # Represents a beta as an array of moves with the following information for each move:
            # [hold difficulty, distance to move to hold, orientation of hold, the angle of the move direction]
            processed_beta = []

            for i, move in enumerate(beta):
                hold_diff = hold_avg_grade[beta[i][0]]
                distance = beta[i][1]
                orientation = hold_positions[beta[i][0]]
                move_angle = beta[i][2]
                processed_beta.append(
                    [hold_diff, distance, orientation, move_angle])

            processed_problem.append(processed_beta)
            processed_problem.append(problems[problem_id]['grade'])
            dataset.append(processed_problem)

    return dataset


def format_problems(problems):
    problems = convert_grades(problems)

    # Converts holds to tuples so that they can be used as keys for a dictionary
    for problem_id in problems:
        for section in ['start', 'mid', 'end']:
            problems[problem_id][section] = list(
                map(tuple, problems[problem_id][section]))

    return problems


# Preprocessing steps for LSTM approach
def preprocess_lstm(problems, hold_positions, random_beta=False):
    problems = copy.deepcopy(problems)
    problems = format_problems(problems)
    betas = generate_betas(problems, random_beta=random_beta)

    hold_avg_grade = calculate_hold_difficulty(problems)
    hold_positions = convert_hold_orientation_to_angle(hold_positions)

    dataset = create_lstm_dataset(
        problems, betas, hold_avg_grade, hold_positions)
    dataset_scaled = scale_values(dataset)
    dataset_padded = pad_dataset(dataset_scaled)
    dataset_pd = pd.DataFrame(dataset_padded, columns=['id', 'moves', 'grade'])

    return upsample_and_split(dataset_pd, x_col_name='moves')


# Returns a boolean hold map where a hold that is being used in a Moonboard problem is marked by a 1
def problems_to_hold_map(problems):
    hold_map = []

    for key in problems:
        curr_map = [[0 for i in range(11)] for i in range(18)]

        for section in ['start', 'mid', 'end']:
            for hold in problems[key][section]:
                curr_map[hold[1]][hold[0]] = 1

        hold_map.append(curr_map)

    return hold_map


# Returns an array of grades in the V-scale for each Moonboard problem
def get_grades(problems):
    grades = []

    for key in problems:
        int_grade = problems[key]['grade']
        grades.append(constants.grade_map[int_grade]['v_scale'])

    return grades


def preprocess_cnn(problems):
    hold_map = problems_to_hold_map(problems)
    problem_ids = [key for key in problems]
    grades = get_grades(problems)

    problem_dict = {'ids': problem_ids, 'hold_map': hold_map, 'grade': grades}
    dataset = pd.DataFrame(problem_dict)

    return upsample_and_split(dataset, x_col_name='hold_map')
