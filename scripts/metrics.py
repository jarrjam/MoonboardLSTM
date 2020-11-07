import numpy as np
import pandas as pd
from .constants import active_models


# Macro-averaged MSE used for ordinal regression when there is imbalanced classes
def macro_mse(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    # Gets label and label counts
    unique, counts = np.unique(y_true, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    mse = 0

    for i in range(len(y_true)):
        mse += ((y_true[i]-y_pred[i])**2)/counts_dict[y_true[i]]

    return mse / len(unique)


# Generates a report containing MSE and MAE scores for each climbing grade
def ordinal_evaluation_report(y_true, y_pred, labels=None, return_mse=False):
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    # Gets label and label counts
    unique, counts = np.unique(y_true, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    if labels is None:
        labels = unique

    class_scores = {label: {'mse': 0, 'mae': 0} for label in labels}

    total_mse = 0
    total_mae = 0

    # These scores are based off grades with at least 100 samples in the test dataset
    subset_mse = 0
    subset_mae = 0
    subset_support = 0
    subset_grades = {}

    for i in range(len(y_true)):
        curr_label = y_true[i]

        if counts[curr_label - 4] >= 30:
            subset_grades[curr_label] = True

        if curr_label in class_scores:
            prediction_error = abs(curr_label-y_pred[i])
            mse = (prediction_error ** 2)/counts_dict[curr_label]
            mae = prediction_error / counts_dict[curr_label]

            class_scores[curr_label]['mae'] += mae
            total_mae += mae

            class_scores[curr_label]['mse'] += mse
            total_mse += mse

            if curr_label in subset_grades:
                subset_support += 1
                subset_mse += mse
                subset_mae += mae

    class_scores['subset'] = subset_mse/len(subset_grades)
    class_scores['total'] = total_mse/len(unique)

    headers = ['Grade', 'MSE', 'MAE', 'Support']
    data = [headers] + \
        list(zip(labels, ["{0:.5g}".format(class_scores[label]['mse'])
                          for label in labels], ["{0:.5g}".format(class_scores[label]['mae'])
                                                 for label in labels], counts))
    output_string = ''

    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(8) for x in d)
        output_string += line + '\n'

        if i in [0, len(data) - 1]:
            output_string += '-' * len(line) + '\n'

    output_string += '|'.join(str(item).ljust(8) for item in ['Subset', "{0:.5g}".format(
        class_scores['subset']), "{0:.5g}".format(subset_mae/len(subset_grades)), subset_support])
    output_string += '\n'

    output_string += '|'.join(str(item).ljust(8) for item in ['Total', "{0:.5g}".format(class_scores['total']), 
        "{0:.5g}".format(total_mae/len(unique)), len(y_true)])
    output_string += '\n'

    output_string += '\nSubset scores are based on grades with at least 30 samples in the test dataset.\n'

    if return_mse:
        return output_string, class_scores
    else:
        return output_string


def final_results_table(results):
    headers = ["Grade"] + active_models
    header_row = '|'.join(str(x).ljust(12) for x in headers)
    
    output_string = header_row + '\n'
    output_string += '-' * len(header_row) + '\n'

    for i in range(4, 15):
        line = '|'.join(str(x).ljust(12) for x in [str(i)] 
            + ["{0:.5g}".format(results[j][i]['mse']) for j in range(len(active_models))])
        output_string += line + '\n'

    output_string += '-' * len(header_row) + '\n'
    
    output_string += '|'.join(str(x).ljust(12) for x in ["Subset"] 
        + ["{0:.5g}".format(results[i]['subset']) for i in range(len(active_models))]) + '\n'
    output_string += '|'.join(str(x).ljust(12) for x in ["Total"] 
        + ["{0:.5g}".format(results[i]['total']) for i in range(len(active_models))]) + '\n'

    output_string += '\nSubset scores are based on grades with at least 30 samples in the test dataset.\n'
    
    return output_string