import os
from datetime import datetime

def log_output(model, message, print_to_console=True, write_to_file=True):
    string = "[{}][{}] {}".format(datetime.now(), model, message)
    
    if print_to_console:
        print(string)

    if write_to_file:
        with open('output.log', 'a') as file:
            file.write(string + '\n')