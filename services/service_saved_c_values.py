import numpy as np
import pandas as pd


def load_solution_as_nparray(iter, filepath):
    df = pd.read_csv(filepath)
    selected = df[df['iter'] == iter]
    if selected.empty:
        raise Exception("This iteration number is not present in this file.")
    selected = selected.drop(columns=['iter', 'fitness', 'fitness_order'])
    return np.array(selected)[0]

