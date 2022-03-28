import os
import numpy as np
import pandas as pd
from itertools import groupby


def load_data(filename, sheet_name):
    """This function loads to the data to a pd DataFrame"""
    return pd.read_excel(os.path.join('data', f'{filename}.xlsx'), sheet_name, engine='openpyxl')
    # return pd.read_excel(os.path.join('data', 'SPdata.xls'))


def target(w, *args):
    """This function defines the objective that is minimised during optimisation."""
    x = (w - args[0]) ** 2 / args[0]
    return np.sum(x)


def f_constraint(w, selection_vector):
    """This is a helper function that is used to enforce strict evaluation by the constraint lambda functions"""
    f = lambda s: 0.5 - np.inner(s, selection_vector)  # i've simplified this
    return f(w)


def sector_vectors(codes, portfolio):
    vectors = []
    for code in codes:
        rows = np.array(portfolio["Sector Code"] == code).astype(int)
        vectors.append(rows)
    return vectors


def sector_codes(data):
    sectors = data["Sector Code"]
    codes = []
    for code, rows in groupby(sectors):
        if code not in codes:
            codes.append(code)
    codes.sort()
    return codes


def sector_data(codes, selection_vectors, portfolio, upper_bounds):
    sec_data = pd.DataFrame(codes, columns=["Sector Code"])
    sec_data["Num Stocks"] = [np.sum(selection_vectors[i]) for i in range(len(selection_vectors))]
    sec_data["Uncapped Weights"] = [np.sum(portfolio.loc[portfolio["Sector Code"] == code]["Unconstrained "
                                                                                           "Weights"]) for
                                    code in codes]
    sec_data["Capped Weights"] = [np.sum(portfolio.loc[portfolio["Sector Code"] == code]["Constrained "
                                                                                         "Weights"]) for
                                  code in codes]
    sec_data["Max Weight"] = [np.minimum(np.inner(vector, upper_bounds), 0.5) for vector in selection_vectors]

    return sec_data
