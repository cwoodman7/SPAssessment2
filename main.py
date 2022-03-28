import os
import utils
import numpy as np
import pandas as pd
import scipy.optimize as sco
from functools import partial
from itertools import groupby


def main():
    data = utils.load_data('Python assessment', 'Start Universe')
    constituents = question1(data)
    portfolios, objectives, sector_data = question2(data, constituents)
    portfolios, objectives, sector_data = question2m(data, constituents)

    # print results to terminal
    for i, portfolio in enumerate(portfolios):
        print(f"Portfolio {i + 1}: \n")
        print(portfolio)
        print("\n")
        print(f"Value of objective function for rebalance {i + 1}: {objectives[i]} \n")
        print("\n")
        print("Sector Breakdown: \n")
        print(sector_data[i])
        print("\n")

    # export data to excel file
    with pd.ExcelWriter(os.path.join('output', "Output.xlsx")) as writer:
        for i, portfolio in enumerate(portfolios):
            ref_date = str(portfolio["Ref Date"][0].date())
            portfolio.to_excel(writer, sheet_name=f"{ref_date} - Constituents", index=False)
            sector_data[i].to_excel(writer, sheet_name=f"{ref_date} - Sectors", index=False)


def question1(data):
    """This function derives the constituent assets for each portfolio subject to the stipulated construction rules"""
    # split data into sub-frames by date
    data = data.sort_values('Ref Date', axis=0, ascending=True, ignore_index=True)
    dates = data['Ref Date']
    unique_dates = list()
    for date, items in groupby(dates):
        unique_dates.append(date)

    sub_frames = list()
    for date in unique_dates:
        sub_frame = data.loc[data['Ref Date'] == date]
        sub_frames.append(sub_frame)

    A, B, C = sub_frames

    # construct portfolios subject to construction rules
    portfolios = []
    for x, y in enumerate([A, B, C]):
        y = y.sort_values('Z_Value', axis=0, ascending=False)
        portfolio = y.head(40)
        y = y.tail(len(y) - len(portfolio))
        y.reset_index(inplace=True)
        if x > 0:
            R = y.head(20)
            S = portfolios[x - 1]
            inter = R.loc[R["RIC"].isin(list(S["RIC"]))]  # finds intersection
            inter = inter[portfolio.columns]  # gets the necessary columns for the data
            portfolio = portfolio.append(inter.head(10), ignore_index=True)
            y = y.drop(list(inter.index.values), axis=0)  # remove the stocks added from the remaining set
            y.reset_index(inplace=True)

        if len(portfolio) < 50:
            portfolio = portfolio.append(y.loc[0:50 - len(portfolio) - 1][portfolio.columns], ignore_index=True)
        portfolio = portfolio.sort_values('Z_Value', axis=0, ascending=False, ignore_index=True)
        portfolios.append(portfolio)

    return portfolios


def question2(data, constituents):
    """This function derives the constrained and unconstrained weights for each portfolio"""
    portfolios = []
    objectives = []
    sector_df = []

    sector_codes = utils.sector_codes(data)

    for portfolio in constituents:
        # Compute unconstrained weights
        FCap = np.array(portfolio["FCap Wt"])
        Z = np.array(portfolio["Z_Value"])
        Wu = pd.DataFrame(np.multiply(FCap, 1 + Z), columns=["Unconstrained Weights"])
        Wu = Wu * (1 / np.sum(Wu))
        portfolio["Unconstrained Weights"] = Wu  # add an unconstrained weights column to the portfolio dataframe

        # Define upper and lower parameter bounds
        upper_bounds = np.minimum(0.05 * np.ones(len(FCap)), 20 * FCap)
        bounds = tuple()
        for i in range(len(upper_bounds)):
            bounds = bounds + ((0.0005, upper_bounds[i]),)

        # Define sector constraints
        selection_vectors = utils.sector_vectors(sector_codes, portfolio)

        cons = list()
        for x, y in enumerate(selection_vectors):
            cons.append({'type': 'ineq', 'fun': partial(utils.f_constraint, y)})

        # Ensure weights sum to 1
        cons.append({'type': 'eq', 'fun': lambda w: 1 - np.inner(w, np.ones(len(w)))})

        # Set initial weight vector equal to unrestricted weight vector
        init = np.array(Wu).reshape(len(Wu))

        # Minimise objective function subject to bounds and constraints using Sequential Least Squares
        result = sco.minimize(utils.target, x0=init, args=init, method='SLSQP', bounds=bounds, constraints=cons)

        portfolio["Constrained Weights"] = result['x']  # append constrained weights to portfolio dataframe
        portfolio = portfolio.sort_values('Z_Value', axis=0, ascending=False, ignore_index=True)
        portfolios.append(portfolio)

        objectives.append(result['fun'])

        # Construct Sector Data
        sector_data = utils.sector_data(sector_codes, selection_vectors, portfolio, upper_bounds)
        sector_df.append(sector_data)

    return portfolios, objectives, sector_df


def question2m(data, constituents):
    """This function provides a manual implementation of the portfolio optimisation process"""
    portfolios = []
    objectives = []
    sector_df = []

    sector_codes = utils.sector_codes(data)

    for portfolio in constituents:
        # Compute unconstrained weights
        FCap = np.array(portfolio["FCap Wt"])
        Z = np.array(portfolio["Z_Value"])
        Wu = pd.DataFrame(np.multiply(FCap, 1 + Z), columns=["Unconstrained Weights"])
        Wu = Wu * (1 / np.sum(Wu))
        portfolio["Unconstrained Weights"] = Wu  # add an unconstrained weights column to the portfolio dataframe

        # define boundaries
        upper_bounds = pd.DataFrame(np.minimum(0.05 * np.ones(len(FCap)), 20 * FCap), columns=["Upper Bounds"])
        lower_bounds = pd.DataFrame(np.ones(len(FCap)) * 0.0005, columns=["Lower Bounds"])

        # initialise constrained weights
        portfolio["Constrained Weights"] = Wu

        # Keep track of the weights that cannot be further altered
        constrained_indices = set()

        while True:
            # Check for existence of individual boundary violations
            DL = pd.DataFrame(portfolio["Constrained Weights"] - lower_bounds["Lower Bounds"], columns=["Diff"])
            DU = pd.DataFrame(portfolio["Constrained Weights"] - upper_bounds["Upper Bounds"], columns=["Diff"])

            indices_L = np.array(DL.loc[DL["Diff"] < 0].index.values)
            indices_U = np.array(DU.loc[DU["Diff"] > 0].index.values)
            indices_LU = np.concatenate([indices_L, indices_U])

            # Check for existence of sector constraint violations
            sector_violations = []
            for code in sector_codes:
                rows = portfolio.loc[portfolio["Sector Code"] == code]
                weights = rows["Constrained Weights"]
                if np.sum(weights) > 0.5:
                    sector_violations.append(code)

            if len(indices_LU) == 0 and len(sector_violations) == 0:
                break

            else:
                # First impose uni-variate boundary constraints
                portfolio["Constrained Weights"] = pd.DataFrame(np.minimum(np.array(portfolio["Constrained Weights"]),
                                                                           np.array(upper_bounds["Upper Bounds"])),
                                                                columns=["Constrained Weights"])
                portfolio["Constrained Weights"] = pd.DataFrame(np.maximum(np.array(portfolio["Constrained Weights"]),
                                                                           np.array(lower_bounds["Lower Bounds"])),
                                                                columns=["Constrained Weights"])
                constrained_indices.update(list(indices_LU))

                # Now impose the multi-variate sector constraints.
                for code in sector_codes:
                    weights = portfolio["Constrained Weights"].loc[portfolio["Sector Code"] == code]
                    if np.sum(np.array(weights)) > 0.5:
                        subset_A = weights.copy()
                        subset_B = []
                        for i in range(len(weights)):
                            if weights.index.values[i] in indices_L:  # allows weights at the upper limit to be reduced
                                subset_B.append(weights[i])
                                subset_A[i] = 0

                        g = (0.5 -
                             np.sum(subset_B)) / np.sum(np.array(subset_A))
                        subset_A = g * subset_A

                        for x, y in enumerate(indices_L):
                            subset_A[y] = subset_B[x]

                        for index in list(subset_A.index.values):
                            portfolio["Constrained Weights"][index] = subset_A[index]

                        constrained_indices.update(list(weights.index.values))

                # scale weights to sum to 1
                weights = portfolio["Constrained Weights"]
                subset_A = weights.copy()
                subset_B = []
                for i in range(len(weights)):
                    if i in constrained_indices:
                        subset_B.append(weights[i])
                        subset_A[i] = 0

                c = (1 - np.sum(subset_B)) / np.sum(subset_A)
                subset_A = c * subset_A
                for x, y in enumerate(constrained_indices):
                    subset_A[y] = subset_B[x]

                portfolio["Constrained Weights"] = subset_A

        # append portfolio to list
        portfolios.append(portfolio)

        # Compute optimised objective function
        objective = np.sum(
            (np.array(portfolio["Constrained Weights"]) - np.array(portfolio["Unconstrained Weights"])) ** 2 / \
            np.array(portfolio["Unconstrained Weights"]))
        objectives.append(objective)

        # construct sector data
        selection_vectors = utils.sector_vectors(sector_codes, portfolio)

        # Construct Sector Data
        sector_data = utils.sector_data(sector_codes, selection_vectors, portfolio, upper_bounds["Upper Bounds"])
        sector_df.append(sector_data)

    return portfolios, objectives, sector_df


if __name__ == '__main__':
    main()