import copy
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.optimize import minimize
pd.options.mode.chained_assignment = None  # default='warn'


class Portfolio:
    def __init__(self, data):
        self.__data = data
        self.__dates = self.__get_dates(data)
        self.__portfolio = dict()
        self.__construct()

        assert (type(data).__name__ == "DataFrame"), 'Data must be a Pandas DataFrame'

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, new_data):
        if type(new_data).__name__ == "DataFrame":
            self.__data = new_data

    @property
    def portfolio(self):
        return self.__portfolio

    @property
    def dates(self):
        return [str(pd.Timestamp.date(date)) for date in self.__dates]

    @staticmethod
    def __get_dates(data):
        group = groupby(data["Ref Date"].sort_values(ascending=True))
        dates = []
        for x, y in group:
            dates.append(x)
        return dates

    @staticmethod
    def __target_function(x, *args):
        return np.sum((x - args[0]) ** 2 / args[0])

    @staticmethod
    def __compute_bounds(data):
        bounds = []
        for stock in data.iterrows():
            bounds.append((0.0005, np.minimum(0.05, 20 * float(stock[1]["FCap Wt"]))))
        return bounds

    @staticmethod
    def __sector_constraint_function(weights, selection_vector):
        return 0.5 - np.inner(weights, selection_vector)

    @staticmethod
    def __sector_codes(data):
        sector_codes = data.sort_values("Sector Code", ascending=True)["Sector Code"]
        codes = []
        for x, y in groupby(sector_codes):
            codes.append(x)
        return codes

    @staticmethod
    def __sector_vectors(data, sector_codes: list):
        sector_vectors = []
        for code in sector_codes:
            sector_vector = np.array([data["Sector Code"] == code]) * 1
            sector_vectors.append(list(sector_vector[0]))
        return sector_vectors

    @staticmethod
    def __check_sector_constraints(sector_vectors, weights, limit=0.5):
        violated_sector_indices = []
        for index, sector_vector in enumerate(sector_vectors):
            if np.inner(sector_vector, weights) > limit:
                violated_sector_indices.append(index)
        return violated_sector_indices

    @staticmethod
    def __check_stock_bounds(lower_bounds, upper_bounds, weights):
        violated_weight_indices = []
        values = []
        for index, weight in enumerate(weights):
            if weight < lower_bounds[index]:
                values.append(lower_bounds[index])
                violated_weight_indices.append(index)

            elif weight > upper_bounds[index]:
                values.append(upper_bounds[index])
                violated_weight_indices.append(index)

        return violated_weight_indices, values

    @staticmethod
    def __scaled_weights(weights, free_weights, fixed_weights):
        scaling_factor = (1 - np.inner(weights, fixed_weights)) / np.inner(weights, free_weights)
        scaling_vector = scaling_factor * free_weights
        multiplication_vector = \
            [1 if scaling_vector[i] == 0 else scaling_vector[i] for i in range(len(scaling_vector))]
        scaled_weights = np.multiply(multiplication_vector, weights)
        return scaled_weights

    def __compute_constraints(self, data):
        codes = self.__sector_codes(data)
        sector_vectors = self.__sector_vectors(data, codes)
        constraints = []
        for index, code in enumerate(codes):
            constraints.append(
                {'type': 'ineq', 'fun': self.__sector_constraint_function, 'args': [sector_vectors[index]]})
        constraints.append({'type': 'eq', 'fun': lambda weights: 1 - np.inner(weights, np.ones(len(weights)))})
        return constraints

    def __split_data(self):
        subsets = []
        for date in self.__dates:
            subset = self.data.loc[self.data["Ref Date"] == date]
            subsets.append(subset)
        return subsets

    def __construct(self):
        subsets = self.__split_data()
        for index, date in enumerate(self.__dates):
            data = subsets[index].sort_values("Z_Value", ascending=False)
            if index == 0:
                portfolio = data.head(50)
                portfolio.reset_index(drop=True, inplace=True)
            else:
                portfolio = data.head(40)
                x = data.iloc[40:60]
                prev_date = pd.Timestamp.date(self.__dates[index - 1])
                portfolio = portfolio.append(x.loc[x["RIC"].isin(self.__portfolio[f"{prev_date}"]["RIC"])]).head(50)
                portfolio = portfolio.append(data.loc[~data["RIC"].isin(portfolio["RIC"])]).head(50)
                portfolio = portfolio.sort_values("Z_Value", ascending=False)
                portfolio.reset_index(drop=True, inplace=True)

            # maximum_weights, unconstrained_weights, constrained_weights, loss = self.__compute_weights(portfolio)
            maximum_weights, unconstrained_weights, constrained_weights, loss = self.__manual_weights(portfolio)
            """My manual implementation outperforms slightly"""

            portfolio["unconstrained_weights"] = unconstrained_weights
            portfolio["maximum_weights"] = maximum_weights
            portfolio["constrained_weights"] = constrained_weights
            portfolio.reset_index(drop=True, inplace=True)
            self.__portfolio[f"{pd.Timestamp.date(date)}"] = portfolio

    def __compute_weights(self, portfolio):
        unconstrained_weights = pd.Series((portfolio["FCap Wt"] * (1 + portfolio["Z_Value"])) / \
                                          np.sum(portfolio["FCap Wt"] * (1 + portfolio["Z_Value"])))
        maximum_weights = pd.Series(np.minimum(0.05, 20 * portfolio["FCap Wt"]))
        w0 = np.ones(len(unconstrained_weights)) / len(unconstrained_weights)
        bounds = self.__compute_bounds(portfolio)
        constraints = self.__compute_constraints(portfolio)
        result = minimize(self.__target_function, args=unconstrained_weights, x0=w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        constrained_weights = pd.Series(result["x"])
        minimised_loss = self.__target_function(constrained_weights, unconstrained_weights)
        return maximum_weights, unconstrained_weights, constrained_weights, minimised_loss

    def __manual_weights(self, portfolio):
        weights = np.array((portfolio["FCap Wt"] * (1 + portfolio["Z_Value"])) / \
                           np.sum(portfolio["FCap Wt"] * (1 + portfolio["Z_Value"])))
        min_weights = 0.0005 * np.ones(len(weights))
        max_weights = np.array(np.minimum(0.05, 20 * portfolio["FCap Wt"]))
        constrained_weights = copy.deepcopy(weights)

        sector_codes = self.__sector_codes(portfolio)
        sector_vectors = self.__sector_vectors(portfolio, sector_codes)

        violated_weight_indices, values = self.__check_stock_bounds(min_weights, max_weights, constrained_weights)
        violated_sector_indices = self.__check_sector_constraints(sector_vectors, constrained_weights)

        iterations = 0
        while len(violated_weight_indices) > 0 or len(violated_sector_indices) > 0:
            """The loop is necessary because updating individual weights or the weights of assets in a sector to
            satisfy a constraint can then result in a violation of a sector or stock level constraint."""
            free_weights = np.ones(len(constrained_weights))
            fixed_weights = np.zeros(len(constrained_weights))

            violated_weight_indices, values = self.__check_stock_bounds(min_weights, max_weights, constrained_weights)
            new_weights = copy.deepcopy(constrained_weights)

            for index, stock_index in enumerate(violated_weight_indices):
                """Setting the violated weight equal to its lower or upper bound minimises the amount the other weights
                need to be re-scaled, so it is efficient."""
                new_weights[stock_index] = values[index]
                free_weights[stock_index] = 0
                fixed_weights[stock_index] = 1

            new_weights = self.__scaled_weights(new_weights, free_weights, fixed_weights)
            violated_sector_indices = self.__check_sector_constraints(sector_vectors, new_weights)

            for sector_index in violated_sector_indices:
                """If we violate a sector constraint, then it may be optimal to change one of the weights we previously
                 set equal to its bound to change back in the direction of its unconstrained value. This is why we
                 compare the two."""
                # option 1 -- permit previously fixed weights to change
                scale_factor = 0.5 / np.inner(sector_vectors[sector_index], new_weights)
                mul_vector = [scale_factor if sector_vectors[sector_index][i] == 1 else 1 for i in range(len(weights))]
                new_weights1 = np.multiply(mul_vector, new_weights)
                loss1 = self.__target_function(new_weights1, weights)

                # option 2 -- respect previously fixed weights
                free_vec = np.multiply(np.array(sector_vectors[sector_index]), free_weights)
                scaling_factor = (0.5 - np.inner((np.array(sector_vectors[sector_index]) - np.array(free_vec)),
                                                 new_weights)) / np.inner(free_vec, new_weights)
                scaling_vector = scaling_factor * free_vec
                mul_vector = [1 if scaling_vector[i] == 0 else scaling_vector[i] for i in range(len(scaling_vector))]
                new_weights2 = new_weights * mul_vector
                loss2 = self.__target_function(new_weights2, weights)

                if loss1 < loss2:
                    # occasionally this is optimal - it is this that makes the manual a bit better.
                    new_weights = new_weights1
                    free_weights = np.array([free_weights[i] if sector_vectors[sector_index][i] == 0 else 0
                                             for i in range(len(weights))])
                    fixed_weights = -1 * (free_weights - 1)
                else:
                    # most of the time this is optimal.
                    new_weights = new_weights2
                    free_weights = np.multiply(free_weights, (-1 * (free_vec - 1)))
                    fixed_weights = -1 * (free_weights - 1)

            new_weights = self.__scaled_weights(new_weights, free_weights, fixed_weights)
            violated_sector_indices = self.__check_sector_constraints(sector_vectors, new_weights)
            violated_weight_indices, values = self.__check_stock_bounds(min_weights, max_weights, new_weights)
            constrained_weights = new_weights
            iterations += 1

        minimised_loss = self.__target_function(constrained_weights, weights)

        return pd.Series(max_weights), pd.Series(weights), pd.Series(constrained_weights), minimised_loss

    def get_constituents(self, date=None):
        if date is not None:
            return self.__portfolio[f'{date}']
        else:
            return self.__portfolio[f"{self.__dates[-1]}"]

    def sector_breakdown(self, date=None):
        portfolio = self.get_constituents(date)
        sector_codes = self.__sector_codes(portfolio)
        sector_vectors = self.__sector_vectors(portfolio, sector_codes)

        sector_frame = pd.DataFrame(columns=
                                      ["Sector Code", "Num Stocks", "Uncapped Weight", "Max Weight", "Capped Weight"])
        sector_frame["Sector Code"] = sector_codes

        for index, vector in enumerate(sector_vectors):
            sector_frame["Num Stocks"][sector_frame["Sector Code"] == sector_codes[index]] = np.sum(vector)
            sector_frame["Uncapped Weight"][sector_frame["Sector Code"] == sector_codes[index]] = \
                np.inner(vector, portfolio["unconstrained_weights"])
            sector_frame["Max Weight"][sector_frame["Sector Code"] == sector_codes[index]] = \
                np.minimum(np.inner(vector, portfolio["maximum_weights"]), 0.5)
            sector_frame["Capped Weight"][sector_frame["Sector Code"] == sector_codes[index]] = \
                np.inner(vector, portfolio["constrained_weights"])

        return sector_frame
