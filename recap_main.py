import os
import copy
import numpy as np
import pandas as pd
from itertools import groupby
from portfolio import Portfolio


def main():
    filename = 'data'
    sheet_name = 'Start Universe'
    data = pd.read_excel(os.path.join('data', f'{filename}.xlsx'), sheet_name=sheet_name, engine='openpyxl')

    portfolios = Portfolio(data)

    with pd.ExcelWriter(os.path.join('output', "Output.xlsx")) as writer:
        for date in portfolios.dates:
            portfolio = portfolios.get_constituents(date)
            sector = portfolios.sector_breakdown(date)
            portfolio.to_excel(writer, sheet_name=f"{date} - Constituents", index=False)
            sector.to_excel(writer, sheet_name=f"{date} - Sector Breakdown", index=False)

    """Note: the dates used in the dictionaries and the code are strings of date objects, but the dates in the data
    frames are date times. """


if __name__ == "__main__":
    main()
