import time
from contextlib import contextmanager
from scripts.helper_functions import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def exploratory_data_analysis(dataframe):
    check_dataframe(dataframe)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # Categorical variable analysis
    for col in cat_cols:
        cat_summary(dataframe, col, plot=True)

    # Numerical variable analysis
    for col in num_cols:
        num_summary(dataframe, col, plot=True)

    # Target analysis
    low_correlations, high_correlations = find_correlation(dataframe, num_cols)
    print(f"Low Correlations: {low_correlations}, \nHigh Correlations: {high_correlations}")

