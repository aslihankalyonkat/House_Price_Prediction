from scripts.helper_functions import *


def data_preprocessing(dataframe):

    ##################
    # Missing Values
    ##################

    dataframe.columns = [col.upper() for col in dataframe.columns]

    # for categorical columns that have NaN as category
    nan_but_cat = ["ALLEY", "UTILITIES", "MASVNRTYPE", "BSMTQUAL", "BSMTCOND", "BSMTEXPOSURE", "BSMTFINTYPE1",
                   "BSMTFINTYPE2", "ELECTRICAL", "FIREPLACEQU", "POOLQC", "FENCE", "MISCFEATURE", "GARAGETYPE",
                   "GARAGEFINISH", "GARAGEQUAL"]

    dataframe[nan_but_cat] = dataframe[nan_but_cat].fillna("None")

    # for numerical columns that have NaN as zero
    nan_but_zero = ["LOTFRONTAGE", "MASVNRAREA", "BSMTFINSF1", "BSMTFINSF2", "BSMTUNFSF", "TOTALBSMTSF", "BSMTFULLBATH",
                    "BSMTHALFBATH", "GARAGEYRBLT", "GARAGECARS", "GARAGEAREA", "GARAGECOND"]

    dataframe[nan_but_zero] = dataframe[nan_but_zero].fillna(0)

    # for nan values EXTERIOR1ST and EXTERIOR2ND is unknown
    dataframe[["EXTERIOR1ST", "EXTERIOR2ND"]] = dataframe[["EXTERIOR1ST", "EXTERIOR2ND"]].fillna("Unknown")

    # for nan values kitchenqual and functional is unknown
    dataframe[["KITCHENQUAL", "FUNCTIONAL"]] = dataframe[["KITCHENQUAL", "FUNCTIONAL"]].fillna("Unknown")

    # fixing wrong garage year
    dataframe.loc[dataframe["GARAGEYRBLT"] == 2207, ["GARAGEYRBLT"]] = 2007

    ##################
    # Outliers
    ##################

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    num_cols = [col for col in num_cols if col not in ["SALEPRICE"]]

    # for col in num_cols:
    #    print(col, check_outlier(dataframe, col, q1=0.01, q3=0.99))

    for col in num_cols:
        replace_with_thresholds(dataframe, col, q1=0.01, q3=0.99)

    #dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

    return dataframe