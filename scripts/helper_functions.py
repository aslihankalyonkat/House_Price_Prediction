import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def get_namespace():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eda', dest='eda', action='store_true')
    parser.set_defaults(eda=False)

    return parser.parse_args()


def get_train_dataframe():
    return pd.read_csv("data/train.csv")


def get_test_dataframe():
    return pd.read_csv("data/test.csv")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def hist_for_numeric_columns(dataframe, numeric_columns):
    """
    -> Sayısal değişkenlerin histogramını çizdirir.
    :param dataframe: İşlem yapılacak dataframe.
    :param numeric_columns: Sayısal değişkenlerin adları
    """
    col_counter = 0

    data = dataframe.copy()

    for col in numeric_columns:
        data[col].hist(bins=20)

        plt.xlabel(col)

        plt.title(col)

        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted!")

def num_summary(dataframe, col_name, plot=False):
    print(dataframe[col_name].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T)
    if plot:
        dataframe[col_name].hist(bins=10)
        script_dir = os.path.dirname(__file__)
        result_dir = script_dir + '/../outputs/num_col_graphs/'
        plt.savefig(f"{result_dir}/{col_name}.png")


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': dataframe[col_name].value_counts() * 100 / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        script_dir = os.path.dirname(__file__)
        result_dir = script_dir + '/../outputs/cat_col_graphs/'
        plt.savefig(f"{result_dir}/{col_name}.png")


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


def check_dataframe(dataframe, row=5):
    print('####################  Shape   ####################')
    print(dataframe.shape)
    print('####################  Types   ####################')
    print(dataframe.dtypes)
    print('####################   Head   ####################')
    print(dataframe.head(row))
    print('####################   Tail   ####################')
    print(dataframe.tail(row))
    print('####################    NA    ####################')
    print(dataframe.isnull().sum())
    print('#################### Describe ####################')
    print(dataframe.describe().T)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def feature_selection(X, y):
    cols = X.columns
    # Backward Elimination
    cols = list(X.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    return selected_features_BE
