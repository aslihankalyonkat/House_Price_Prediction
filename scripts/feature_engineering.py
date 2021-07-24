import pickle as pickle

from scripts.data_preprocessing import *


def feature_engineering(dataframe):
    # Does it have garage ?
    dataframe["HAS_GARAGE"] = dataframe["GARAGETYPE"].apply(lambda x: 0 if x == "None" else 1)

    # Does it have basement ?
    dataframe["HAS_BASEMENT"] = dataframe["BSMTCOND"].apply(lambda x: 0 if x == "None" else 1)

    # Does it have pool ?
    dataframe["HAS_POOL"] = dataframe["POOLQC"].apply(lambda x: 0 if x == "None" else 1)

    # Does it have fireplace ?
    dataframe["HAS_FIREPLACE"] = dataframe["FIREPLACEQU"].apply(lambda x: 0 if x == "None" else 1)

    # Does it have fence ?
    dataframe["HAS_FENCE"] = dataframe["FENCE"].apply(lambda x: 0 if x == "None" else 1)

    # Does it have second floor ?
    dataframe["HAS_SECOND_FLOOR"] = dataframe["HOUSESTYLE"].apply(lambda x: 1 if "2" in x else 0)

    # Does it have wood deck ?
    dataframe["HAS_WOOD_DECK"] = dataframe["WOODDECKSF"].apply(lambda x: 0 if x == 0 else 1)

    # Does it have open porch ?
    dataframe["HAS_OPEN_PORCH"] = dataframe["OPENPORCHSF"].apply(lambda x: 0 if x == 0 else 1)

    # Does it have enclosed porch ?
    dataframe["HAS_ENCLOSED_PORCH"] = dataframe["ENCLOSEDPORCH"].apply(lambda x: 0 if x == 0 else 1)

    # Does it have 3Ssn porch ?
    dataframe["HAS_3SSN_PORCH"] = dataframe["3SSNPORCH"].apply(lambda x: 0 if x == 0 else 1)

    # Does it have screen porch ?
    dataframe["HAS_SCREEN_PORCH"] = dataframe["SCREENPORCH"].apply(lambda x: 0 if x == 0 else 1)

    # Has it been renovated ?
    dataframe.loc[dataframe["YEARREMODADD"] == dataframe["YEARBUILT"], "HAS_BEEN_RENOVATED"] = 0
    dataframe.loc[dataframe["YEARREMODADD"] != dataframe["YEARBUILT"], "HAS_BEEN_RENOVATED"] = 1

    # Total Home Quality
    dataframe['TOTAL_HOME_QUALITY'] = dataframe['OVERALLQUAL'] + dataframe['OVERALLCOND']

    # Garage age
    dataframe.loc[(dataframe["GARAGEYRBLT"] != 0), ['GARAGE_AGE']] = 2011 - dataframe["GARAGEYRBLT"]
    dataframe.loc[(dataframe["GARAGEYRBLT"] == 0), ['GARAGE_AGE']] = 0
    # House Age
    dataframe["HOUSE_AGE"] = 2011 - dataframe["YEARBUILT"]
    # dropping date colums
    dataframe.drop(["YEARBUILT", "YEARREMODADD", "GARAGEYRBLT"], inplace=True, axis=1)

    # Total Porch square feet
    dataframe["TOTALPORCH"] = dataframe["OPENPORCHSF"] + dataframe["ENCLOSEDPORCH"] + dataframe["3SSNPORCH"] + \
                              dataframe["SCREENPORCH"]

    # Total square feet
    dataframe["TOTALSF"] = dataframe["1STFLRSF"] + dataframe["2NDFLRSF"] + dataframe["TOTALBSMTSF"]

    # Total Bathrooms
    dataframe['TOTAL_BATHROOMS'] = (dataframe['FULLBATH'] + (0.5 * dataframe['HALFBATH']) +
                                    dataframe['BSMTFULLBATH'] + (0.5 * dataframe['BSMTHALFBATH']))

    # Ratios
    dataframe["GARAGE_RATIO"] = dataframe["GARAGEAREA"] / dataframe["TOTALSF"]
    dataframe["PORCH_RATIO"] = dataframe["TOTALPORCH"] / dataframe["TOTALSF"]
    dataframe["HOUSE_RATIO"] = dataframe["TOTALSF"] / dataframe["LOTAREA"]
    dataframe["LIVING_AREA_RATIO"] = dataframe["GRLIVAREA"] / dataframe["TOTALSF"]

    ###### Rare Encoding ######
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    #rare_analyser(dataframe, "SALEPRICE", cat_cols)

    dataframe = rare_encoder(dataframe, 0.01)

    useless_cols = [col for col in cat_cols if dataframe[col].nunique() == 1 or
                    (dataframe[col].nunique() == 2 and (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(
                        axis=None))]

    cat_cols = [col for col in cat_cols if col not in useless_cols]

    for col in useless_cols:
        dataframe.drop(col, axis=1, inplace=True)

    ###### Label Encoding ######
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [np.int64, np.float64]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    ###### One-Hot Encoding ######
    cat_cols = cat_cols + cat_but_car

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=False)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    useless_cols_new = [col for col in cat_cols if
                        (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(axis=None)]

    for col in useless_cols_new:
        dataframe.drop(col, axis=1, inplace=True)

    ###### Saving Test and Train dataframe as pickle ######
    train_dataframe = dataframe[dataframe["SALEPRICE"].notnull()]
    test_dataframe = dataframe[dataframe["SALEPRICE"].isnull()].drop("SALEPRICE", axis=1)

    curr_dir = os.getcwd()
    result_dir = curr_dir + '/outputs/pickles/'

    with open(result_dir + 'train_dataframe.pkl', 'wb') as f:
        pickle.dump(train_dataframe, f)

    with open(result_dir + 'test_dataframe.pkl', 'wb') as f:
        pickle.dump(test_dataframe, f)
