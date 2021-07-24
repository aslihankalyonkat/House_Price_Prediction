import pickle as pickle
import warnings

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from scripts.helper_functions import *

warnings.simplefilter("ignore", category=ConvergenceWarning)

def traininig():
    pickle_dir = os.getcwd() + '/outputs/pickles/'

    train_df = pickle.load(open(pickle_dir + 'train_dataframe.pkl', 'rb'))
    test_df = pickle.load(open(pickle_dir + 'test_dataframe.pkl', 'rb'))

    y = np.log1p(train_df['SALEPRICE'])
    X = train_df.drop(["SALEPRICE", "ID"], axis=1)
    selected_features = feature_selection(X, y)
    X = X[selected_features]

    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    print("\n########### BASE MODELS ###########\n")
    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

    ######################################################
    # Automated Hyperparameter Optimization
    ######################################################

    ###### CART ######
    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    ###### Random Forests ######
    rf_params = {"max_depth": [20, 22, 24],
                 "max_features": [30, 32, 34],
                 "n_estimators": [400, 600, 800],
                 "min_samples_split": [2, 4]}

    ###### GBM Model ######
    gbm_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [3, 4, 5],
                  "n_estimators": [1600, 1800, 2000],
                  "subsample": [0.2, 0.3, 0.4],
                  "loss": ['huber'],
                  "max_features": ['sqrt']}

    ###### XGBoost ######
    xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                      "max_depth": [5, 8, 12, 15, 20],
                      "n_estimators": [100, 500, 1000],
                      "colsample_bytree": [0.5, 0.7, 1]}

    ###### LightGBM ######
    lightgbm_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [1300, 1500, 1700],
                       "colsample_bytree": [0.2, 0.3, 0.4]}

    ###### CatBoost ######
    catboost_params = {"iterations": [400, 500, 600],
                       "learning_rate": [0.01, 0.1],
                       "depth": [4, 5, 6, 7]}


    regressors = [("CART", DecisionTreeRegressor(), cart_params),
                  ("RF", RandomForestRegressor(), rf_params),
                  ('GBM', GradientBoostingRegressor(), gbm_params),
                  ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
                  ('LightGBM', LGBMRegressor(), lightgbm_params),
                  ("CatBoost", CatBoostRegressor(verbose=False), catboost_params)]

    best_models = {}
    print("\n########### Hyperparameter Optimization ###########\n")
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

    voting_model = VotingRegressor(estimators=[('LightGBM', best_models["LightGBM"]),
                                             ('GBM', best_models["GBM"]),
                                             ('CatBoost', best_models["CatBoost"])])

    print("\n########## Voting Regressor ##########\n")
    rsme = np.mean(np.sqrt(-cross_val_score(voting_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} (''Voting Regressor) ")
    voting_model.fit(X,y)

    return voting_model;


