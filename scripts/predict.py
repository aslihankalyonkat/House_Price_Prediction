import pickle as pickle

from scripts.helper_functions import *


#######################################
# Uploading Results
#######################################
def predict(final_model):
    pickle_dir = os.getcwd() + '/outputs/pickles/'
    train_df = pickle.load(open(pickle_dir + 'train_dataframe.pkl', 'rb'))
    test_df = pickle.load(open(pickle_dir + 'test_dataframe.pkl', 'rb'))

    y = np.log1p(train_df['SALEPRICE'])
    X = train_df.drop(["SALEPRICE", "ID"], axis=1)
    selected_features = feature_selection(X, y)

    ##### Predicting #####
    submission_df = pd.DataFrame()
    submission_df['Id'] = test_df["ID"].astype(int)
    y_pred_sub = final_model.predict(test_df[selected_features])
    y_pred_sub = np.expm1(y_pred_sub)
    submission_df['SalePrice'] = y_pred_sub
    curr_dir = os.getcwd()
    result_dir = curr_dir + '/outputs/submission/'
    submission_df.to_csv(result_dir + 'submission.csv', index=False)
