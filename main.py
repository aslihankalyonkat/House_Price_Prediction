from scripts.eda import *
from scripts.feature_engineering import *
from scripts.predict import *
from scripts.train import *


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    print(" ")


def main(eda):
    with timer("Pipeline"):
        print("Pipeline started")
        with timer("Reading Dataset"):
            print("Reading Dataset Started")
            train = get_train_dataframe()
            test = get_test_dataframe()
            df = train.append(test).reset_index(drop=True)

        if eda:
            with timer("Exploratory Data Analysis"):
                print("Exploratory Data Analysis Started")
                exploratory_data_analysis(df)

        with timer("Data Preprocessing"):
            print("Data Preprocessing Started")
            df = data_preprocessing(df)

        with timer("Feature Engineering"):
            print("Feature Engineering Started")
            feature_engineering(df)

        with timer("Training"):
            print("Training Started")
            final_model = traininig()

        with timer("Predicting"):
            print("Predicting the Results")
            predict(final_model)


if __name__ == "__main__":
    namespace = get_namespace()
    main(namespace.eda)
