from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor


def get_best_n_neighbors(
    df: pd.DataFrame, chunk_size: int, features: list[str], range_num: int = 10
):
    """
    This function gets the best n_neighbors for the LocalOutlierFactor model.
    """
    # get the index of the fraud and legit rows
    fraud_perc = df["is_fraud"].value_counts(normalize=True)["Yes"]
    fraud_idx = round(fraud_perc * chunk_size)
    legit_idx = chunk_size - fraud_idx
    # use the index to get the fraud and legit rows
    df_fraud_subset = df.iloc[fraud_rows_index[:fraud_idx], :]
    df_legit_subset = df.iloc[legit_rows_index[:legit_idx], :]
    # concat the two dataframes
    df_subset = pd.concat([df_fraud_subset, df_legit_subset]).reset_index(drop=True)
    # get the features and target
    X = df_subset[features]
    y = df_subset["is_fraud"]
    # lael encode features and target
    X = X.apply(LabelEncoder().fit_transform)
    y = y.map({"Yes": 1, "No": 0})

    # create an empty list to store the scores
    scores = []

    # loop over the range of n_neighbors
    for num in range(1, range_num):
        lof = LocalOutlierFactor(n_neighbors=num, contamination=0.00521)
        y_pred = lof.fit_predict(X)
        y_pred = pd.Series(y_pred).map({-1: 1, 1: 0})
        score = f1_score(y, y_pred, average="macro")
        scores.append(score)
        print(f'n_neighbors: {num}, F1-score: {score}')
        print('====' * 10)

    # get the best n_neighbors
    best_n_neighbors = np.argmax(scores) * 1 + 1

    # log the best n_neighbors and max score
    print('\n')
    print(f"Best n_neighbors: {best_n_neighbors} with score {max(scores)}")

    # plot n_neighbors vs F1-score
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, range_num), scores)
    plt.xticks(range(1, range_num))
    plt.title("n_neighbors vs F1-score")
    plt.xlabel("n_neighbors")
    plt.ylabel("F1-score")
    plt.show()
