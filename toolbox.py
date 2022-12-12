# Pure Python
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Union

# Plots
import matplotlib.pyplot as plt
import numpy as np

# Numerical
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud


def word_cloud(occurences: Union[List[str], Counter], title: str = None):
    """
    Plot a word cloud based on a list of occurences, or a counter.
    The more an item appears in the list, the bigger it will be displayed.
    """
    if not isinstance(occurences, Counter):
        freqs = Counter(occurences)
    else:
        freqs = occurences

    plt.figure(figsize=(12, 15))
    if title:
        plt.title(title)

    wc = WordCloud(
        max_words=1000, background_color="white", random_state=1, width=1200, height=600
    ).generate_from_frequencies(freqs)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def get_ground_truth(df: pd.DataFrame, X: pd.DataFrame, n=100) -> Dict[str, np.ndarray]:
    """
    For each entry in X, returns the n closest Pokémon present in df.

    Here, we do not consider the appearance duration, since it makes
    implementation much more difficult.

    :param df: a m1 by m2 DataFrame, without any preprocessing on columns
        "name", "type", "lat", "lng", and "date"
    :param X: a k1 by k2 DataFrame, usually a subset of df
    :return: a dict of two k1 by n arrays, one with names, and the other with types
    """
    epoch_time = datetime(1970, 1, 1)
    columns = ["lat", "lng", "date"]
    A = df[columns]
    X = X[columns]
    A["date"] = (A.date - epoch_time).dt.total_seconds()
    X["date"] = (X.date - epoch_time).dt.total_seconds()

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    X = scaler.transform(X)

    d = cdist(X, A)

    i_min = np.argsort(d, axis=1)

    names = df["name"].values[i_min[:, :n]]
    types = df["type"].values[i_min[:, :n]]
    return dict(name=names, type=types)


def train_val_indices(
    size: int, val_frac: float = 0.25, seed=None
) -> Dict[str, np.ndarray]:
    """
    Generate train and val indices.

    :param size: the dataset size (number of entries)
    :param frac: the fraction of the dataset that will serve as a validation set
    :param seed: random seed used to generate the indices
    """
    assert 0 <= val_frac <= 1.0, "val_frac must be between 0 and 1"
    indices = np.arange(size)
    n = size - int(size * val_frac)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return indices[:n], indices[n:]


def biplot_visualization(
    pca: PCA,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    columns: List[str] = None,
):
    """
    Plot a biplot graph: the scaled data after applying a 2D PCA with loadings in vector forms.

    :param pca: PCA object
    :param X: a n by m matrix (or DataFrame), containing the input prior to the PCA transformation
    :param y: a vector of length n containing the target
    :param columns: a list of length m contained the names of the columns
        If not given, X.columns will be used
    """

    columns = columns if columns is not None else X.columns

    X /= X.max(axis=0) - X.min(axis=0)

    df = pd.DataFrame(data=X, columns=["PC1", "PC2"])

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=columns)

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=y,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig.update_layout(
        annotations=[
            dict(
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                x=0,
                y=0,
                text=index,
                showarrow=True,
                ax=row.PC1,
                ay=row.PC2,
                arrowhead=0,
                arrowcolor="#636363",
            )
            for index, row in loadings.iterrows()
        ],
        width=600,
        height=600,
        xaxis_range=[-1, 1],
        yaxis_range=[-1, 1],
    )

    return fig


def accuracy_metric(y_true, y_pred):
    """
    Return the accuracy vector between y_true and y_pred.

    :param y_true: an n by m array obtained with "get_ground_truth"  (n points and m closest pokemon)
    :param y_pred: an n by m array obtained with Kmeans (n points and m potential pokemon)
    :return: an array of accuracies of size n
    """
    acc = []
    # For each entry
    for true, pred in zip(y_true, y_pred):
        errors = 0
        # Group Pokémon name/type by count
        true_count = Counter(true)
        pred_count = Counter(pred)
        for name in true_count:
            # We penalize the difference from expected (true) count
            errors += abs(true_count[name] - pred_count[name])
            del pred_count[name]

        errors += sum(pred_count.values())
        max_errors = 2 * true.size

        acc.append(1 - errors / max_errors)

    return np.array(acc)
