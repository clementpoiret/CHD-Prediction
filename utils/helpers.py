import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def pipeline(X, onehotcolumn=3):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    ct = ColumnTransformer(
        [("education", OneHotEncoder(categories="auto"), [onehotcolumn])],
        remainder="passthrough")
    X = ct.fit_transform(X)
    X = X[:, 1:]

    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, sc


def plot3d(X, c, colors=["black", "red"], cmap=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if cmap:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, cmap=ListedColormap(colors))
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c)

    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")

    plt.show()
