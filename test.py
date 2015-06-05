from time import time
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ]
y = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ]

t0 = time()
print cross_val_score(LinearRegression(), X, y, n_jobs=1)
print time()-t0

t0 = time()
print cross_val_score(LinearRegression(), X, y, n_jobs=-1)
print time()-t0
