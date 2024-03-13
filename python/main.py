from fusinter import FUSINTERDiscretizer
import time

import pandas as pd

# x = load_iris()["data"]
# y = load_iris()["target"]
#
# from fusinter import FUSINTERDiscretizer
#
# discretizer = FUSINTERDiscretizer(0.95, 0.99)
# discretizer.fit(x,y)
#
# print(discretizer.splits)
# print(discretizer.transform(x))

cov_x = pd.read_csv("datasets/covtype.data", header=None)
cov_y = cov_x.pop(cov_x.shape[1] - 1).to_numpy()
cov_x = cov_x.iloc[:, 0:10].to_numpy()

start = time.perf_counter()

n_runs = 5
for i in range(n_runs):
    discretizer = FUSINTERDiscretizer(0.95, 0.99, not_concurrent=False)
    discretizer.fit(cov_x, cov_y)

end = time.perf_counter()

print(discretizer.splits)
print(discretizer.transform(cov_x))
print(discretizer.transform(cov_x).shape)
print((end - start)/n_runs)
