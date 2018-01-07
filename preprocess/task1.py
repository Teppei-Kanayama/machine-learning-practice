import numpy as np
#import preprocess_answer as preprocess
import preprocess
import time

ACC = 0.000001

sample_data1 = np.array([[1.2, 2.0], [1.0, 0.1], [1.3, 1.5], [5.2, 1.0], [1.8, 3.9]])
sample_data2 = np.random.random((1000, 500))

answer = np.cov(sample_data2, rowvar=0, bias=1)

start = time.time()
error = (preprocess.calc_covariance_matrix(sample_data2) - answer) ** 2
end = time.time()

print(end - start)
if np.all(error < ACC):
    print("OK!")
else:
    print("NG.")
