import numpy as np
#import preprocess
import preprocess_answer as preprocess
import time

ACC = 0.000001

#sample_data1 = np.array([[1.2, 2.0], [1.0, 0.1], [1.1, 1.5], [5.2, 1.0], [1.8, 3.9]])
sample_data1 = np.random.random((1000, 500))
#sample_data1 = np.random.random((5, 2))
#print(sample_data1)

start = time.time()
normalized_data = preprocess.normalize(sample_data1)
end = time.time()
print(end - start)

average_vector = preprocess.calc_average(normalized_data)
covariance_matrix = preprocess.calc_covariance_matrix(normalized_data)
#print(average_vector)
#print(covariance_matrix)

average_check = average_vector < ACC
covariance_check = ((covariance_matrix * np.eye(sample_data1.shape[1]) - np.eye(sample_data1.shape[1])))**2 < ACC
#print(covariance_check)

if np.all(average_check) and np.all(covariance_check):
    print("OK!")
else:
    print("NG.")
