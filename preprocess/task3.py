import numpy as np
import numpy.linalg as LA
import preprocess

ACC = 0.000001

sample_data1 = np.array([[1.2, 2.0], [1.0, 0.1], [1.3, 1.5], [5.2, 1.0], [1.8, 3.9]])
covariance_matrix = preprocess.calc_covariance_matrix(sample_data1)

non_correlated_data = preprocess.non_correlate(sample_data1)

non_correlated_covariance_matrix = preprocess.calc_covariance_matrix(non_correlated_data)
eigen_values = LA.eig(covariance_matrix)[0]
answer = np.tile(eigen_values, (sample_data1.shape[1], 1)) * np.eye(sample_data1.shape[1])
error = (non_correlated_covariance_matrix - answer) ** 2

if np.all(error < ACC):
    print("OK!")
else:
    print("NG.")
