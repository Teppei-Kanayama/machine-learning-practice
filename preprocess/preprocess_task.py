import numpy as np
import numpy.linalg as LA

# 入力データ行列の例
# このデータ例は、2次元特徴量を持ったデータ5個を表している。
sample_data1 = np.array([[1.2, 2.0], [1.0, 0.1], [1.3, 1.5], [5.2, 1.0], [1.8, 3.9]])

# 平均ベクトルを計算する関数
# calc_average: np.array(n, d) -> np.array(d,)
def calc_average(data):
    return np.sum(data, axis=0) / data.shape[0]

# 分散・共分散行列を計算する関数
# calc_covariance_matrix: np.array(n, d) -> np.array(d, d)
def calc_covariance_matrix(data):
    # TASK 1
    covariance_matrix = None
    return covariance_matrix

# 標準化を行う関数
# normalize: np.array(n, d) -> np.array(n, d)
def normalize(data):
    # TASK 2
    normalized_data = None
    return normalized_data

# 無相関化を行う関数
# non_correlate: np.array(n, d) -> np.array(n, d)
def non_correlate(data):
    # TASK 3
    non_correlate = None
    return non_correlated_data
