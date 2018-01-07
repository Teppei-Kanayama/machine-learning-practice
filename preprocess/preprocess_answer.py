import numpy as np
import numpy.linalg as LA
import pdb
#入力データの例
sample_data1 = np.array([[1.2, 2.0], [1.0, 0.1], [1.3, 1.5], [5.2, 1.0], [1.8, 3.9]])

# 平均ベクトルを計算する関数
# calc_average: np.array(n, d) -> np.array(d,)
def calc_average(data):
    return np.sum(data, axis=0) / data.shape[0]

# 分散・共分散行列を計算する関数
# calc_covariance_matrix: np.array(n, d) -> np.array(d, d)
def calc_covariance_matrix(data):
    average_vector = calc_average(data)
    # broadcastの利用
    X_bar = data - average_vector
    # はじパタP137の式を利用
    covariance_matrix = X_bar.T @ X_bar / data.shape[0]
    return covariance_matrix

# 標準化を行う関数
# normalize: np.array(n, d) -> np.array(n, d)
def normalize(data):
    average_vector = calc_average(data)
    # 単位行列との要素積を計算して対角成分のみ取り出す→列方向に和をとる
    covariance_vector = np.sum(calc_covariance_matrix(data) * np.eye(data.shape[1]), axis=1)
    # broadcastの利用
    normalized_data = (data - average_vector) / np.sqrt(covariance_vector)
    return normalized_data

# 無相関化を行う関数
# non_correlate: np.array(n, d) -> np.array(n, d)
def non_correlate(data):
    covariance_matrix = calc_covariance_matrix(data)
    S = LA.eig(covariance_matrix)[1]
    # S^T @ x を各データに対して施す処理は、X @ Sと等価（Xはデータ行列）
    non_correlated_data = data @ S
    return non_correlated_data
