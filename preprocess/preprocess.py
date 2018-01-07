import numpy as np
import numpy.linalg as LA

#
# 解答者：東京大学工学部計数工学科２年 山中耀裕 1/4
#


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
    # 平均ベクトル(d次元)を求める
    # あらかじめN次元ベクトルYi=(Xni-μi)_(n=1..N)を計算しておく
    # それぞれの内積を取る
    # 上の操作で，ベクトルたちをまとめて行列表記すると，以下のようになる

    averages = calc_average(data)
    S = np.array([x - averages for x in data])
    covariance_matrix = S.T @ S / len(S)
    return covariance_matrix

# 標準化を行う関数
# normalize: np.array(n, d) -> np.array(n, d)
def normalize(data):
    # TASK 2
    averages = calc_average(data)
    S = calc_covariance_matrix(data)
    T = np.array([x - averages for x in data])
    normalized_data = np.array([T.T[i]/np.sqrt(S[i][i]) for i in range(len(T.T))]).T
    return normalized_data

# 無相関化を行う関数
# non_correlate: np.array(n, d) -> np.array(n, d)
def non_correlate(data):
    # TASK 3
    S = np.linalg.eig(calc_covariance_matrix(data))
    non_correlated_data = np.array([S[1].T @ x for x in data])
    return non_correlated_data
