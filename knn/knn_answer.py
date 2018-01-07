#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

import sys
import pdb

#二点間のユークリッド距離を計算する関数
def distance(p0, p1):
    return np.sum((p0 - p1) ** 2)

# ライブラリを使わない場合のNN
class NearestNeighbors(object):
    def __init__(self):
        self._train_data = None
        self._target_data = None

    def fit(self, train_data, target_data):
        """訓練データを用いて学習を行う"""
        # Nearest Neighborの場合、あらかじめ計算しておけるものが特にないので保存だけする
        self._train_data = train_data
        self._target_data = target_data

    def predict(self, x):
        """
        学習データを用いて予測する
        @param x テストデータのarray
        @return 予測ラベルのarray
        """
        # 判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([distance(p, x) for p in self._train_data])
        # 最もユークリッド距離の近い要素のインデックスを得る
        nearest_index = distances.argmin()
        # 最も近い要素のラベルを返す
        return self._target_data[nearest_index]

# ライブラリを使わない場合のkNN
class KNearestNeighbors(object):

    def __init__(self, k=1):
        self._train_data = None
        self._target_data = None
        self._k = k

    def fit(self, train_data, target_data):
        """訓練データを用いて学習を行う"""
        # あらかじめ計算しておけるものが特にないので保存だけする
        self._train_data = train_data
        self._target_data = target_data

    def predict(self, x):
        """
        学習データを用いて予測する
        @param x テストデータのarray
        @return 予測ラベルのarray
        """
        pdb.set_trace()
        # 判別する点と教師データとのユークリッド距離を計算する
        distances = np.array([distance(p, x) for p in self._train_data])
        # ユークリッド距離の近い順でソートしたインデックスを得る
        nearest_indexes = distances.argsort()[:self._k]
        # 最も近い要素のラベルを返す
        nearest_labels = self._target_data[nearest_indexes]
        # 近傍のラベルで一番多いものを予測結果として返す
        c = Counter(nearest_labels)
        return c.most_common(1)[0][0]

# Iris データセットをロードするための関数
def iris_data_loader():
    iris_dataset = datasets.load_iris()
    # 特徴データとラベルデータを取り出す
    features = iris_dataset.data
    targets = iris_dataset.target

    return (features, targets)


def main(task):
    # Iris データセットをロードする
    pdb.set_trace()
    ### features.shape = (150, 4)
    ### targets.shape = (150,)
    features, targets = iris_data_loader()

    # LOO 法で汎化性能を調べる
    predicted_labels = []

    # 一つ抜き法
    loo = LeaveOneOut()

    #モデルの定義
    if task == 1:
        # scikit-learnのkNN
        model = KNeighborsClassifier(n_neighbors=1)

    elif task == 2:
        # scikit-leranを使わない場合のNN
        model = NearestNeighbors()

    elif task == 3:
        # scikit-learnを使わない場合のkNN
        model = KNearestNeighbors(k=3)

    else:
        sys.stderr.write('Task Number is invalid!\n')
        return -1

    for train, test in loo.split(features):
        # 0から順番にtestしていく。
        train_data = features[train]
        target_data = targets[train]

        #定義したモデルに学習データを当てはめる
        model.fit(train_data, target_data)

        # 一つ抜いたテストデータを識別させる
        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)

    # 正解率を出力する
    score = accuracy_score(targets, predicted_labels)
    print(score)

if __name__ == "__main__":
    main(task=3)
