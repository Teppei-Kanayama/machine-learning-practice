#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# 解答者：東京大学工学部計数工学科２年 山中耀裕
#

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

import sys
import pdb

import queue

#二点間のユークリッド距離の２乗を計算する関数
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
		self._train_data = train_data # 教師データ(4次元 * N)
		self._target_data = target_data # グループのリスト

	def predict(self, x):
		"""
		学習データを用いて予測する
		@param x テストデータのarray
		@return 予測ラベルのarray
		"""
		# 判別する点と教師データとのユークリッド距離を計算する
		### TASK 2 ###
		# テストデータxと各訓練データとの距離のリストdistancesを計算。
		distances=[]
		for data in self._train_data:
			distances.append(distance(data,x))
		distances=np.array(distances)

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
		### TASK3 ###

		"""
		priority_queueのpython版をググる
		priority_queueにpython版pair(距離,index)を入れる
		距離の小さい方からk個取り出して，もっとも多いもののindexを返す
		"""

		#pdb.set_trace()
		distances = queue.PriorityQueue()
		for i in range(len(self._train_data)):
			distances.put( (distance(self._train_data[i],x),i ) )

		index = {}
		for i in range(self._k):
			v = distances.get()[1]
			if v in index.keys():
				index[v]+=1
			else:
				index[v]=1

		pdb.set_trace()
		nearest_index, group_size = max(index.items(), key=lambda x: x[1]) #これで辞書中の最大の値,indexを求められる
		pdb.set_trace()
		return self._target_data[nearest_index]


# Iris データセットをロードするための関数
def iris_data_loader():
	iris_dataset = datasets.load_iris()
	# 特徴データとラベルデータを取り出す
	features = iris_dataset.data
	targets = iris_dataset.target

	return (features, targets)


def main(task):
	# Iris データセットをロードする
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
