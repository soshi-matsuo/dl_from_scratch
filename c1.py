import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

%matplotlib inline


x = np.array([1.0, 2.0, 3.0])
print(x)
# numpyの配列はndarray型で定義される
print(type(x))
# ndarrayどうしの計算
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
# 要素ごと（element-wise）の計算を行う
print(x * y)
print(x / y)

# ２次元配列(行列)
A = np.array([[1, 2], [3, 4]])
print(A)
# ndarrayの形状
print(A.shape)
# 配列内要素のデータ型
print(A.dtype, '\n')
# 行列どうしの計算
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)


# ブロードキャスト
# numpyで形状の異なる配列どうしの計算を行うための機能
# 2*2サイズのAと、スカラー10が同じ形状になるように[[10, 10], [10, 10]]に変形され、要素積が計算される
print(A * 10)
# Aと、1*2サイズのCが同じ形状になるように、Cが[[5, 5], [5, 5]]に変形され、要素積が計算される
C = np.array([5, 5])
print(A * C)


# 要素へのアクセス
# Pythonの通常のインデックス操作
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1], '\n')
# Xをベクトルに変換し、複数の要素をインデックス指定によって抽出
X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])], '\n')
# 条件を満たす要素のみを抽出
# ndarrayに比較演算子を使うとBool配列が返る
# これを利用してX > 15がTrueであるXの要素を抽出
print(X > 15)
print(X[X > 15])


# データ作成
x = np.arange(0, 6, 0.1)  # 0~6まで0.1ステップで配列を生成
y = np.sin(x)

# グラフの描画
plt.plot(x, y)
plt.show()

# cos関数の追加
y1 = np.sin(x)
y2 = np.cos(x)

# グラフの描画
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos')
plt.xlabel('x')  # x軸のラベル
plt.ylabel('y')  # y軸のラベル
plt.title('sin & cos')  # グラフのタイトル
plt.legend()
plt.show()

# 画像の表示
img = imread('deep-learning-from-scratch-master/dataset/lena.png')
plt.imshow(img)
plt.show()