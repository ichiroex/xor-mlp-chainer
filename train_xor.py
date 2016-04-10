# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils

"""
Multilayer perceptron

多層パーセプトロンでXORを近似
 - 入力層2ユニット
 - 隠れ層2ユニット (1層)
 - 出力層1ユニット 

 回帰問題としてXORを学習する。
 予測する時は閾値を設ける必要あり。
 @author ichiroex
"""

#引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,              help='1: use gpu, 0: use cpu')
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=50000,          help='number of epochs to learn')

args = parser.parse_args()
n_epoch     = args.epoch        # エポック数(パラメータ更新回数)

# Prepare dataset
source = [[0, 0], [1, 0], [0, 1], [1, 1]]
target = [[0], [1], [1], [0]]
dataset = {}
dataset['source'] = np.array(source, dtype=np.float32)
dataset['target'] = np.array(target, dtype=np.float32)

N = len(source) # train data size

in_units  = 2   # 入力層のユニット数
n_units   = 2   # 隠れ層のユニット数
out_units = 1   # 出力層のユニット数

#モデルの定義
model = chainer.Chain(l1=L.Linear(in_units, n_units),
                      l2=L.Linear(n_units , out_units))

#GPUを使うかどうか
if args.gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

def forward(x, t):
    h1 = F.sigmoid(model.l1(x))
    return model.l2(h1)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
loss_val = 100
epoch = 0
while loss_val > 1e-5:

    # training
    x = chainer.Variable(xp.asarray(dataset['source'])) #source
    t = chainer.Variable(xp.asarray(dataset['target'])) #target
    
    model.zerograds()       # 勾配をゼロ初期化
    y    = forward(x, t)    # 順伝搬

    loss = F.mean_squared_error(y, t) #平均二乗誤差
    
    loss.backward()              # 誤差逆伝播
    optimizer.update()           # 最適化 
    
    # 途中結果を表示
    if epoch % 1000 == 0:
        #誤差と正解率を計算
        loss_val = loss.data

        print 'epoch:', epoch
        print 'x:\n', x.data
        print 't:\n', t.data
        print 'y:\n', y.data

        print('train mean loss={}'.format(loss_val)) # 訓練誤差, 正解率
        print ' - - - - - - - - - '
    
    # n_epoch以上になると終了
    if epoch >= n_epoch:
        break

    epoch += 1

#modelとoptimizerを保存
print 'save the model'
serializers.save_npz('xor_mlp.model', model)
print 'save the optimizer'
serializers.save_npz('xor_mlp.state', optimizer)

