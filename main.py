# -*- coding: utf-8 -*-
import numpy as np

from scipy import signal
from scipy import io

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import manifold

# データセットモジュール
import DataSetFunction as DSF

# 処理をまとめた関数
import utils

# NaNデータ除去
def ExceptNaN(DataBase, CLS, trialaxis='Trial'):
    tmp = 0.0
    nan_list = []
    for cls in CLS:
        tmp = tmp + DataBase.data(cls)

    orgkeys = DataBase.Class[0].axiskeys[:]
    chkeys =  DataBase.Class[0].axiskeys[:]

    chkeys.pop(chkeys.index(trialaxis))
    chkeys.insert(0, trialaxis)

    DataBase.Transpose(chkeys)
    l = tmp.ndim - 1
    for i in range(l):
        tmp = np.sum(tmp, axis=1)
    DataBase.Transpose(orgkeys)

    tmp = np.argwhere(np.isnan(tmp))
    for i in range(tmp.shape[0]):
        nan_list.append(tmp[i, 0])
    return nan_list


# ここからメインプログラム ===============================================================
srate = 250 #実験データのサンプリングレート 単位[Hz]
CLS = ['left', 'right', 'foot', 'tongue'] #クラス名をクラス1から順に左手、右手、足、舌とする
classloop = ['class1', 'class2', 'class3', 'class4'] #matファイルから取り出す際に使用
ClassNum = len(CLS)

ChNum = 22 #EEGチャネル数
EOGChNum = 3 #EOG(眼電)チャネル数

subject = 9 #被験者数
kNum = 4 #交差検証数(n-foldの場合のnに相当)

RestTimeRange = [0, 1] #レストタイムの時間範囲 単位[s]

dsrate = 2 #ダウンサンプリングレート

rest = (0, 2) #レストタイム0~2[s]
task = (2, 6) #タスクタイム2~6[s]

#=======================================================================================

print('<START>')
for sj in range(subject):
    print('Subject'+str(sj+1)+'+++++++++++++++++++++')
    print('データロード中')
    matdata_T = io.loadmat('Datasets/DATA_BCICIV_2a_A0'+str(sj+1)+'T.mat', squeeze_me=True)
    EEG_T = DSF.DataBase()
    for i, cls in enumerate(classloop):
        EEG_T.AddClassData( ClassName=CLS[i], axiskeys=['Ch', 'Time', 'Trial'],
                        data = np.array(matdata_T[cls])[:ChNum, :, :])

    EEG_T.Transpose(['Trial', 'Ch', 'Time']) #データ軸はこの順番にしておく
    del matdata_T

    print('NaN含むトライアル除去中')
    nan_list = ExceptNaN(EEG_T, CLS, trialaxis='Trial')
    EEG_T.ClassdataDelete(nan_list, axis=0)

    print('ベースライン補正中')
    EEG_T_blc = DSF.DataBase()
    for cls in CLS:
        EEG_T_blc.AddClassData( ClassName=cls, axiskeys=['Trial', 'Ch', 'Time'],
                        data = signal.detrend(EEG_T.data(cls)[:, ..., :], axis=2))
    del EEG_T


    #解析に用いるデータ================================
    epoch = int(3)
    classNum = int(4) #クラス数

    #data = utils.CreateTestData()
    #[trial, Ch, sample]  sample = rest:0～500, task:500～1500
    data = np.concatenate([
        EEG_T_blc.data('left')[:,:,0:1500],
        EEG_T_blc.data('right')[:,:,0:1500],
        EEG_T_blc.data('foot')[:,:,0:1500],
        EEG_T_blc.data('tongue')[:,:,0:1500]
        ],
        axis=0
    )
    #================================================
    #フィルタ処理
    print('フィルタ処理中')
    trialNum = data.shape[0]
    sampleNum = data.shape[2]
    # ローパスフィルター バタワース
    fs = 250
    fc = 50 #カットオフ周波数
    b, a = signal.butter(6, fc/(fs/2), "low")

    for i in range(trialNum):
        for j in range(ChNum):
            data[i,j,:] = signal.filtfilt(b, a, data[i,j,:])

    #電流源密度分布に変換
    print('電流源密度波形に変換中')
    data, k = utils.kCSD(data)

    #電流源密度分布のグラフを表示
    #電流密度分布を表示 CSDDisp(x座標, y座標, 推定した電流密度, チャンネル位置)
    #utils.CSDDisp(k.estm_x, k.estm_y, data_kCSD_tmp[:,:,0], ChPos) #(x座標, y座標, 表示したいデータ)

    #yeo-johnson変換
    #print('正規分布に近似中')
    #data = utils.YeoJohnson(data)

    #分散共分散行列を計算
    print('分散共分散行列計算中')
    #method には 'GL' or 'TVGL'
    cov_tmp, pre_tmp, u_tmp = utils.Calculate_Cov(data, epoch, method='TVGL')

    #restデータを削除
    cov, pre, u = (np.zeros((int(cov_tmp.shape[0]*2/3), ChNum, ChNum)), np.zeros((int(cov_tmp.shape[0]*2/3), ChNum, ChNum)), np.zeros((int(cov_tmp.shape[0]*2/3), ChNum)))
    n = 0
    for i in range(cov_tmp.shape[0]):
        if i%3 != 0:
            cov[n,:,:] = cov_tmp[i,:,:]
            pre[n,:,:] = pre_tmp[i,:,:]
            u[n,:] = u_tmp[i,:]
            n += 1

    #相関行列作成
    print('相関行列計算中')
    corr = utils.CalculateCorr(cov)

    # ネットワーク表示
    # print('ネットワーク表示')
    # utils.DispNetwork(corr[1,:,:])

    #距離行列作成
    print('距離行列計算中')
    kld = utils.KLD(cov, pre, u)
    # jsd = utils.JSD(cov, pre, u, data, epoch)

    # ヒートマップを表示
    print('ヒートマップ作成中')
    sns.heatmap(kld)
    plt.show()

    #次元削減
    print('次元削減中')
    dim = 2 #削減後の次元数

    #MDS
    mds = manifold.MDS(n_components=dim, random_state=0, dissimilarity="precomputed")
    graph = mds.fit_transform(kld)
    """
    #Laplacian Eigenmaps
    spca = manifold.SpectralEmbedding(n_components=dim)
    graph = spca.fit_transform(kld)
    """

    #分類
    print('クロスバリデーション（SVM）')
    y = np.repeat(range(classNum), graph.shape[0]/classNum).astype(int)
    utils.Classfication(graph, y, kNum, sj)

    #グラフ表示
    if dim <= 3:
        utils.GraphDisp(graph, dim, classNum)
