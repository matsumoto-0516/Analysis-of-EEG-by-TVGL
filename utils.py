import numpy as np

from kcsd import KCSD2D
from scipy import stats
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.metrics import accuracy_score
from regain.covariance import LatentTimeGraphicalLasso as TVGL

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

#解析===========================================================================================================
#電流源密度波形に変換
def kCSD(data):
    trialNum = data.shape[0]
    ChNum = data.shape[1]
    sampleNum = data.shape[2]

    ChPos = np.array([
        [4,6],
        [2,5],
        [3,5],
        [4,5],
        [5,5],
        [6,5],
        [1,4],
        [2,4],
        [3,4],
        [4,4],
        [5,4],
        [6,4],
        [7,4],
        [2,3],
        [3,3],
        [4,3],
        [5,3],
        [6,3],
        [3,2],
        [4,2],
        [5,2],
        [4,1]
    ])

    h = 10.
    sigma = 1

    xmax=8.0    #電流密度推定範囲のx最大値
    ymax=7.0    #電流密度推定範囲のy最大値
    step_x = xmax/100
    step_y = ymax/100
    Ch_x, Ch_y = (np.zeros(ChPos.shape[0]), np.zeros(ChPos.shape[0]))
    for i in range(ChPos.shape[0]):
        Ch_x[i] = np.round(ChPos[i,0]/step_x)
        Ch_y[i] = np.round(ChPos[i,1]/step_y)

    data_kCSD = np.zeros((trialNum, ChNum, sampleNum))
    for i in range(trialNum):
        k = KCSD2D(ChPos, data[i,:,:], h=h, sigma=sigma,
            xmin=0.0, xmax=xmax,
            ymin=0.0, ymax=ymax,
            n_src_init=1000, src_type='gauss', R_init=1.)

        data_kCSD_tmp = k.values('CSD')

        for j in range(ChPos.shape[0]):
            data_kCSD[i,j,:] = data_kCSD_tmp[int(Ch_x[j]),int(Ch_y[j]),:]

    return data_kCSD, k

#データ成型
def Molding(data, epoch):
    Ch = data.shape[1]
    div = int(data.shape[2]/epoch)
    moldedData = np.zeros((data.shape[0]*epoch, Ch, div))
    k = 0
    for i in range(data.shape[0]):
        for j in range(epoch):
            moldedData[k,:,:] = data[i, :, (div*j):(div*(j+1))]
            k += 1

    return moldedData

#分散共分散行列、精度行列、中心を求める（data = [trial][Ch][sample]）
def Calculate_Cov(data, epoch, method):
    #epoch数が誤っているとき
    if data.shape[2]%epoch != 0:
        print('epoch：データ点数を割り切れる値に変更してください')

    else:
        trialNum = data.shape[0] #trial数
        Ch = data.shape[1]  #チャネル数
        div = int(data.shape[2]/epoch)  #1epochあたりのデータ数
        #GLを使用するとき
        if method == 'GL':
            #cov:分散共分散行列, pre:精度行列, u:中心
            cov_tmp, pre_tmp, u_tmp = (np.zeros((epoch,Ch,Ch)), np.zeros((epoch,Ch,Ch)), np.zeros((epoch,Ch)))
            cov, pre, u = (np.empty((0,Ch,Ch)), np.empty((0,Ch,Ch)), np.empty((0,Ch)))
            #共分散行列
            for i in range(trialNum):
                for j in range(epoch):
                    cov_tmp[j,:,:] = np.cov(data[i,:,(div*j):(div*(j+1))])
                    pre_tmp[j,:,:] = np.linalg.inv(cov_tmp[j,:,:])
                    u_tmp[j:] = np.mean(data[i,:,(div*j):(div*(j+1))], axis=1)
                cov = np.concatenate([cov, cov_tmp], 0)
                pre = np.concatenate([pre, pre_tmp], 0)
                u = np.concatenate([u, u_tmp], 0)
            return cov, pre, u

        #TVGLを使用するとき
        elif method == 'TVGL':
            #cov:分散共分散行列, pre:精度行列, u:中心
            u_tmp = np.zeros((epoch,Ch))
            cov, pre, u = (np.empty((0,Ch,Ch)), np.empty((0,Ch,Ch)), np.empty((0,Ch)))
            y = np.repeat(range(epoch), div).astype(int) #epoch分割用のyを求める
            for i in range(trialNum):
                for j in range(epoch):
                    u_tmp[j:] = np.mean(data[i,:,(div*j):(div*(j+1))], axis=1)
                u = np.concatenate([u, u_tmp], 0)
                mdl = TVGL(max_iter=1000).fit(data[i,:,:].T,y) #data = [行:サンプル、 列:次元]の形式にして入力
                cov = np.concatenate([cov, mdl.covariance_], 0)
                pre = np.concatenate([pre, mdl.precision_], 0)
            return cov, pre, u

#距離を計算
def Distance(P, Q):
    sigma1 = P['cov']
    sigma2 = Q['cov']
    #lambda2 = np.linalg.inv(sigma2)
    lambda2 = Q['pre']

    u1 = P['u']
    u2 = Q['u']

    d = sigma1.shape[0]
    tmp1 = np.log(np.linalg.det(sigma2)/np.linalg.det(sigma1))
    tmp2 = np.trace(np.dot(lambda2, sigma1))
    tmp3 = np.dot(np.dot((u2 - u1), lambda2), (u2 - u1))

    return  1/2 * (tmp1 + tmp2 + tmp3 - d)

# KLダイバージェンスを計算
def KLD(cov, pre, u):
    dataNum = cov.shape[0]  #trial数 × epoch
    kld = np.zeros((dataNum, dataNum))
    #KLダイバージェンスを計算
    print('KLダイバージェンス計算中')
    for i in range(dataNum):
        for j in range(dataNum):
            if i == j:
                kld[i,j] = 0

            elif i > j:
                P = {'cov':cov[i,:,:],
                    'pre':pre[i,:,:],
                    'u':u[i,:]
                    }
                Q = {'cov':cov[j,:,:],
                    'pre':pre[j,:,:],
                    'u':u[j,:]
                    }

                kld[i,j] = Distance(P,Q)
                kld[j,i] = kld[i,j]

            """
            kld_mean = np.zeros((dataNum, dataNum))
            for i in range(dataNum):
                for j in range(dataNum):
                    if i > j:
                        kld_mean[i,j] = (kld[i,j]+kld[j,i])/2
                        kld_mean[j,i] = kld_mean[i,j]
            """

    return kld

#JSダイバージェンスを計算 平均分布改良
def JSD(cov, pre, u, data, epoch):
    dataNum = cov.shape[0]  #trial数 × epoch
    jsd = np.zeros((dataNum, dataNum))

    #平均分布計算のためのデータ成型
    moldedData = Molding(data, epoch)

    #JSダイバージェンスを計算
    for i in range(dataNum):
        for j in range(dataNum):
            if i == j:
                jsd[i,j] = 0

            else:
                P = {'cov':cov[i,:,:],
                    'pre':pre[i,:,:],
                    'u':u[i,:]
                    }
                Q = {'cov':cov[j,:,:],
                    'pre':pre[j,:,:],
                    'u':u[j,:]
                    }

                #平均分布の分散共分散行列、精度行列、平均を計算
                meanDist = (moldedData[i,:,:]+moldedData[j,:,:]) / 2
                cov_meanDist = np.cov(meanDist)
                pre_meanDist = np.linalg.inv(cov_meanDist)
                u_meanDist = np.mean(meanDist, axis=1)

                M = {'cov':cov_meanDist,
                    'pre':pre_meanDist,
                    'u':u_meanDist
                    }

                jsd[i,j] = (Distance(P,M)+Distance(Q,M))/2
    return jsd


#データを正規分布に近づける（data = [trial数][チャネル][サンプル]  成形後のデータ）
def YeoJohnson(data):
    dataNum = data.shape[0]
    Ch = data.shape[1]
    sampleNum = data.shape[2]
    conversionData = np.zeros((dataNum, Ch, sampleNum))
    for i in range(dataNum):
        for j in range(Ch):
            conversionData[i,j,:], lambda_tmp = stats.yeojohnson(data[i,j,:])

    return conversionData

#相関行列からネットワークを出力する
def DispNetwork(corr):
    #ノード位置（チャネル配置）
    ChPos = {
        1 : (13,21),
        2 : (7,19),
        3 : (10,18),
        4 : (13,17),
        5 : (16,18),
        6 : (19,19),
        7 : (1,14.5),
        8 : (5,14),
        9 : (9,13.5),
        10 : (13,13),
        11 : (17,13.5),
        12 : (21,14),
        13 : (25,14.5),
        14 : (5,8),
        15 : (9,8.5),
        16 : (13,9),
        17 : (17,8.5),
        18 : (21,8),
        19 : (10,4),
        20 : (13,5),
        21 : (16,4),
        22 : (13,1)
    }

    #無向グラフ定義
    G = nx.Graph()

    #ノードの追加
    for i in ChPos.keys():
        G.add_node(i, pos=ChPos[i])

    #閾値（0～1.0）：これ以下の相関は表示しない
    threshold = 0.95

    #エッジリスト
    Ch = corr.shape[0]
    weight = []
    connection = np.zeros(Ch)
    for i in range(Ch):
        for j in range(Ch):
            if (np.abs(corr[i,j])>threshold) and (corr[i,j] != 1):
                G.add_edge(i+1, j+1)
                weight.append(corr[i,j])
                connection[i] += 1
                connection[j] += 1
    """
    #エッジカラーマップ用のリスト作成
    edgeColors = []
    cmap = plt.cm.get_cmap('seismic')   #カラーマップ取得
    for w in weight:
        edgeColors.append(cmap((w+1)/2))
    """
    #ノードサイズを決定（接続されているエッジが多いほど大きい）
    nodeSize = 10 + (np.power(connection,2)*3)
    #for w in weight:
    #    nodeSize.append(0.5+(connection/2))

    #エッジカラーを決定（正:赤色   負:青色）
    edgeColors = []
    """
    for w in weight:
        edgeColors.append(((w-threshold)/(1-threshold),0,0,1))
    """
    for w in weight:
        #正：赤色
        if w >= 0:
            edgeColors.append((1,0,0,1))
        #負：青色
        elif w < 0:
            edgeColors.append((0,0,1,1))

    #エッジの幅を決定（相関が強いほど太い）
    edgeWidth = []
    for w in weight:
        edgeWidth.append(((np.abs(w)-threshold)/(1-threshold))*5)
        #edgeWidth.append(1)

    #ネットワークを表示
    nx.draw_networkx_nodes(G, ChPos, alpha=0.4, node_color='w', node_size=nodeSize)    #ノードを出力
    nx.draw_networkx_labels(G, ChPos, font_color='k')   #ノードのラベル
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")  #ノードの枠線を設定
    nx.draw_networkx_edges(G, ChPos, alpha=0.6, edge_color=edgeColors, width=edgeWidth)  #エッジを出力
    plt.show()

#分散共分散行列から相関行列を求める
def CalculateCorr(cov):
    dataNum = cov.shape[0]  #trial数 × epoch
    Ch = cov.shape[1]       #チャネル数
    corr = np.zeros((dataNum,Ch,Ch))
    for i in range(dataNum):
        cov_diag = np.expand_dims(np.diag(cov[i,:,:]), axis=0)
        S = np.sqrt(cov_diag.T * cov_diag)
        corr[i,:,:] = cov[i,:,:]/S

    return corr

#SVMクロスバリデーション(2クラス)
#X:学習用データ, y:ラベルデータ, kNum:フォールド数
def Classfication(X, y, kNum, sj):
    print('分類精度計算中')
    num = int(X.shape[0] // kNum)
    meanAccuracy = 0
    X_tmp = X.copy()
    y_tmp = y.copy()
    #データをランダムに並び替え
    for l in [X_tmp, y_tmp]:
        np.random.seed(1)
        np.random.shuffle(l)
    #クロスバリデーション
    for i in range(kNum):
        X_train = np.delete(X_tmp, slice(i*num, (i+1)*num), 0)
        X_test = X_tmp[i*num : (i+1)*num, :]
        y_train = np.delete(y_tmp, slice(i*num, (i+1)*num), 0)
        y_test = y_tmp[i*num : (i+1)*num]

        #学習
        #model = SVC(kernel='linear')
        model = SVC()
        model.fit(X_train, y_train)

        #テストデータに対する制度
        pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred_test)
        print(str(sj+1)+'人目（'+str(i+1)+'回目）：'+str(accuracy))
        meanAccuracy += accuracy 
        """
        #境界を表示(2次元) 2次元に次元削減したデータのみに使用可=========================================================
        gridsize = 0.02
        margin = 0.5
        #グラフの表示範囲を決定
        x1_min = X[:, 0].min() - margin
        x1_max = X[:, 0].max() + margin
        x2_min = X[:, 1].min() - margin
        x2_max = X[:, 1].max() + margin
        #グリッドポイントを作成（グラフをグリッドの集合として考える）
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, gridsize),
                            np.arange(x2_min, x2_max, gridsize))
        #各グリッドポイントに予測を適用
        Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        #予測結果を元のグリッドポイントのデータサイズに変換
        Z = Z.reshape(xx1.shape)
        #テストデータをプロット
        for n in range(X_test.shape[0]):
            if y_test[n] == 0:
                plt.scatter(X_test[n,0], X_test[n,1], c='r')
            elif y_test[n] == 1:
                plt.scatter(X_test[n,0], X_test[n,1], c='b')
        #plt.legend()
        plt.contourf(xx1, xx2, Z, cmap="RdBu", alpha=0.1)
        plt.show()
        """
    meanAccuracy = meanAccuracy/kNum
    print('平均精度は'+str(meanAccuracy))
#===============================================================================================================

#グラフ==========================================================================================================
#kCSDにより推定した電流分布グラフを表示
def CSDDisp(xx, yy, zz, ChPos):
    title='Estimated CSD without CV'
    cmap=cm.bwr
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(zz))
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.scatter(ChPos[:, 0], ChPos[:, 1], 10, c='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    plt.show()

# グラフ表示
def GraphDisp(graph, dim, classNum):
    print('グラフ作成中')
    colors = ['r', 'b', 'g', 'k']
    taskNum = int(graph.shape[0] / classNum)
    # 2次元グラフ
    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # プロット
        for i in range(classNum):
            ax.scatter(graph[taskNum*i:taskNum*(i+1), 0], graph[taskNum*i:taskNum*(i+1), 1], color=colors[i])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    # 3次元グラフ
    elif dim == 3:
        # グラフのスタイル
        sns.set_style("darkgrid")
        # グラフの枠
        fig = plt.figure()
        ax = Axes3D(fig)
        # プロット
        for i in range(classNum):
            ax.plot(graph[taskNum*i:taskNum*(i+1), 0], graph[taskNum*i:taskNum*(i+1), 1], graph[taskNum*i:taskNum*(i+1), 2], color = colors[i], marker="o", linestyle="None")

        # ラベル
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

# 境界線付き2次元グラフ
def GraphDispWithBorder(graph, y):
    taskNum = int(graph.shape[0] / 2)
    #学習
    model = SVC()
    model.fit(graph, y)
    #境界を表示(2次元)
    gridsize = 0.02
    margin = 0.5
    #グラフの表示範囲を決定
    x1_min = graph[:, 0].min() - margin
    x1_max = graph[:, 0].max() + margin
    x2_min = graph[:, 1].min() - margin
    x2_max = graph[:, 1].max() + margin
    #グリッドポイントを作成（グラフをグリッドの集合として考える）
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, gridsize),
                        np.arange(x2_min, x2_max, gridsize))
    #各グリッドポイントに予測を適用
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    #データをプロット
    plt.scatter(graph[0:taskNum,0], graph[0:taskNum,1], label='left', c='r')
    plt.scatter(graph[taskNum::,0], graph[taskNum::,1], label='rest', c='b')
    plt.legend()
    plt.contourf(xx1, xx2, Z, cmap="RdBu", alpha=0.1)
    """
    #データ点に番号を付ける
    for i in range(graph.shape[0]):
        plt.annotate(str(i), xy=(graph[i,0],graph[i,1]))
    """
    plt.show()

#テスト用データ作成
def CreateTestData():
    mean = np.array([0,0,0,0])

    cov1 = np.array([[10 ,8 ,8,0],
                    [8 ,10 ,8 ,0],
                    [8 ,8 ,10 ,0],
                    [0 ,0 ,0 ,10]])

    cov2 = np.array([[10 ,0 ,0,0],
                    [0 ,10 ,8 ,8],
                    [0 ,8 ,10 ,8],
                    [0 ,8 ,8 ,10]])

    cov3 = np.array([[10 ,0 ,8 ,8],
                    [0 ,10 ,0 ,0],
                    [8 ,0 ,10 ,8],
                    [8 ,0 ,8 ,10]])

    cov4 = np.array([[10,8 ,0 ,8],
                    [8 ,10 ,0 ,8],
                    [0 ,0 ,10 ,0],
                    [8 ,8 ,0 ,10]])

    size = 10000
    data1 = np.random.multivariate_normal(mean, cov1, size=size)
    data2 = np.random.multivariate_normal(mean, cov2, size=size)
    data3 = np.random.multivariate_normal(mean, cov3, size=size)
    data4 = np.random.multivariate_normal(mean, cov4, size=size)

    data = np.zeros((4,4,size))
    data[0,:,:] = data1.T
    data[1,:,:] = data2.T
    data[2,:,:] = data3.T
    data[3,:,:] = data4.T

    return data