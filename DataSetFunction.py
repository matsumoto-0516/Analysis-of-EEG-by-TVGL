import numpy as np

# データクラス
class DATA:

    def __init__(self, data, axiskeys):
        if data.ndim != len(axiskeys):
            print('dataの次元とaxiskeysの数が合いません。')
            sys.exit(1)

        self.axiskeys = axiskeys
        self.shape = data.shape
        self.data = data
        self.dim = data.ndim


    # キーを入力、axisを返す[OK]
    def FindAxis(self, key):
        if type(key) is str:
            return self.axiskeys.index(key)
        elif type(key) is list or type(key) is np.ndarray:
            return [list(self.axiskeys).index(k) for k in key]


    # データの軸を任意の順番で入れ替え
    def Transpose(self, newaxis):
        try:
            # 新しくaxiskeysを更新するために、newaxisをすべてキーに
            for i, na in enumerate(newaxis):
                newaxis[i] = self.FindKey(na) if type(na) is int else na
        except TypeError as err:
            print('リスト型である必要があります。>', err)
            sys.exit(1)

        #ソートする順番のインデックを獲得(newaxisをすべてインデックスに)
        sort_idx = [self.FindAxis(na) for na in newaxis]
        #入れ替え作業
        self.axiskeys = [self.axiskeys[i] for i in sort_idx] #軸キー
        self.shape = tuple([self.shape[i] for i in sort_idx]) #確認用shape
        self.data = self.data.transpose(sort_idx) #データ
        self.dim = self.data.ndim


    def DeleteDATA(self, deleteindex, axis=0):
        self.data = np.delete(self.data, deleteindex, axis=axis)

        self.dim = self.data.ndim
        self.shape = self.data.shape

# データベースクラス
class DataBase:

    def __init__(self):
        self.Class = []

        self.ClassNames = []

        self.ClassLabel = []
        self.nextLabel = 0

        self.ClassNum = 0


    #データベース情報表示
    def info(self):
        print('=== DataBase Information =========')
        print(f'Class:{self.ClassNames}')
        print(f'Label:{self.ClassLabel}')
        print(f'ClassNum:{self.ClassNum}')
        print(f'DataShape -----------------------')
        for i in range(self.ClassNum):
            print(f'{self.ClassNames[i]}:(', end='')
            flag = 0
            for j in range(len(self.Class[i].axiskeys)):
                if flag == 0:
                    print(f'{self.Class[i].axiskeys[j]}={self.Class[i].data.shape[j]}', end='')
                    flag = 1
                else:
                    print(f', {self.Class[i].axiskeys[j]}={self.Class[i].data.shape[j]}', end='')
            else:
                print(')')
        print('----------------------------------')
        print('==================================')


    # クラスデータ追加
    def AddClassData(self, ClassName, axiskeys, data):
        self.Class.append(DATA(data, axiskeys))
        self.ClassNames.append(ClassName)

        self.ClassLabel.append(self.nextLabel)
        self.nextLabel = self.nextLabel + 1

        self.ClassNum = self.ClassNum + 1


    # クラスデータからデータ削除
    def ClassdataDelete(self, deleteindex, axis=0):
        for i in range(self.ClassNum):
            self.Class[i].DeleteDATA(deleteindex, axis=axis)


    # データの軸入れ替え
    def Transpose(self, axis):
        for i in range(self.ClassNum):
            self.Class[i].Transpose(axis)


    # データの参照
    def data(self, ClassName, index=None):
        if type(ClassName) is str:
            Classi = self.ClassNames.index(ClassName)
        elif type(ClassName) is int:
            Classi = ClassName

        if index == None:
            return self.Class[Classi].data
        elif type(index) is str:
            return self.Class[Classi].data[self.Class[Classi].FindAxis(index)]
        else:
            return self.Class[Classi].data[index]
