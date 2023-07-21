# coding=utf-8
# 导入相关的库
from osgeo import gdal
from osgeo import gdal_array as ga
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft2,ifft2
import scipy.ndimage as ndimage
from PIL import Image
import geoist
from geoist.inversion import mesh
from geoist import gridder
from geoist.inversion import geometry
from geoist.pfm import prism,pftrans,giutils
from geoist.inversion.geometry import Prism
from geoist.vis import giplt
import pandas
from pandas import DataFrame
import imblearn
import joblib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_regression
from mpl_toolkits.mplot3d import Axes3D


# 任意缩放栅格数据
def zoom(x,shape):
    x = x.astype(np.float32)
    x = np.array(Image.fromarray(x).resize(shape))
    return x


# 读取模型
basin = ga.LoadFile(r'C:\Users\Administrator\Desktop\大同汇总\模型及生成的磁异常\model1(sample1)\model_1.tif')
area = (4455255, 4495488, 468211, 539238)
x, y = gridder.regular(area, basin.shape)
plt.figure()
plt.title("True model (m)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], basin, basin.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

# 读取相应磁异常
T = ga.LoadFile(r'C:\Users\Administrator\Desktop\大同汇总\模型及生成的磁异常\model1(sample1)\tf_1.tif')
area = (4455255, 4495488, 468211, 539238)
x, y = gridder.regular(area, T.shape)
plt.figure()
plt.title("Total-field anomaly (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], T, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

# X、Y、Z分量
X_vensor = []
Y_vensor = []
Z_vensor = []
T_abs = T.reshape(29928,)
for i in range(29928):
    vensor = giutils.ang2vec(T_abs[i], 60.18, -6.88)
    X_vensor.append(vensor[0])
    Y_vensor.append(vensor[1])
    Z_vensor.append(vensor[2])

plt.figure()
plt.title("Total-field anomaly Z vensor (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], Z_vensor, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

X_vensor = np.array(X_vensor).reshape(129, 232)
Y_vensor = np.array(Y_vensor).reshape(129, 232)
Z_vensor = np.array(Z_vensor).reshape(129, 232)

# 化极
inc, dec = 60.18, -6.88  # Geomagnetic field direction
sinc, sdec = 60.18, -6.88  # Source magnetization direction
area = (4455255, 4495488, 468211, 539238)
shape = T.shape
x, y = gridder.regular(area, shape)

data_at_pole = pftrans.reduce_to_pole(x, y, T, shape, inc, dec, sinc, sdec)

plt.figure()
plt.title("Reduced to the pole (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], data_at_pole, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

data_at_pole = np.array(data_at_pole).reshape(129, 232)

# 延拓
bgas_contf_200 = pftrans.upcontinue(x, y, data_at_pole, shape, 200)
bgas_contf_500 = pftrans.upcontinue(x, y, data_at_pole, shape, 500)

plt.figure()
plt.title("Upward continuation 500m (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], bgas_contf_500, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

bgas_contf_200 = np.array(bgas_contf_200).reshape(129, 232)
bgas_contf_500 = np.array(bgas_contf_500).reshape(129, 232)

# 方向导数
bgas_contf = giutils.nt2si(bgas_contf_500)
bgas_dx = pftrans.derivx(x, y, bgas_contf, shape)
bgas_dy = pftrans.derivy(x, y, bgas_contf, shape)
bgas_dz = pftrans.derivz(x, y, bgas_contf, shape)
bgas_dx = giutils.si2nt(bgas_dx)
bgas_dy = giutils.si2nt(bgas_dy)
bgas_dz = giutils.si2nt(bgas_dz)

plt.figure()
plt.title("Derivative in Z direction (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], bgas_dz, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

bgas_dx = np.array(bgas_dx).reshape(129, 232)
bgas_dy = np.array(bgas_dy).reshape(129, 232)
bgas_dz = np.array(bgas_dz).reshape(129, 232)

# 解析信号
bgas_tga = pftrans.tga(x, y, bgas_contf, shape)
bgas_tga = giutils.si2nt(bgas_tga)

plt.figure()
plt.title("Total gradient amplitude (nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], bgas_tga, T.shape, 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

bgas_tga = np.array(bgas_tga).reshape(129, 232)

# 九点均值
x, y = gridder.regular(area, (127, 230))
num = (T.shape[0] - 2) * (T.shape[1] - 2)
data_at_pole_mean = np.zeros(num,)
T_mean = np.zeros(num,)
data_at_pole = data_at_pole.reshape(T.shape)
mm = 0
for i in range(1, T.shape[0]-1):
    for j in range(1, T.shape[1]-1):
        data_at_pole_mean[mm] = (data_at_pole[i-1, j-1]+data_at_pole[i-1, j]+data_at_pole[i-1, j+1]+data_at_pole[i, j-1]
                                 + data_at_pole[i, j]+data_at_pole[i, j+1]+data_at_pole[i+1, j-1]+data_at_pole[i+1, j] +
                                 data_at_pole[i+1, j+1]) / 9
        T_mean[mm] = (T[i-1, j-1]+T[i-1, j]+T[i-1, j+1]+T[i, j-1]+T[i, j]+T[i, j+1]+T[i+1, j-1]+T[i+1, j]+T[i+1, j+1]) /9
        mm = mm + 1
plt.figure()
plt.title("Average of the closest nine points(nT)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], data_at_pole_mean, ((T.shape[0] - 2), (T.shape[1] - 2)), 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

T_mean = T_mean.reshape(127, 230)
data_at_pole_mean = data_at_pole_mean.reshape(127, 230)

num = (T.shape[0] - 2) * (T.shape[1] - 2)
X = np.zeros((num, 9))
m = 0
for i in range(1, T.shape[0]-1):
    for j in range(1, T.shape[1]-1):
#        X[m,0] = T[i,j]
        X[m, 0] = T_mean[i-1, j-1]
        X[m, 1] = data_at_pole[i, j]
        X[m, 2] = data_at_pole_mean[i-1, j-1]
#        X[m,4] = X_vensor[i,j]
        X[m, 3] = Y_vensor[i, j]
#        X[m,4] = Z_vensor[i,j]
#        X[m,7] = bgas_contf_200[i,j]
        X[m, 4] = bgas_contf_500[i, j]
        X[m, 5] = bgas_dx[i, j]
        X[m, 6] = bgas_dy[i, j]
        X[m, 7] = bgas_dz[i, j]
        X[m, 8] = bgas_tga[i, j]
        m = m + 1


def convert_label(x):
    if (x >= 0) & (x < 500):
        return 500
    if (x >= 500) & (x < 1000):
        return 1000
    if (x >= 1000) & (x < 1500):
        return 1500
    if (x >= 1500) & (x < 2000):
        return 2000
    if (x >= 2000) & (x < 2500):
        return 2500
    if (x >= 2500) & (x < 3000):
        return 3000
    if (x >= 3000) & (x < 3500):
        return 3500
    if (x >= 3500) & (x < 4000):
        return 4000


true_model = zoom(basin, (230, 127))
Y = true_model.reshape(29210, 1)
dataset = np.concatenate((X, Y), axis=1)

df = DataFrame({'T_mean': dataset[:, 0], 'data_at_pole': dataset[:, 1]
               , 'data_at_pole_mean': dataset[:, 2], 'Y_vensor': dataset[:, 3],
                'bgas_contf_500': dataset[:, 4], 'bgas_dx': dataset[:, 5], 'bgas_dy': dataset[:, 6]
               , 'bgas_dz': dataset[:, 7], 'bgas_tga': dataset[:, 8], 'label': dataset[:, 9], 'label_conv': dataset[:, 9]})
df['label_conv'] = df['label_conv'].transform(convert_label)
data = df.values

data_x = data[:, :-1]
data_y = data[:, -1]
print(data_x, data_y)
# 过采样
smote = SMOTE()
x2, y2 = smote.fit_resample(data_x, data_y)
X = x2[:, 0:-1]
Y = x2[:, -1]

XX = data_x[:, :-1]
for i in range(0, 1):
    max_ = X[:, i].max()
    min_ = X[:, i].min()
    print(X[:, i].max(), X[:, i].min())
    X[:, i] = (((X[:, i] - min_)*2)/(max_-min_))-1
    XX[:, i] = (((XX[:, i] - min_)*2)/(max_-min_))-1

# 分离训练数据和检验数据
X_train, X_test, y_train, y_test = train_test_split(X, Y)
rf = RandomForestRegressor(n_estimators=33, min_samples_split=4, min_samples_leaf=1, max_features=6)
# 训练模型
rf.fit(X_train, y_train)
# 计算拟合优度和平均绝对误差
print(rf.score(X_test, y_test))
