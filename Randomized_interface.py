# coding=utf-8
# 导入相关的库
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


# 导出栅格数据
def export_shange(x, filename):
    x = x.astype(np.float32)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(r'C:\Users\Administrator\Desktop\%s' % filename, x.shape[1], x.shape[0], 1, gdal
                                 .GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(x)
    out_band.FlushCache()


# 平滑
def smooth(x, n):
    x = x.astype(np.float32)
    for i in range(n):
        x = ndimage.median_filter(x,size = 25)
        x = x.astype(np.float32)
    return x


# 任意缩放栅格数据
def zoom(x, shape):
    x = x.astype(np.float32)
    x = np.array(Image.fromarray(x).resize(shape))
    return x


# 随机中点位移法   --------生成随机因子
n = 8
num_n = 2 ** n + 1
H = 0.95
D_std = 2
rand = np.random.random(num_n ** 2) - 0.5
rand = rand.reshape((num_n, num_n))
D = ((1 - 2 ** (2 * H - 2)) ** 0.5) * (2 ** (-H)) * D_std * rand
T = np.zeros(shape=(num_n, num_n), dtype=np.float16)
T[0, 0] = 0
T[num_n - 1, 0] = 0
T[0, num_n - 1] = 0
T[num_n - 1, num_n - 1] = 0
for i in range(0, n):
    m = int(2 ** n / (2 ** (i + 1)))
    P_mm = m
    for j in range(0, 2 ** i):
        if j == 0:
            T[m, m] = (T[0, 0] + T[2 * m, 0] + T[0, 2 * m] + T[2 * m, 2 * m]) / 4 + D[m, m]
            T[0, m] = (T[0, 0] + T[0, 2 * m]) / 2 + D[0, m]
            T[m, 0] = (T[0, 0] + T[2 * m, 0]) / 2 + D[m, 0]
            T[m, 2 * m] = (T[0, 2 * m] + T[2 * m, 2 * m]) / 2 + D[m, 2 * m]
            T[2 * m, m] = (T[2 * m, 0] + T[2 * m, 2 * m]) / 2 + D[2 * m, m]
        else:
            P_mm = int(P_mm + 2 ** (n - i))
            for k in range(0, j + 1):
                P_distance = k * (2 ** (n - i))
                X_zx = P_mm
                Y_zx = int(P_mm - P_distance)
                T_zs_ZX = T[X_zx - m, Y_zx - m]
                T_zx_ZX = T[X_zx + m, Y_zx - m]
                T_ys_ZX = T[X_zx - m, Y_zx + m]
                T_yx_ZX = T[X_zx + m, Y_zx + m]
                T[X_zx, Y_zx] = (T_zs_ZX + T_zx_ZX + T_ys_ZX + T_yx_ZX) / 4 + D[X_zx, Y_zx]
                T[X_zx, Y_zx - m] = (T_zs_ZX + T_zx_ZX) / 2 + D[X_zx, Y_zx - m]
                T[X_zx, Y_zx + m] = (T_ys_ZX + T_yx_ZX) / 2 + D[X_zx, Y_zx + m]
                T[X_zx - m, Y_zx] = (T_zs_ZX + T_ys_ZX) / 2 + D[X_zx - m, Y_zx]
                T[X_zx + m, Y_zx] = (T_zx_ZX + T_yx_ZX) / 2 + D[X_zx + m, Y_zx]
                X_ys = int(P_mm - P_distance)
                Y_ys = P_mm
                T_zs_YS = T[X_ys - m, Y_ys - m]
                T_zx_YS = T[X_ys + m, Y_ys - m]
                T_ys_YS = T[X_ys - m, Y_ys + m]
                T_yx_YS = T[X_ys + m, Y_ys + m]
                T[X_ys, Y_ys] = (T_zs_YS + T_zx_YS + T_ys_YS + T_yx_YS) / 4 + D[X_ys, Y_ys]
                T[X_ys, Y_ys - m] = (T_zs_YS + T_zx_YS) / 2 + D[X_ys, Y_ys - m]
                T[X_ys, Y_ys + m] = (T_ys_YS + T_yx_YS) / 2 + D[X_ys, Y_ys + m]
                T[X_ys - m, Y_ys] = (T_zs_YS + T_ys_YS) / 2 + D[X_ys - m, Y_ys]
                T[X_ys + m, Y_ys] = (T_zx_YS + T_yx_YS) / 2 + D[X_ys + m, Y_ys]
# 归一化
Zp = (((T - T.min())) / (T.max() - T.min()))
Zp = smooth(Zp, 4)
x = np.array(range(0, 257))
y = np.array(range(0, 257))
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, Zp, cmap=plt.get_cmap('RdYlBu'))
plt.show()

