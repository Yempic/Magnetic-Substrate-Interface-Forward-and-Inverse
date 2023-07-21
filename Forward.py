# coding=utf-8
# 导入相关的库
from osgeo import gdal_array as ga
import matplotlib.pyplot as plt
from geoist.inversion import mesh
from geoist import gridder
from geoist.pfm import prism, giutils
from geoist.vis import giplt


model = ga.LoadFile(r'C:\Users\Administrator\Desktop\大同汇总\模型及生成的磁异常\model1(sample1)\model_1.tif')
model = -(model-model.max())-1005
area = (40246, 40585, 113631, 114256)
x, y = gridder.regular(area, model.shape)
plt.figure()
plt.title("Initial Model (m)")
plt.axis('scaled')
giplt.contourf(y, x[::-1], model, model.shape, 15)
plt.colorbar()
plt.xlabel('latitude (°)')
plt.ylabel('longitude (°)')
giplt.m2km()
plt.show()

# 正演生成磁异常
area = (4455255, 4495488, 468211, 539238)
x, y = gridder.regular(area, model.shape)
inc, dec = 60.18, -6.88
height = model.reshape(-1)
nodes = (x, y, height)
nodes_tra = (x, y, height)
reference = 4000
relief = mesh.PrismRelief(reference, model.shape, nodes)
mag = giutils.ang2vec(0.02, inc, dec)
xp, yp, zp = gridder.regular(area, (129, 232), z=-1005.265)
tf = prism.tf(xp, yp, zp, relief, inc, dec, mag)
plt.figure()
plt.title("Total-field anomaly (nT)")
plt.axis('scaled')
giplt.contourf(yp, xp[::-1], tf, (129, 232), 15)
plt.colorbar()
plt.xlabel('East y (km)')
plt.ylabel('North x (km)')
giplt.m2km()
plt.show()

