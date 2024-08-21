##Example script of how to estimate a simplex b spline when the independent
##variable has 3 dimensions (N=3). First a grid is generated for triangulation.
##Then data is generated to use for fitting the simplex b spline, and then
##the arrays for plotting. A triangulation object is made that is used by the
##simplex b spline model estimator. Both data and model are then plotted.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import sys
sys.path.append('G:/My Drive/Documents/Python3/MyPackages/simplexbsplines')
import simplexbsplines as sbs


# Making triangulation grid
x1_range = 2
x2_range = 2
tri_size = 3
x1 = np.linspace(-x1_range/2, x1_range/2, tri_size)
x2 = np.linspace(-x2_range/2, x2_range/2, tri_size)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_flat = x1_mesh.flatten()
x2_flat = x2_mesh.flatten()
points = np.vstack([x1_flat, x2_flat]).T    #points in 2d space used for triangulation

# Making data
data_res = 30
X1 = np.linspace(-x1_range/2, x1_range/2, data_res)
X2 = np.linspace(-x2_range/2, x2_range/2, data_res)
X1_mesh, X2_mesh = np.meshgrid(X1, X2)
X1_flat = X1_mesh.flatten()
X2_flat = X2_mesh.flatten()
X_fit = np.vstack([X1_flat, X2_flat]).T
R = X1_flat**2 + X2_flat**2
Y_fit = np.sin(10*R)/R


# Making arrays for plotting
plot_res = 50
X1_model = np.linspace(-x1_range/2, x1_range/2, plot_res)
X2_model = np.linspace(-x2_range/2, x2_range/2, plot_res)
X1_mesh_model, X2_mesh_model = np.meshgrid(X1_model,X2_model)
X1_flat_model = X1_mesh_model.flatten()
X2_flat_model = X2_mesh_model.flatten()
X_model = np.vstack([X1_flat_model, X2_flat_model]).T

# Make triangulation mesh
Tri = sbs.Triangulation(points)

# Make model
r = 1               # order of continuity. 0 -> same f(x), 1 -> same f'(x), etc
poly_order = 6      # order of the simplex polynomials
Model = sbs.SSModel.from_data(X_fit, Y_fit, Tri, poly_order, r)   #Estimate the model from data

# Save and load model
sbs.save_model(Model, 'Models/model1')                 # save the model in folder 'Models'
NewModel = sbs.open_model('Models/model1') # load 'model1' from the folder 'Models'

# Evaluate model for plotting
Y_est = NewModel.eval(X_model)          # calculate the estimated Y values according to the model

# Plot data
zmin = np.min(Y_fit) - 0.1*(np.max(Y_fit) - np.min(Y_fit))

fig = plt.figure(figsize=(13,6))
ax1 = plt.subplot(121, projection='3d')
ax1.scatter(X_fit[:,0],X_fit[:,1],Y_fit,alpha = 0.4, marker = 'x')
ax1.triplot(Model.Tri.points[:,0], Model.Tri.points[:,1], Model.Tri.simplices, color = 'm', zs = zmin,label = 'Simplex edges')
ax1.plot(Model.Tri.points[:,0], Model.Tri.points[:,1], 'o', color = 'g', zs = zmin, label = 'Simplex vertices')
plt.legend()
plt.title('Data used for fitting')

# Plot model
#fig = plt.figure()
ax2 = plt.subplot(122, projection='3d')
ax2.plot_surface(X1_mesh_model, X2_mesh_model, Y_est.reshape((plot_res, plot_res)), rstride = 1, cstride = 1, cmap=cm.coolwarm)
plt.title('Simplex B-spline')

plt.show()
