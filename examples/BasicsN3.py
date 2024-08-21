##Example script of how to estimate a simplex b spline when the independent
##variable has 3 dimensions (N=3). Note that a uniform grid can result in
##degenerate simplices. To get around this, the grid points are give a random
##small offset, making the grid nonuniform. For plotting a slice along the
##third axis is taken, so the third plot axis can contain the simplex b spline
##value.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import sys
sys.path.append('G:/My Drive/Documents/Python3/MyPackages/simplexbsplines')
import simplexbsplines as sbs

np.random.seed(0)


x1_range_min = -1
x1_range_max = 1
x2_range_min = -1
x2_range_max = 1
x3_range_min = -1
x3_range_max = 1

# Making triangulation grid
tri_size = 3
x1_grid_arr = np.linspace(x1_range_min, x1_range_max, tri_size)
x2_grid_arr = np.linspace(x2_range_min, x2_range_max, tri_size)
x3_grid_arr = np.linspace(x3_range_min, x3_range_max, tri_size)
x1_grid_mesh, x2_grid_mesh, x3_grid_mesh = np.meshgrid(x1_grid_arr, x2_grid_arr, x3_grid_arr)
x1_grid_flat = x1_grid_mesh.flatten()
x2_grid_flat = x2_grid_mesh.flatten()
x3_grid_flat = x3_grid_mesh.flatten()
X_grid_flat = np.vstack([x1_grid_flat, x2_grid_flat, x3_grid_flat]).T    #points in 2d space used for triangulation
X_grid_flat += np.random.uniform(-.1, .1, X_grid_flat.shape)


# Making data
data_res = 8
x1_data_arr = np.linspace(x1_range_min, x1_range_max, data_res)
x2_data_arr = np.linspace(x2_range_min, x2_range_max, data_res)
x3_data_arr = np.linspace(x3_range_min, x3_range_max, data_res)
x1_data_mesh, x2_data_mesh, x3_data_mesh = np.meshgrid(x1_data_arr, x2_data_arr, x3_data_arr)
x1_data_flat = x1_data_mesh.flatten()
x2_data_flat = x2_data_mesh.flatten()
x3_data_flat = x3_data_mesh.flatten()
X_data_flat = np.vstack([x1_data_flat, x2_data_flat, x3_data_flat]).T
Y_data = np.sqrt(x1_data_flat**2 + x2_data_flat**2)**x3_data_flat
Y_data_mesh = Y_data.reshape((data_res, data_res, data_res))

# Making arrays for plotting (slice along x-axis) eval = evaluate
eval_res = 40
x3_slice_nr = 1
x3_slice = x3_data_arr[x3_slice_nr]

x1_eval_arr = np.linspace(x1_range_min, x1_range_max, eval_res)
x2_eval_arr = np.linspace(x2_range_min, x2_range_max, eval_res)
x1_eval_mesh, x2_eval_mesh = np.meshgrid(x1_eval_arr, x2_eval_arr)
x1_eval_flat = x1_eval_mesh.flatten()
x2_eval_flat = x2_eval_mesh.flatten()
x3_eval_flat = np.ones(x1_eval_flat.size)*x3_slice
X_eval_flat = np.vstack([x1_eval_flat, x2_eval_flat, x3_eval_flat]).T

# Make triangulation mesh
Tri = sbs.Triangulation(X_grid_flat)

# Make model
r = 1               # order of continuity. 0 -> same f(x), 1 -> same f'(x), etc
poly_order = 3      # order of the simplex polynomials
Model = sbs.SSModel.from_data(X_data_flat, Y_data, Tri, poly_order, r)   #Estimate the model from data

# Evaluate model for plotting
Y_est = Model.eval(X_eval_flat)          # calculate the estimated Y values according to the model

# Plot data
zmin = np.min(Y_data) - 0.1*(np.max(Y_data) - np.min(Y_data))

fig = plt.figure(figsize=(13,6))
ax1 = plt.subplot(121, projection='3d')


ax1.scatter(x1_data_mesh[:,:,x3_slice_nr], x2_data_mesh[:,:,x3_slice_nr], Y_data_mesh[:,:,x3_slice_nr], alpha = 0.4, marker = 'x')
plt.legend()
plt.title('Data used for fitting')

# Plot model
#fig = plt.figure()
ax2 = plt.subplot(122, projection='3d')
ax2.plot_surface(x1_eval_mesh, x2_eval_mesh, Y_est.reshape((eval_res, eval_res)), rstride = 1, cstride = 1, cmap=cm.coolwarm)
plt.title('Simplex B-spline')

plt.show()
