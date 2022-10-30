import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from SimplexBSplines import SSmodel as ss


# Making triangulation grid
x1_range = 2
x2_range = 2
tri_size = 4
x1 = np.linspace(-x1_range/2, x1_range/2, tri_size)
x2 = np.linspace(-x2_range/2, x2_range/2, tri_size)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_flat = x1_mesh.flatten()
x2_flat = x2_mesh.flatten()
points = np.vstack([x1_flat, x2_flat]).T

# Making data
data_res = 50
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

# Make model
r = 1
poly_order = 6
Model = ss.modelFromData(X_fit, Y_fit, points, poly_order, r)

# Save and load model
Model.save('Models/model3')
NewModel = ss.modelFromCsv('Models/model3')

# Evaluate model for plotting
Y_est = NewModel.eval(X_model)

# Plot triangulation grid
plt.figure()
plt.triplot(Model.Tri.points[:,0], Model.Tri.points[:,1], Model.Tri.simplices)
plt.plot(Model.Tri.points[:,0], Model.Tri.points[:,1], 'o')
plt.title('Triangulation')

# Plot data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X1_mesh, X2_mesh, Y_fit)
plt.title('Data used for fitting')

# Plot model
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1_mesh_model, X2_mesh_model, Y_est.reshape((plot_res, plot_res)), rstride = 1, cstride = 1, cmap=cm.coolwarm)
plt.title('Simplex B-spline')
plt.show()
