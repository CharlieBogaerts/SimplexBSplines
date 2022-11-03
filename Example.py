import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from SimplexBSplines import Modeling as ss
from SimplexBSplines.Tools import Random_PartitionData


# Making triangulation grid
x1_range = 2
x2_range = 2
tri_size = 4
x1 = np.linspace(-x1_range/2, x1_range/2, tri_size)
x2 = np.linspace(-x2_range/2, x2_range/2, tri_size)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
x1_flat = x1_mesh.flatten()
x2_flat = x2_mesh.flatten()
points = np.vstack([x1_flat, x2_flat]).T    #points in 2d space used for triangulation

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

# split data into training and test sets
trainSet, testSet = Random_PartitionData(np.hstack((X_fit, Y_fit.reshape(-1,1))), 0.8)

X_fit_train = trainSet[0][:,:-1]
Y_fit_train = trainSet[0][:,-1]

X_fit_test = testSet[0][:,:-1]
Y_fit_test = testSet[0][:,-1]

trainIdx = trainSet[1]
testIdx = testSet[1]

# Making arrays for plotting
plot_res = 50
X1_model = np.linspace(-x1_range/2, x1_range/2, plot_res)
X2_model = np.linspace(-x2_range/2, x2_range/2, plot_res)
X1_mesh_model, X2_mesh_model = np.meshgrid(X1_model,X2_model)
X1_flat_model = X1_mesh_model.flatten()
X2_flat_model = X2_mesh_model.flatten()
X_model = np.vstack([X1_flat_model, X2_flat_model]).T

# Make model
r = 1               # order of continuity. 0 -> same f(x), 1 -> same f'(x), etc
poly_order = 6      # order of the simplex polynomials
Model = ss.modelFromData(X_fit_train, Y_fit_train, points, poly_order, r)   #Estimate the model from data

# Save and load model
Model.save('Models/model1')                 # save the model in folder 'Models'
NewModel = ss.modelFromCsv('Models/model1') # load 'model1' from the folder 'Models'

# Print model performance metrics for testing and training data
print (f'[ INFO ] Model coefficient of determination (R2) for training data : {ss._CoeffOfDetermination_R2_model(X_fit_train, Y_fit_train, NewModel)}')
print (f'[ INFO ] Model coefficient of determination (R2) for testing data  : {ss._CoeffOfDetermination_R2_model(X_fit_test, Y_fit_test, NewModel)}')

print (f'[ INFO ] Model RMSE for training data : {ss._RMSE_model(X_fit_train, Y_fit_train, NewModel)}')
print (f'[ INFO ] Model RMSE for testing data  : {ss._RMSE_model(X_fit_test, Y_fit_test, NewModel)}')

# Evaluate model for plotting
Y_est = NewModel.eval(X_model)          # calculate the estimated Y values according to the model

# Plot triangulation grid
# plt.figure()
# plt.triplot(Model.Tri.points[:,0], Model.Tri.points[:,1], Model.Tri.simplices)
# plt.plot(Model.Tri.points[:,0], Model.Tri.points[:,1], 'o')
# plt.title('Triangulation')

# Plot data
zmin = np.min(Y_fit) - 0.1*(np.max(Y_fit) - np.min(Y_fit))

fig = plt.figure(figsize=(13,6))
ax1 = plt.subplot(121, projection='3d')
ax1.scatter(X_fit_train[:,0],X_fit_train[:,1],Y_fit_train,alpha = 0.4, marker = 'x', label = 'Training data')
ax1.scatter(X_fit_test[:,0],X_fit_test[:,1],Y_fit_test,   alpha = 0.4, label = 'Test data')
ax1.triplot(Model.Tri.points[:,0], Model.Tri.points[:,1], Model.Tri.simplices, zs = zmin,label = 'Simplex edges')
ax1.plot(Model.Tri.points[:,0], Model.Tri.points[:,1], 'o', zs = zmin, label = 'Simplex vertices')
plt.legend()
plt.title('Data used for fitting')

# Plot model
#fig = plt.figure()
ax2 = plt.subplot(122, projection='3d')
ax2.plot_surface(X1_mesh_model, X2_mesh_model, Y_est.reshape((plot_res, plot_res)), rstride = 1, cstride = 1, cmap=cm.coolwarm)
plt.title('Simplex B-spline')

# fancy stuff to co-rotate 3D subplots
def on_move(event):
    if event.inaxes == ax1:
        ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    elif event.inaxes == ax2:
        ax1.view_init(elev=ax2.elev, azim=ax2.azim)
    else:
        return
    fig.canvas.draw_idle()

c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
