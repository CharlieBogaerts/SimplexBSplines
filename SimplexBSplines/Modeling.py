import numpy as np
import scipy as sp
import pickle as p
import Triangulation as tri
import MultiIndexSet as mis
import Tools as t
import SSmodel as ssm


def modelFromCsv(path):
    """
    Load simplex spline model from a set of csv files. These files can be created
    unsing the ssm.SSmodel.save() function.

    :param path: Model path and name. E.g. 'models/thrust_model'
    """
    Tri = p.load(open(path+ '/Triangulate.p', 'rb'))
    params = np.loadtxt(path+ '/params.csv', delimiter=',')
    poly_order = int(np.loadtxt(path+ '/misc.csv', delimiter=','))
    return ssm.SSmodel(Tri, params, poly_order)

def modelFromData(X, Y, points, poly_order, continuity):
    '''
    Estimate a simplex spline model based data, preferred order of the simplex
    polynomials  and the continuity order ('r' in the slides).

    :param X: (2D numpy array) Independent variable of data. Dimension
        should be 2 or higher. The 0th axis contains the different data points, and
        the 1st axis the entries of the induvidual data points
    :param Y: (1D numpy array) Dependent variable of data corresponding to the
        independent variables X. 
    :param poly_order: (int >=1) Order of the simplex polynomials ('d' in the
        lecture slides)
    :param continuity: (int >=0) Order of continuity ('r' in the slides).
    '''
    X, Y, nan_rows_found = removeNans(X, Y)
    if nan_rows_found:
        print('Warning: data points containing nan values have been removed!')
    nr_vertices, dimension = X.shape
    nr_vertices2 = Y.shape[0]
    nr_points, dimension2 = points.shape
    if nr_vertices != nr_vertices2:
        raise ValueError("'X' and 'Y' do not have the same length.")
    if dimension != dimension2:
        raise ValueError("dimension of 'X' and 'points' do not match.")
    MIS = mis.makeMISet(dimension, poly_order)
    Tri = tri.Triangulation(points)
    B_matrix, Y_vec = makeRegressionMats(Tri, MIS, X, Y)
    H = makeContinuityMat(Tri, MIS, continuity)
    params = t.ECLQS(B_matrix, Y_vec, H)
    return ssm.SSmodel(Tri, params, poly_order)

def removeNans(X, Y):
    """
    Remove data points that have have a nan value in either the X or Y arrays.
    """
    nan_rows_X = np.any(np.isnan(X), axis=1)
    nan_rows_Y = np.isnan(Y)
    nan_rows = np.any(np.vstack([nan_rows_X, nan_rows_Y]), axis = 0)
    no_nan_rows = np.invert(nan_rows)
    nan_rows_found = np.any(nan_rows)
    return X[no_nan_rows], Y[no_nan_rows], nan_rows_found

def makeRegressionMats(Tri, MIS, X, Y):
    '''
    (internal function) Built the estimation matrices and vector, indicated with B
    and Y in the lecture slides, repsectively.

    :param Tri: A Triangulation object as is specified in the class
            'Triangulation'
    :param MIS: A MultiIndexSet object as is specified in the class
            'MultiIndexSet'
    :param X: Independent data points, see modelFromData()
    :param Y: Dependent data points, see modelFromData()
    '''
    X_buckets, labels = Tri.classify(X)
    nr_simplices = len(labels)
    B_sub_matrices = [None]*nr_simplices
    for i in range(nr_simplices):
        X_b = t.toBarrys(X_buckets[i], Tri.getPointMat(i))
        B_sub_matrices[i] = makeBMatrix(X_b, MIS)
    B_matrix = sp.linalg.block_diag(*B_sub_matrices)
    Y_vec = Y[np.concatenate(labels)]
    return B_matrix, Y_vec    

def makeContinuityMat(Tri, MISmain, continuity):
    '''
    (internal function) Built the continuity matrix, indicated with H the lecture
    slides.

    :param Tri: A Triangulation object as is specified in the class
            'Triangulation'
    :param MIS: A MultiIndexSet object as is specified in the class
            'MultiIndexSet'
    :param continuity: (int >=0) Order of continuity ('r' in the slides)
    '''
    nr_simps, M = Tri.simplices.shape
    nr_borders = int(np.count_nonzero(Tri.neighbors >= 0)/2)
    pairs_dupli = np.zeros((nr_borders*2, 2), dtype = int)
    oop_vertices_dupli = np.zeros((nr_borders*2,), dtype = int)
    oop_vert_nr_dupli = np.zeros((nr_borders*2), dtype = int)

    insert = 0
    for simp_nr in range(nr_simps):
        for entry_nr in range(M):
            neighbor = Tri.neighbors[simp_nr, entry_nr]
            if neighbor >= 0:
                pairs_dupli[insert] = np.array([simp_nr, neighbor])             #double pairs of simplices (t2, t1)
                oop_vertices_dupli[insert] = Tri.simplices[simp_nr, entry_nr]
                oop_vert_nr_dupli[insert] = entry_nr
                insert +=1

    indices_inv = np.empty(nr_borders, dtype = int)
    indices_inv[:] = -1
    indices_norm = np.zeros(nr_borders, dtype = int)
    j=0
    for i in range(nr_borders*2):
        if not np.any(indices_inv == i):
            bool_list = np.all(pairs_dupli == pairs_dupli[i,::-1], axis=1)
            indices_inv[j] = np.flatnonzero(bool_list)[0]
            indices_norm[j] = i
            j += 1

    pairs = pairs_dupli[indices_norm]                   # pairs of simplex indices that form a border (t1, t2)
    oop_vertices_t2 = oop_vertices_dupli[indices_inv]   # oop (out of plane) vertex index (i in v_i) for 2st simplex in 'pairs'
    oop_vert_nr_t1 = oop_vert_nr_dupli[indices_norm]    # the position the vertex has in 1st simplex in 'pairs' (for 2d, 0, 1 or 2)
    oop_vert_nr_t2 = oop_vert_nr_dupli[indices_inv]

    order = MISmain.getOrder()
    dimension = M-1

    H_list = [0]*(continuity+1)
    for m in range(continuity+1):
        H_m_list = [0]*nr_borders
        for pair_nr in range(nr_borders):
            MI_t2 = mis.makeMISet(dimension, order, oop_vert_nr_t2[pair_nr], m)           # MI of left side of eq
            MI_t1_k = mis.makeMISet(dimension, order-m, oop_vert_nr_t1[pair_nr], 0)      # MI of right side of eq without gamma
            MI_gamma = mis.makeMISet(dimension, m)
            vertex_star_c = Tri.points[oop_vertices_t2[pair_nr]]
            vertex_star_b = t.toBarry(vertex_star_c, Tri.getPointMat(pairs[pair_nr, 0]))
            B_funcs = np.zeros(MI_gamma.length)
            for func_nr in range(MI_gamma.length):
                B_funcs[func_nr] = t.evalBasisFunc(vertex_star_b, MI_gamma.matrix[func_nr])
            B_matrix = np.zeros((MI_t2.length, nr_simps*MISmain.length))            
            for B_row in range(MI_t2.length):
                MI_t1 = MI_gamma.addMultiIndex(MI_t1_k.matrix[B_row])
                indices_t1_local = MISmain.getIndices(MI_t1.matrix)
                indices_t1 = indices_t1_local + pairs[pair_nr, 0]*MISmain.length
                index_t2_local = MISmain.getIndex(MI_t2.matrix[B_row])
                index_t2 = index_t2_local + pairs[pair_nr, 1]*MISmain.length
                B_matrix[B_row, index_t2] = -1
                B_matrix[B_row, indices_t1] = B_funcs
            H_m_list[pair_nr] = B_matrix
        H_m = np.vstack(H_m_list)
        H_list[m] = H_m
    H = np.vstack(H_list)
    return H

def makeBMatrix(vertices_b, MultiIndexSet):
    '''
    (internal function) Built the matrix named 'B' in the lecture slides for a
    single simplex. See step 7 on slide 62

    :param vertices_b: (1D numpy array) The vertex in barrycentric form for the
        basis functions to evaluate.
    :param MIS: A MultiIndexSet object as is specified in the class
            'MultiIndexSet'
    '''
    nr_vertices = vertices_b.shape[0]
    B_matrix = np.zeros((nr_vertices, MultiIndexSet.length))
    for k in range(MultiIndexSet.length):
        B_matrix[:,k] = t.evalBasisFunc(vertices_b, MultiIndexSet.matrix[k])
    return B_matrix


def _RMSE_model(X,Y_true,model):

    ''' 
    Calculate the RMSE of a spline model given the training data of independent variables, the model object, and true observations observations
    
    :param X: numpy array with shape [N x m] containing each of the (m) 
        independent training variables in their own column
    :param Y_true: Targets (Measurements), array with shape [N x 1]
    :param model: B-Spline model object
    
    :returns RMSE: Calculated root measn squared error of model residuals'''

    Y_model = model.eval(X)

    N = len(Y_true)
    RMSE = np.sqrt(np.sum((Y_true - Y_model.reshape(Y_true.shape))**2)/N)

    return RMSE

def _RMSE(Y_true, Y_model):

    ''' Calculate RMSE from true observations and model predictions
    
    :param Y_true: Targets (Measurements), array with shape [N x 1]
    :param Y_model: Model predictions, array with shape [N x 1]
    
    :returns RMSE: Root mean square error'''

    N = len(Y_true)

    RMSE = np.sqrt(np.sum((Y_true - Y_model.reshape(Y_true.shape))**2)/N)

    return RMSE


def _CoeffOfDetermination_R2_model(X, Y_true, model):
    '''(Internal) Function to calculate the coefficient of determination (R squared) from true observations, independent modelling variables, and 
       a spline model.

    :param X: numpy array with shape [N x m] containing each of the (m) independent training variables in their own column
    :param Y_true: Targets (Measurements), array with shape [N] (1D)
    :param model: B-Spline model object

    :return: Coefficient of Determination

    This function is taken from the 'droneidentification' prject authored by Jasper van Beers
    '''

    Y_model = model.eval(X)

    N = np.max(Y_true.shape)

    SSr = np.dot(Y_model, Y_true) - N*np.nanmean(Y_true)**2
    SSt = np.dot(Y_true, Y_true) - N*np.nanmean(Y_true)**2
    return SSr/SSt


def _CoeffOfDetermination_R2(Y_true, Y_model):
    '''(Internal) Function to calculate the coefficient of determination (R squared)

    :param Y_true: Targets (Measurements), array with shape [N] (1D)
    :param Y_model: Model predictions, array with shape [N] (1D) where N = number observations
    :return: Coefficient of Determination

    This function is taken from the 'droneidentification' prject authored by Jasper van Beers
    '''
    N = np.max(Y_true.shape)
    SSr = np.dot(Y_model, Y_true)- N*np.nanmean(Y_true)**2
    SSt = np.dot(Y_true, Y_true) - N*np.nanmean(Y_true)**2
    return SSr/SSt
