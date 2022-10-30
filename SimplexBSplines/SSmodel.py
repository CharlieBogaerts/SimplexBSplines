import numpy as np
import scipy as sp
import os
import pickle as p
from SimplexBSplines import Triangulation as tri
from SimplexBSplines import MultiIndexSet as mis
from SimplexBSplines import Tools as t


class SSmodel:
    def __init__(self, Tri, params, poly_order):
        """
        Constructs SSmodel object. The model is defined by a triangulation object,
        a set of model parameters (b-coefficients) and the order of the simplex
        polynomials. 

        :param Tri: A Triangulation object as is specified in the class
            'Triangulation'
        :param params: (1D numpy array) Array including the parameters of the model.
            The first parameters belong to the 0th simplex, then the 1st simplex,
            etc. The parameters for each simplex are ordered based on increasing
            multi index of the basis functions
        :param poly_order: The order of the polynomials. E.g. a basis function with
            multi index 0,2,1 would have an order of 3.
        :returns: Simplex spline model object
        """
        self.nr_points, dimension = Tri.points.shape
        self.MIS = mis.makeMISet(dimension, poly_order)
        nr_basis_funcs, M = self.MIS.matrix.shape
        if dimension != M-1:
            raise ValueError('points and poly_order arguments do not match.')
        if nr_basis_funcs != int(params.size/Tri.simplices.shape[0]):
            raise ValueError('params and poly_order arguments do not match.')
        self.Tri = Tri
        self.params = params
        self.poly_order = poly_order

    def evalSingle(vertex_c):
        """
        Same function as 'SSmodel.eval', but accepts a single vertex.

        :param vertices_c: (1D numpy array) The dependent variable vertex.
        :returns: (float) calculated dependent variable value.
        """
        if vertex_c.ndim != 1:
            raise ValueError("'vertex_c' should be a 1D numpy array")
        vertex_c = vertex.reshape(1,-1)
        Y = self.eval(self, vertex_c)[0]
        return Y

    def eval(self, vertices_c):
        """
        Evaluates the simplex spline at independent variables
        vertices_c (cartesian form). For vertices outside the spline domain
        np.NaN is returned

        :param vertices_c: (2D numpy array) The dependent variable vertices. The
            0th axis should contain the different vertices, and the 1st axis the
            different entries of each vertex.
        :returns: (1D numpy array) calculated dependent variable values.
        """
        nr_vertices, N = vertices_c.shape
        vertex_buckets, labels = self.Tri.classify(vertices_c)
        Y_simplex_list = [0]*self.Tri.simplices.shape[0]
        for i in range(self.Tri.simplices.shape[0]):
            vertex_bucket_b = t.toBarrys(vertex_buckets[i], self.Tri.getPointMat(i))
            B_matrix = makeBMatrix(vertex_bucket_b, self.MIS)
            simplex_params = self.params[i*self.MIS.length:(i+1)*self.MIS.length]
            Y_simplex_list[i] = simplex_params@B_matrix.T
        labels_lumped = np.concatenate(labels)
        Y_unordered = np.concatenate(Y_simplex_list)
        Y = np.empty(nr_vertices)
        Y.fill(np.nan)
        Y[labels_lumped] = Y_unordered
        return Y

    def save(self, path):
        """
        Save current simplex spline model in a set of files.

        :param path: Model path and name. E.g. 'models/thrust_model'
        """
        exists = os.path.exists(path)
        if not exists:
            os.makedirs(path)
        p.dump(self.Tri , open(path + '/Triangulate.p', 'wb'))
        np.savetxt(path + '/params.csv', self.params, delimiter=",")
        np.savetxt(path + '/misc.csv', np.array([self.poly_order]), delimiter=",")

def modelFromCsv(path):
    """
    Load simplex spline model from a set of csv files. These files can be created
    unsing the SSmodel.save() function.

    :param path: Model path and name. E.g. 'models/thrust_model'
    """
    Tri = p.load(open(path+ '/Triangulate.p', 'rb'))
    params = np.loadtxt(path+ '/params.csv', delimiter=',')
    poly_order = int(np.loadtxt(path+ '/misc.csv', delimiter=','))
    return SSmodel(Tri, params, poly_order)

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
    params = t.constrainedLQS(B_matrix, Y_vec, H)
    return SSmodel(Tri, params, poly_order)

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

    diff = pairs_dupli - np.sort(pairs_dupli, axis = 1)
    include = (diff==0)[:,0]
    pairs = pairs_dupli[include]                                # pairs of simplex indices that form a border (t1, t2)
    oop_vertices_t2 = oop_vertices_dupli[np.invert(include)]    # oop (out of plane) vertex index (i in v_i) for 2st simplex in 'pairs'
    oop_vert_nr_t1 = oop_vert_nr_dupli[include]                 # the position the vertex has in 1st simplex in 'pairs' (for 2d, 0, 1 or 2)
    oop_vert_nr_t2 = oop_vert_nr_dupli[np.invert(include)]      # the position the vertex has in 2st simplex in 'pairs' (for 2d, 0, 1 or 2)

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
