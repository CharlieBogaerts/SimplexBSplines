import numpy as np
import scipy as sp
import os
import pickle as p
import Triangulation as tri
import MultiIndexSet as mis
import Tools as t
import Modeling as mod


class SSmodel:
    def __init__(self, Tri, params, poly_order, vars):
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
        self.param_vars = vars

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
        nr_vertices, dim = vertices_c.shape
        if dim != self.Tri.points.shape[1]:
            raise ValueError('Mismatch between given vertices and triangulation dimension.')
        vertex_buckets, labels = self.Tri.classify(vertices_c)
        Y_simplex_list = [0]*self.Tri.simplices.shape[0]
        for i in range(self.Tri.simplices.shape[0]):
            vertex_bucket_b = t.toBarrys(vertex_buckets[i], self.Tri.getPointMat(i))
            B_matrix = mod.makeBMatrix(vertex_bucket_b, self.MIS)
            simplex_params = self.params[i*self.MIS.length:(i+1)*self.MIS.length]
            Y_simplex_list[i] = simplex_params@B_matrix.T
        labels_lumped = np.concatenate(labels)
        Y_unordered = np.concatenate(Y_simplex_list)
        Y = np.empty(nr_vertices)
        Y.fill(np.nan)
        Y[labels_lumped] = Y_unordered
        return Y

    def save(self, path, pickleWholeModel = False, trainingData = None):
        """
        Save current simplex spline model in a set of files.

        :param path: Model path and name. E.g. 'models/thrust_model'
        :param pickleWholeModelObject: Bool to specifiy whether to save the whole model (including triangulation, params, misc) into one file
        :param trainingData: Data used for model training as Pandas dataframe (or any other consistent data structure), only save if not None.
        """
        exists = os.path.exists(path)
        if not exists:
            os.makedirs(path)

        if pickleWholeModel:
            modelOut = {'Triangulate':self.Tri,
                        'params':self.params,
                        'variances':self.param_vars,
                        'misc':np.array([self.poly_order])}

            if trainingData is not None:
                modelOut['training data'] = trainingData

            p.dump(modelOut, open(path + '/model.pkl', 'wb'))
            
        p.dump(self.Tri , open(path + '/Triangulate.p', 'wb'))
        np.savetxt(path + '/params.csv', self.params, delimiter=",")
        np.savetxt(path + '/misc.csv', np.array([self.poly_order]), delimiter=",")
