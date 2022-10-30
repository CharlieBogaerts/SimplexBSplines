import numpy as np
import scipy.spatial as spat

class Triangulation:
    def __init__(self, points):
        """
        Constructs Triangulation object. This is pretty much an extension
        to the spat.Delaunay(points). 

        :param points: (2D numpy array) Cartesian vertors which should be
            used in the triangulation. The 0th axis contains the different
            vectors. The vector dimension should be higher than 2
        :returns: Triangulation model object
        """
        self.points = points
        self.Tri = spat.Delaunay(points)
        nr_simps, M = self.Tri.simplices.shape
        reorder = np.argsort(self.Tri.simplices, axis = 1)
        self.simplices = np.zeros((nr_simps, M), dtype = int)
        self.neighbors = np.zeros((nr_simps, M), dtype = int)
        for i in range(nr_simps):
            self.simplices[i] = self.Tri.simplices[i,reorder[i]]
            self.neighbors[i] = self.Tri.neighbors[i,reorder[i]]

    def classify(self, vertices_c):
        """
        Find the simplices of which vertices are encompassed by. The vertices
        are devided into buckets where each bucket corresponds to a simplex
        in Triangulation.simplices.

        :param vertices_c: (2D numpy array) Cartesian vertors which should be
            divided based on the simplex they fall into.
        :returns: (List of 2D numpy arrays) Each list entry corresponds to a
            simplex. 
        """
        nr_simplices = self.Tri.simplices.shape[0]
        locations = self.Tri.find_simplex(vertices_c)
        labels = [0]*nr_simplices
        vertex_buckets = [0]*nr_simplices
        for i in range(nr_simplices):
            labels_simplex = np.nonzero(locations == i)[0]
            labels[i] = labels_simplex
            vertex_buckets[i] = vertices_c[labels_simplex]
        return vertex_buckets, labels

    def getPointMat(self, simplex_index):
        """
        Returns a matrix contaning the diffent points that make up a simplex.

        :param simplex_index: (int) Index of the simplex which points should
            be returned
        :returns: (2D numpy array) matrix containing the simplex points.
        """
        simplex = self.simplices[simplex_index]
        point_matrix = self.points[simplex]
        return point_matrix
