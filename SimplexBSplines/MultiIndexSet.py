import numpy as np
import itertools as it

class MultiIndexSet:
    def __init__(self, matrix):
        """
        Constructs MultiIndexSet object. This is simply a collection of several
        multi indexes. 

        :param matrix: (2D numpy array) Matrix containing the different multi
            indexes. The 0th axis contains the different multi indices, and the
            1st axis the different entries of the induvidual multi indices. E.g.
            matrix = np.array([[0,2],[0, 1],[1, 0],[2, 0]]) = k0, k1, k2, k3
        :returns: MultiIndexSet object
        """
        self.matrix = matrix
        self.length, self.M = matrix.shape

    def getOrder(self):
        """
        Get the polynomial order incase the multi index set represents a set of
        basis functions. 
        """
        return np.sum(self.matrix[0])

    def addMultiIndex(self, multi_index):
        """
        Add a multi index to all the multi indices in the multi index set and
        return the resulting multi index set. See lecture slide 85.

        :param multi_index: (1D numpy array) Vector containing the multi index to
            add to the set. 
        :returns: resulting MultiIndexSet object.
        """
        if self.M != multi_index.shape[0]:
            raise ValueError('Multi index sets do not have same dimensions')
        summ =  self.matrix + np.tile(multi_index,(self.length,1))
        return MultiIndexSet(summ)

    def getMultiIndex(self, index):
        return self.matrix[index]

    def getIndices(self, multi_index_mat):
        """
        Get the indices of the induvidual multi indices in 'multi_index_mat'
        in the multi index matrix of this object.

        :param multi_index_mat: (2D numpy array) Matrix containing the different
            multi indices.
        :returns: (1D numpy array) Resulting indices.
        """
        length, M = multi_index_mat.shape
        indices = np.zeros(length, dtype=int)
        for i in range(length):
            indices[i] = self.getIndex(multi_index_mat[i])
        return indices
            
    def getIndex(self, multi_index):
        """
        Same as 'MultiIndexSet.getIndices()', but for a single multi index.

        :param multi_index: (1D numpy array) A multi index.
        :returns: (float) Resulting index.
        """
        if multi_index.size != self.M:
            raise ValueError('multi_index length must be equal to M')
        bool_list = np.all(self.matrix == multi_index, axis=1)
        index = np.flatnonzero(bool_list)[0]
        return index

def makeMISet(dimension, order, pos = None, value = 0):
    """
    Make a MultiIndexSet object to represent a set of basis functions. The args
    'pos' and 'value' allow for setting on column of the multi index set matrix
    equal to 'value'. This is used for the multi indices of the continuity
    equations introduced on slide 69.

    :param dimension: (int) dimension of the resulting multi indices, so the
        amount of columns of the multi index set matrix.
    :param order: (int) Order of the resulting multi indices, which equal to the
        induvidual multi index entries
    :param pos: (int) Position of the inserted value. This is the row nr which
        set equal to 'value'
    :param value: (int) Inserted value.
    :returns: MultiIndexSet object.
    """
    if pos is None:
        width = dimension+1
        matrix = makeFullMIMatrix(width, order)
    elif(pos > dimension):
        raise ValueError("'position' should not exceed matrix dimension")
    elif(value > order):
        raise ValueError("'value' should not exceed matrix order")
    else:
        width = dimension
        order_new = order - value     
        matrix_full = makeFullMIMatrix(width, order_new)

        length = matrix_full.shape[0]
        sub_matrix_1 = matrix_full[:,:pos]
        sub_matrix_2 =np.ones(length, dtype=int).reshape(-1,1)*value
        sub_matrix_3 =matrix_full[:,pos:]
        matrix = np.hstack([sub_matrix_1, sub_matrix_2, sub_matrix_3])
    return MultiIndexSet(matrix)

def makeFullMIMatrix(width, order):
    """
    Same as makeMISet(), but without the options to fix values of a certain
    column
    """
    lst = np.arange(order+1)
    matrix_list = [multi_index for multi_index in it.product(lst, repeat = width)
                   if sum(multi_index) == order]
    matrix = np.array(matrix_list)
    return matrix

