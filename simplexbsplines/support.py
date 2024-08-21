##Some functions that are needed during the spline buidling process

import numpy as np
import scipy as sp
import scipy.special as spec

def evalBasisFunc(vertices_b, multi_index):
    """
    Evaluate a basis function based on a given multi index (1D numpy array)
    and a vertex in barrycentric form (1D numpy array)
    """
    wrong_dim = vertices_b.ndim == 1
    if wrong_dim:
        vertices_b = vertices_b.reshape(1,-1)
    poly_order = np.sum(multi_index)
    factor = np.math.factorial(poly_order)/np.prod(spec.factorial(multi_index))
    B = factor * np.prod(np.power(vertices_b, multi_index), axis = 1)
    if wrong_dim:
        return B.flatten()
    return B

def toBarry(vertex_c, point_mat):
    """
    Same as toBarrys(), but for a single vertex.

    :param vertices_c: (1D numpy array) vertex to convert in cartesian
        form. 
    :param point_mat: (2D numpy array) Corner points of the simplex. The
        0th axis contains the different vectors. 
    :returns vertex_b: (1D numpy array) Converted vertex.
    """
    vertices_c = vertex_c.reshape(1,-1)
    vertices_b = toBarrys(vertices_c, point_mat)
    vertex_b = vertices_b.flatten()
    return vertex_b

def toBarrys(vertices_c, point_mat):
    """
    Convert a cartesian set of vectors to barrycentric form, based on a set
    of corner points of a simplex. See lecture slide 34.

    :param vertices_c: (2D numpy array) vertices to convert in cartesian
        form. The 0th axis contains the different vectors.
    :param point_mat: (2D numpy array) Corner points of the simplex. The
        0th axis contains the different vectors. 
    :returns vertices_b: (2D numpy array) Converted vertices.
    """
          
    length, dimension = vertices_c.shape
    M, dimension2 = point_mat.shape
    if dimension != dimension2:
        raise ValueError("dimensions of 'vertices_c' and 'point_mat' do not match")
    if M-1 != dimension2:
        raise ValueError("'point_mat' sized improper to describe a simplex")
    AT = point_mat[1:] - np.tile(point_mat[0], (M-1, 1))
    try:
        ATinv = sp.linalg.inv(AT)
    except:
        raise Exception("No valid coordinate transformation to barrycentic. "
                        + "This is probably due to a degenerate simplex.")
    b_main =  (vertices_c - np.tile(point_mat[0], (length, 1))) @ ATinv
    b_zero = np.ones(length) - np.sum(b_main, axis = 1)
    vertices_b = np.hstack([b_zero.reshape(-1, 1), b_main])
    return vertices_b

def ECLQS(A, b, H):
    """
    Solve an equality constrained least squares problem, solve
    A @ params = Y subject to H @ params = 0 This is based on Lagrangian
    multipliers as shown on lecture slide 106. Apparently this method is
    quite inefficient. 

    :param A: (2D numpy array) Matrix corresponding to B in the lecture
        slides
    :param b: (1D numpy array) Vector denoted as Y in the lecture slides.
    
    :returns params_aug, C1: the estimated parameters (2D numpy array) and
        the covariance matrix of the estimated parameters.
    """
    n_A, m_A = A.shape
    n_H, m_H = H.shape
    M1 = np.block([[A.T@A, H.T],
                      [H, np.zeros((n_H, n_H))]])
    M2 = np.concatenate([A.T@b, np.zeros(n_H)])
    M1inv = np.linalg.pinv(M1)
    C1 = M1inv[:m_A,:m_A]       #b coefficient covariance matrix
    params_aug = M1inv @ M2
    return params_aug[:m_A], C1
