import pickle as p
import os
import numpy as np
#import simplexbsplines.ssmodel as sbs


def save_model(SSModel, path):
    """
    Save simplex spline model in a set of files.

    :param SSModel: SSModel object to be saved
    :param path: Model path and name. E.g. 'models/thrust_model'
    """
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)
    p.dump(SSModel, open(path + '.p', 'wb'))
    #np.savetxt(path + '/params.csv', SSModel.params, delimiter=",")
    #np.savetxt(path + '/misc.csv', np.array([SSModel.poly_order]), delimiter=",")


def open_model(path):
    """
    Load simplex spline model from a storage folder as created by 'save_model()'.

    :param path: Model path and name. E.g. 'models/thrust_model'
    """
    SSModel = p.load(open(path + '.p', 'rb'))
    #params = np.loadtxt(path+ '/params.csv', delimiter=',')
    #poly_order = int(np.loadtxt(path+ '/misc.csv', delimiter=','))
    return SSModel

