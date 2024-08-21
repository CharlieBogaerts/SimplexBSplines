##Tools and functions that can help write estimation code. These are
##suplementary to the spine moddeling tools.

def random_partition_data(Data, TrainingRatio, *argv):
    '''Function to partition data through random sampling without replacement

    :param Data: Data to be partitioned
    :param TrainingRatio: Ratio of data, as float ]0, 1[, to be allocated to the training data subset
    :param argv: Additional positional arguments, unused.
    
    :return: Tuple of lists as ([Partitioned training Data, Indices training], [Partitioned testing data, Indices testing])

    This function is taken from the 'droneidentification' prject authored by Jasper van Beers
    '''

    np.random.seed(111)

    N = Data.shape[0]
    if TrainingRatio >= 1:
        print('[ WARNING ] Inputted training ratio is >= 1 when it should be < 1. Defaulting to 0.7')
        TrainingRatio = 0.7
    N_Training = int(TrainingRatio*N)

    indices = np.arange(N)
    indices_bool = np.ones((N, 1))
    indices_training = np.sort(np.random.choice(indices, N_Training, replace = False))
    indices_bool[indices_training] = 0
    indices_test = np.where(indices_bool)[0]

    try:
        TrainingData = Data[:, indices_training]
        TestData = Data[:, indices_test]
    except (IndexError, MemoryError) as e:
        TrainingData  = Data[indices_training, :]
        TestData = Data[indices_test, :]

    return [TrainingData, indices_training], [TestData, indices_test]

def rmse_model(X,Y_true,model):

    ''' 
    Calculate the RMSE of a spline model given the training data of independent variables, the model object, and true observations observations
    
    :param X: numpy array with shape [N x m] containing each of the (m) 
        independent training variables in their own column
    :param Y_true: Targets (Measurements), array with shape [N x 1]
    :param model: B-Spline model object
    
    :returns RMSE: Calculated root measn squared error of model residuals'''

    Y_model = model.eval(X)

    N = len(Y_true)
    RMSE = np.sqrt(np.nansum((Y_true - Y_model.reshape(Y_true.shape))**2)/N)

    return RMSE

def rmse(Y_true, Y_model):

    ''' Calculate RMSE from true observations and model predictions
    
    :param Y_true: Targets (Measurements), array with shape [N x 1]
    :param Y_model: Model predictions, array with shape [N x 1]
    
    :returns RMSE: Root mean square error'''

    N = len(Y_true)

    RMSE = np.sqrt(np.nansum((Y_true - Y_model.reshape(Y_true.shape))**2)/N)

    return RMSE
