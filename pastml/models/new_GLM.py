from os import listdir
from numpy import array, genfromtxt, exp, savetxt, zeros, eye, dot


def get_rate_matrix(shape,predictors_dict,coefficient,indicators=None, predictors_dir=None):
    """
    Calculates the rate matrix from coefficients and input matrices

    :param shape: shape of predictors' matrices
    :type shape: int or tuple
    :param predictors_dict: dictionary of loaded predictors' matrices
    :type predictors_dict: dict of numpy arrays
    :param coefficient: coefficient for each predictor, between boundaries as defined in get_bounds
    :type coefficient: list
    :param indicators: if any, boolean values to take / not to take into account specific matrices
    :type indicators: list
    :param predictors_dir: absolute directory containing predictors' matrices
    :type predictors_dir: str
    :return: lambda matrix (rate matrix)
    :rtype: numpy array
    """

    L = []
    if predictors_dict is None:
        directory=listdir(predictors_dir)
    already_analysed=0
    if type(shape) == tuple:
        if shape[0]!=shape[1]:
            raise ValueError("Predictors' matrices should be the square matrix.")
        shape=shape[0]
    if indicators:
        if predictors_dict:
            for predictor in predictors_dict:
                L.append(coefficient[already_analysed] * indicators[already_analysed] * predictors_dict[predictor])
                already_analysed += 1
            return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))
        for file in range(len(directory)):
            if directory[file][0:8]=='PREDICT_':
                table = genfromtxt(predictors_dir+'/'+directory[file], delimiter=',')
                L.append(coefficient[already_analysed]*indicators[already_analysed]*table)
                already_analysed += 1
        return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))
    else:
        if predictors_dict:
            for predictor in predictors_dict:
                L.append(coefficient[already_analysed] * predictors_dict[predictor])
                already_analysed += 1
            return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))
        for file in range(len(directory)):
            if directory[file][0:8]=='PREDICT_':
                table = genfromtxt(predictors_dir+'/'+directory[file], delimiter=',')
                L.append(coefficient[already_analysed]*table)
                already_analysed += 1
        return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))


def filenumber(directory):
    return len([i for i in listdir(directory) if i[0:8]!='PREDICT_'])

def create_predictors(directory,remove=False):
    """
    Creates normalised predictors matrices, uploaded in directory as 'PREDICTOR_predictor-name.txt'

    :param directory: absolute directory containing each predictor matrix
    :type directory: str

    :return order_of_localities, names, D: order of locations used in predictor matrices, list of predictor names,
            dictionary of all predictor matrices
    :rtype: tuple of a tuple, a list and a dictionary
    """

    input_file = listdir(directory)
    names=input_file.copy()
    D = {}
    #checks wether matrices have the same 'order of locations' and corrects them if needed
    for file in range(len(input_file)):
        if file == 0:
            with open(directory + '/' + input_file[file], 'r') as f:
                order_of_localities = tuple(f.readline().split(',')[1:])
            size = len(order_of_localities)
            col_transf,row_transf = eye(size),eye(size)
            table = genfromtxt(directory + '/' + input_file[file], skip_header=1, delimiter=',')[:,1:]
        else:
            col_transf, row_transf = eye(size), eye(size)
            with open(directory + '/' + input_file[file], 'r') as f:
                #columns' order:
                order_of_localities_predict = tuple(f.readline().split(',')[1:])
                if order_of_localities_predict != order_of_localities:
                    col_transf=zeros((size,size))
                    for i in range(size):
                        try:
                            pos=order_of_localities.index(order_of_localities_predict[i])
                            col_transf[pos,i]=1
                        except ValueError:
                            raise ValueError("Predictors' matrices should have the same set of locations.")
                #lines' order:
                order_of_localities_predict = []
                for i in range(size):
                    order_of_localities_predict.append(f.readline().split(',')[0])
                order_of_localities_predict[-1]=order_of_localities_predict[-1]+'\n'
                if tuple(order_of_localities_predict) != order_of_localities:
                    row_transf=zeros((size,size))
                    for i in range(size):
                        try:
                            pos=order_of_localities.index(order_of_localities_predict[i])
                            row_transf[i,pos]=1
                        except ValueError:
                            raise ValueError("Predictors' matrices should have the same set of locations.")
            table = genfromtxt(directory + '/' + input_file[file], skip_header=1, delimiter=',')[:,1:]
        T = dot(dot(row_transf,table),col_transf)
        D[input_file[file]]=T/T.max()
        savetxt(directory + '/PREDICT_' + input_file[file], T/T.max(), delimiter=",")
    return order_of_localities,names,D