from os import listdir
from numpy import array,genfromtxt,exp,sqrt,max,savetxt

def get_rate_matrix(shape,predictors,coefficient,indicators=None):
    L = []
    directory=listdir(predictors)
    already_analysed=0
    if indicators:
        for file in range(len(directory)):
            if directory[file][0:8]=='PREDICT_':
                table = genfromtxt(predictors+'/'+directory[file], delimiter=',')
                L.append(coefficient[already_analysed]*indicators[already_analysed]*table)
                already_analysed += 1
        return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))
    else:
        for file in range(len(directory)):
            if directory[file][0:8]=='PREDICT_':
                table = genfromtxt(predictors+'/'+directory[file], delimiter=',')
                L.append(coefficient[already_analysed]*table)
                already_analysed += 1
        return exp(array([[sum(X[i,j] for X in L) for j in range(shape)] for i in range(shape)]))

def create_predictors(directory):
    input_file = listdir(directory)
    for file in range(len(input_file)):
        table = genfromtxt(directory + '/' + input_file[file], delimiter=',')
        if input_file[file] == 'location.csv':
            with open(directory + '/' + input_file[file], 'r') as f:
                order_of_localities = []
                for line in f:
                    order_of_localities.append(line.split(',')[0])
                order_of_localities = tuple(order_of_localities)
            table=array(table)[:,1:]
            M = array([[1/sqrt((table[i,0]-table[j,0])**2+(table[i,1]-table[j,1])**2) if i!=j else 1 for j in range (table.shape[0])] for i in range (table.shape[0])])
            M = M/M.max()
            savetxt(directory + "/corrected_location.csv", M, delimiter=",")
        else:
            savetxt(directory + '/' + input_file[file], table/table.max(), delimiter=",")
    return order_of_localities
#ou: d'abord normalisation, puis ajout de 1 sur la diagonale..

def filenumber(directory):
    return len([i for i in listdir(directory) if i[0:8]!='PREDICT_'])

##TODO 05/02:
##      -autoriser l'input de coefficients

def create_predictors2(directory,remove=False):
    input_file = listdir(directory)
    names=input_file.copy()
    for file in range(len(input_file)):
        if file == 0:
            with open(directory + '/' + input_file[file], 'r') as f:
                order_of_localities = []
                for line in f:
                    order_of_localities.append(line.split(',')[0])
                order_of_localities = tuple(order_of_localities[1:])
        table = genfromtxt(directory + '/' + input_file[file], skip_header=1,delimiter=',')[:, 1:]
        if input_file[file] == 'raw_location.csv':
            table = array(table)
            table = array([[1/sqrt((table[i,0]-table[j,0])**2+(table[i,1]-table[j,1])**2) if i!=j else 1 for j in range (table.shape[0])] for i in range (table.shape[0])])
        elif input_file[file] == 'location.csv':
            table=array([[1/table[i,j] if i!=j else 1 for j in range (table.shape[0])] for i in range (table.shape[0])])
        elif input_file[file][0:8]=='PREDICT_':
            break
        savetxt(directory + '/PREDICT_' + input_file[file], table/table.max(), delimiter=",")
    return order_of_localities,names


def end(directory,order_of_localities):
    input_file = listdir(directory)
    print()