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

def location_matrix(directory):
    input_file = listdir(directory)
    names = input_file.copy()
    for file in range(len(input_file)):
        if input_file[file] == 'raw_location.csv':
            table = array(table)
            table = array([[1 / sqrt((table[i, 0] - table[j, 0]) ** 2 + (table[i, 1] - table[j, 1]) ** 2) if i != j else 1 for j
                            in range(table.shape[0])] for i in range(table.shape[0])])
        elif input_file[file] == 'location.csv':
            table = array([[1 / table[i, j] if i != j else 1 for j in range(table.shape[0])] for i in range(table.shape[0])])
        elif input_file[file][0:8]=='PREDICT_':
                break