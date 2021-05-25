from os import listdir
import numpy as np
import operator
import pickle

# Convert matrix to vector
def Img2Vector(filename):
    #print(filename)
    # Return matrix
    ret_val = np.zeros((1, 1024))
    # Read file's content
    file = open(filename)
    content = file.readlines()
    for i in range(32):
        line = content[i]
        for j in range(32): 
            ret_val[0, 32 * i + j] = int(line[j])
            
    return ret_val

# Normalize our data set
def AutoNorm(data_set):
    #print(type(data_set))
    # Get all feature's limits
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    val_range = max_val - min_val
    # Normalize our data set
    m = data_set.shape[0]
    normal_set = data_set - np.tile(min_val, (m, 1))
    normal_set = normal_set / np.tile(val_range, (m, 1))
    return normal_set, min_val, val_range

# Build our classifier
def ClassifyTrain(in_vector, data_set, labels, k):
    m = data_set.shape[0]  #1934..for each dataset
    # Calculate euclidian distance and sort
    diff_mat = data_set - np.tile(in_vector, (m, 1))
    square_diff_mat = diff_mat ** 2
    # Axis is one
    sum_square_diff_mat = np.sum(square_diff_mat, axis=1)
    indexes = np.argsort(sum_square_diff_mat)
    # Select K distance
    class_count = {}
    for i in range(k):
        index = indexes[i]
        label = labels[index]
        
        class_count[label] = class_count.get(label, 0) + 1#increment class by 1
    result = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #print(label)
    #print(labels)
    # Get the most similar label
    return result[0][0]

# Get our data set
def GetDataSetByDir(dirname):
    training_set_dir = listdir(dirname)
    m = len(training_set_dir)
    # Prepare data set
    labels = []
    data_set = np.zeros((m, 1024))
    # Get our data set and targets
    for i in range(m):
        filename = training_set_dir[i]
        label = int(filename.split('_')[0])
        labels.append(label)
        #print('aaaaa')
        #print(dirname + filename)
        data_set[i, :] = Img2Vector(dirname + filename)
        #print(data_set[i, :])
    return data_set, labels


# Create Classify by using kNN
def ClassifyTest():
    # Select our directory
    test_dir_name = "C:/Users/hp/Desktop/KNN/TestDigits/"
    train_dir_name = "C:/Users/hp/Desktop/KNN/TrainingDigits/"
    data_set, labels = GetDataSetByDir(train_dir_name)
    test_set, test_labels = GetDataSetByDir(test_dir_name)
    #print(test_set)
    # Normalize
    # data_set, min_val, val_range = AutoNorm(data_set)
    # Test and calculate error rate
    error_count = 0
    test_num = len(listdir(test_dir_name))
    for i in range(test_num):
        result = ClassifyTrain(test_set[i, :], data_set, labels, 3)
       # print("The real answer is %d, and the classifier came back with %d" %
       #       (test_labels[i], result))
        if test_labels[i] != result:
            error_count += 1
    print("The error rate is %f" % (error_count / test_num))
    
    acc=(error_count / test_num)
    print("The accuracy is %f" %(1-acc) )


    #pickle.dump(1-acc,open('iri.pkl','wb'))



# Create Classify by using kNN
def ClassifyTest1():
    # Select our directory
    #dirname = "C:/Users/hp/Desktop/KNN/TestDigits/"
    #train_dir_name = "C:/Users/hp/Desktop/KNN/TrainingDigits/"
    #data_set, labels = GetDataSetByDir(dirname)
    #filename1 = pickle.load(open('filnam.pkl','rb'))
    #label1=int(filename1.split('_')[0])
    #data_setx[i, :] = Img2Vector(dirname + filename)
    #fname2=dirname + filename
    #retv=np.zeros((1,1024))
    #test_set1, test_label = data_set1,label1
    #content =  pickle.load(open('res.pkl','rb')).readlines()
    #for i in range(32):
     #   line = content[i]
      #  for j in range(32): 
       #     ret_val[0, 32 * i + j] = int(line[j])

    #filename2 = pickle.load(open('filnam.pkl','rb'))
    #data_set1[i, :] = Img2Vector(filename2)
    #my_file, test_label = data_set1,label1

    #test_dir_name = "C:/Users/hp/Desktop/KNN/TestDigits/"
    #test_set, test_labels = GetDataSetByDir(test_dir_name)




    fil=pickle.load(open('filnam.pkl','rb'))
    label=int(filename.split('_')[0])
    retval=np.zeros((1,1024))
    fil1=pickle.load(open('res.pkl','rb'))
    content=fil1.readlines()
    for i in range(32):
        line = content[i]
        for j in range(32): 
            retval[0, 32 * i + j] = int(line[j])
    
    

    result = ClassifyTrain(retval, fil1, label, 3)
    lst=[]
    lst[0]=test_label
    lst[1]=result
    pickle.dump(lst,open('iri.pkl','wb'))
      

    
ClassifyTest()
ClassifyTest1()



