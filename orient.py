##DETAILS INCLUDED WITH THE PDF
#BEST   |  Epoch:  10  |  Hidden Nodes:  125  |  Batch Size:  25  |  ETA:  0.25  |   Lambda: 0.0005  | Accuracy: ~70-72%
#N_NET  |  Epoch:  5   |  Hidden_Nodes:  175  |  Batch Size:  30  |  ETA:  0.2   |   Accuracy: 71.94%
#knn    |  K:  29      |  Accuracy: 70.94%
##----------------------------------------------------------------------------------------------------------------------------##

from __future__ import division
from collections import Counter
from numpy import *
import numpy as np
import random
import timeit
import time
import sys
import os

start = timeit.default_timer()

#Class - NeuralNetwork

class NeuralNetwork():

    def __init__(self, h_nodes, nnet_mode):
        #Initialize Layers
        self.nnet_mode = nnet_mode
        self.layer_i = 192
        self.layer_o = 4
        self.layer_h = h_nodes
        if nnet_mode == 0:
            #Weight_Matrices #w1,w2 - Random Initialization
            self.theta_1 = np.random.randn(self.layer_i, self.layer_h)
            self.theta_2 = np.random.randn(self.layer_h, self.layer_o)
            #Bias_Matrices #b1,b2
            self.beta_1 = np.random.randn(1, self.layer_h)
            self.beta_2 = np.random.randn(1, self.layer_o)
        if nnet_mode == 1:
            #Weight_Matrices #w1,w2 - Gaussian Initialization
            self.theta_1 = np.random.randn(self.layer_i, self.layer_h)/np.sqrt(self.layer_h)
            self.theta_2 = np.random.randn(self.layer_h, self.layer_o)/np.sqrt(self.layer_o)
            #Bias_Matrices #b1,b2
            self.beta_1 = np.random.randn(1, self.layer_h)/np.sqrt(self.layer_h)
            self.beta_2 = np.random.randn(1, self.layer_o)/np.sqrt(self.layer_o)


    def sigmoid(self, z):
        if self.nnet_mode == 0: return np.tanh(z)
        if self.nnet_mode == 1: return 1/(1+np.exp(-z))

    def delta_sigmoid(self, z):
        if self.nnet_mode == 0: return 1/((np.cosh(z))*(np.cosh(z)))
        if self.nnet_mode == 1: return np.exp(-z)/(1+np.exp(-z))**2


    def back_propagation(self, matrix_i):

        #FWD_PROPAGATION
        inpt = [float(matrix_i[0][i]) for i in range(len(matrix_i[0]))]
        m_inpt = np.array(inpt).reshape(1,192)
        m_otpt = int(matrix_i[1])
        if m_otpt == 0.0:
            m = [1.0,0.0,0.0,0.0]
        elif m_otpt == 90.0:
            m = [0.0,1.0,0.0,0.0]
        elif m_otpt == 180.0:
            m = [0.0,0.0,1.0,0.0]
        elif m_otpt == 270.0:
            m = [0.0,0.0,0.0,1.0]
        m_otpt = np.array(m).reshape(1,4)
        matrix_1 = m_inpt.dot(self.theta_1) + self.beta_1
        sigmoid_1 = self.sigmoid(matrix_1)
        matrix_2 = sigmoid_1.dot(self.theta_2) + self.beta_2
        sigmoid_2 = self.sigmoid(matrix_2)
        #BCK_PROPAGATION
        #f'(z2) & f'(z1)
        del_z2 = self.delta_sigmoid(sigmoid_2)
        del_z1 = self.delta_sigmoid(sigmoid_1)
        #(y'-y)
        delta = sigmoid_2 - m_otpt
        #Delta - Error Functions
        ##Root Mean Square Cost Function - Mode 0
        if self.nnet_mode == 0:
            delta_2 = np.multiply(delta, del_z2)
            delta_1 = np.multiply(delta_2.dot(self.theta_2.transpose()), del_z1 )
        ##Cross Entropy Cost Function - Mode 1
        elif self.nnet_mode == 1:
            delta_2 = sigmoid_2 - m_otpt
            delta_1 = (delta_2.dot(self.theta_2.transpose()))*del_z1
        #Weight Error Functions - DC/DW2, DC/DW1
        delta_w2 = sigmoid_1.transpose().dot(delta_2)
        delta_w1 = m_inpt.transpose().dot(delta_1)
        delta_w = []
        delta_w = [delta_w1, delta_w2]
        #Bias Error Functions - DC/DB2, DC/DB1
        delta_b2 = delta_2
        delta_b1 = delta_1
        delta_b = []
        delta_b = [delta_b1, delta_b2]
        return(delta_w, delta_b)


    def train_nerualnet(self, matrix_i, epoch, batch_size, eta):
        print '\n2. TRAINING'
        global parameter
        global l_method
        lmbda = 0.0005
        # start = timeit.default_timer()

        for iterations in xrange(epoch):

            print '\tTraining Progress: ', (iterations+1)*100/epoch,'%'
            ##Split the input matrix in batches of size 10 images. Here, batch_i is a list comprising
            ##lists of 10 images. For 40k i/p images, batch_i will contain 4k lists, each comprising 10 images

            np.random.shuffle(matrix_i)
            batch_i = [matrix_i[i: i + batch_size] for i in xrange(0, len(matrix_i), batch_size)]
            b_count = 0
            for batch in batch_i:
                b_count += 1
                # print "\n\tBatch: ", b_count
                del_w1 = np.zeros(self.theta_1.shape)
                del_w2 = np.zeros(self.theta_2.shape)
                del_b1 = np.zeros(self.beta_1.shape)
                del_b2 = np.zeros(self.beta_2.shape)
                i_count = 0
                for image in batch:
                    i_count += 1
                    # print "\n\tImage: ", i_count
                    delta_w, delta_b = self.back_propagation(image)
                    del_w1 = (del_w1 + delta_w[0])
                    del_w2 = (del_w2 + delta_w[1])
                    del_b1 = (del_b1 + delta_b[0])
                    del_b2 = (del_b2 + delta_b[1])
                ##Update weights without Regulariztion - Mode 0
                if self.nnet_mode == 0:
                    self.theta_1 = self.theta_1 - (del_w1*(eta/batch_size))
                    self.theta_2 = self.theta_2 - (del_w2*(eta/batch_size))
                ##Update weights with Regularization - Mode 1
                elif self.nnet_mode == 1:
                    self.theta_1 = self.theta_1 - (del_w1 + (self.theta_1*lmbda))*eta/batch_size
                    self.theta_2 = self.theta_2 - (del_w2 + (self.theta_2*lmbda))*eta/batch_size
                self.beta_1 = self.beta_1 - ((del_b1/batch_size)*eta)
                self.beta_2 = self.beta_2 - ((del_b2/batch_size)*eta)
        # stop = timeit.default_timer()
        # print "\tTraining | Time Required: ", stop - start


    def test(self, matrix_i):
        print '\n4. TESTING'
        # start = timeit.default_timer()
        count = 0
        acc_sum = 0
        results = []
        # print len(matrix_i)
        for tst_image in matrix_i:
            count += 1
            i_result = []
            # print count
            m_inpt = np.array([float(tst_image[0][i]) for i in range(len(tst_image[0]))]).reshape(1,192)
            # print m_inpt.shape
            m_otpt = float(tst_image[1])
            matrix_1 = m_inpt.dot(self.theta_1) + self.beta_1
            sigmoid_1 = self.sigmoid(matrix_1)
            matrix_2 = sigmoid_1.dot(self.theta_2) + self.beta_2
            sigmoid_2 = self.sigmoid(matrix_2)
            sigmoid_2 = sigmoid_2.tolist()[0]
            pred = max(sigmoid_2)
            indx_prd = sigmoid_2.index(pred)
            for i in xrange(len(sigmoid_2)):
                if i == indx_prd:
                    sigmoid_2[i] = 1
                else: sigmoid_2[i] = 0
            p_orientation = sigmoid_2.index(1)
            if p_orientation == 0: p_orientation = 0
            elif p_orientation == 1: p_orientation = 90
            elif p_orientation == 2: p_orientation = 180
            elif p_orientation == 3: p_orientation = 270
            sigmoid_2 = np.array(sigmoid_2).reshape(1,4)
            if p_orientation == m_otpt: match = 1
            else: match = 0
            i_result = [p_orientation, m_otpt, match]
            results.append(i_result)
            acc_sum = acc_sum + match
        accuracy = acc_sum/count
        # stop = timeit.default_timer()
        # print "\tTesting | Time Required: ", stop - start
        return([results, accuracy])


#PROCESS_I/P_FILE
def preprocess_data(t_mode):
    i_matrix = []
    o_matrix = []
    l_matrix = []
    global train_file
    global test_file

    if t_mode == 0:
        print '\n1. PRE-PROCESSING - TRAINING DATA'
        inpt_f = open(train_file, 'r')
        # inpt_f = train_file
        # start = timeit.default_timer()
    else:
        print '\n3. PRE-PROCESSING - TEST DATA'
        inpt_f = open(test_file, 'r')
        # inpt_f = test_file
        # start = timeit.default_timer()
    # count = 0
    # with open('train-data.txt', 'r') as inpt_f:
    for lines in inpt_f:
            lines = lines.split()
            # count += 1
            l_matrix.append(lines[0])
            o_matrix.append(lines[1])
            ##Normalize
            i_lst = lines[2:]
            i_lst = [float(pxl)/2550 for pxl in i_lst]
            i_matrix.append(i_lst)
            ##No Normalization
            # i_matrix.append(lines[2:])
            # print i_lst

            # print count

    matrix_i = zip(i_matrix,o_matrix)
    # stop = timeit.default_timer()
    if t_mode == 0:
        # print "\tPreprocessing | Time Required : ", stop - start
        return(matrix_i)
    elif t_mode == 1:
        # print "\tPreprocessing | Time Required : ", stop - start
        return(matrix_i, l_matrix)

def confusion_m(prediction, groundtruth):
    global start
    confusion_m = np.zeros((4, 4))

    for index in range(len(prediction)):
        if prediction[index] == "0": j = 0
        elif prediction[index] == "90": j = 1
        elif prediction[index] == "180": j = 2
        elif prediction[index] == "270": j = 3
        if groundtruth[index] == "0": i = 0
        elif groundtruth[index] == "90": i = 1
        elif groundtruth[index] == "180": i = 2
        elif groundtruth[index] == "270": i = 3

        confusion_m[i][j] +=1

    print "\n\nCONFUSION MATRIX: "
    print "---|---0----|---90----|---180---|--270---|"
    print "---|--------|---------|---------|--------|"
    print "  0|","%5s" % confusion_m[0][0]," | ","%5s" % confusion_m[0][1]," | ","%5s" % confusion_m[0][2]," | ","%5s" % confusion_m[0][3],"|"
    print " 90|","%5s" % confusion_m[1][0]," | ","%5s" % confusion_m[1][1]," | ","%5s" % confusion_m[1][2]," | ","%5s" % confusion_m[1][3],"|"
    print "180|","%5s" % confusion_m[2][0]," | ","%5s" % confusion_m[2][1]," | ","%5s" % confusion_m[2][2]," | ","%5s" % confusion_m[2][3],"|"
    print "270|","%5s" % confusion_m[3][0]," | ","%5s" % confusion_m[3][1]," | ","%5s" % confusion_m[3][2]," | ","%5s" % confusion_m[3][3],"|"
    print "-----------------------------------------"
    print '\n\nTIME REQUIRED: ', timeit.default_timer() - start


#NEURALNETWORK
def nnet(nnet_mode):

    global train_file
    global test_file
    global parameter
    h_nodes = parameter
    accuracy = []

    if nnet_mode == 0:
        epoch = 5
        batch_size = 30
        eta = 0.2

    elif nnet_mode == 1:
        epoch = 10
        batch_size = 25
        eta = 0.25

    ##INITIALIZE
    NeuralNet = NeuralNetwork(h_nodes, nnet_mode)
    ##TRAIN
    matrix_i = preprocess_data(0)
    NeuralNet.train_nerualnet(matrix_i, epoch, batch_size, eta)
    ##TEST
    lst = preprocess_data(1)
    matrix_i = lst[0]
    img_id = lst[1]
    results = NeuralNet.test(matrix_i)
    predictions = results[0]
    accuracy = results[1]
    # template = "{0:<10} {1:<25} {2:<20} {3:<20} {4:<20}" # column widths: 5, 15, 10, 10, 10
    # print template.format('Number','Image_ID','Prediction','G_Truth','Match')
    # count = 0
    if nnet_mode == 0:
        f = open("nnet_output.txt", "w")
    elif nnet_mode == 1:
        f = open("best_output.txt", "w")
    prediction = []
    groundtruth = []
    for indx in xrange(len(img_id)):
        # count += 1
        # print template.format(count, img_id[indx] ,predictions[indx][0], int(predictions[indx][1]), predictions[indx][2])
        ##ConfusionMatrix O/P
        prediction.append(str(predictions[indx][0]))
        groundtruth.append(str(int(predictions[indx][1])))
        f.write(str(img_id[indx]) + ' ' + str(predictions[indx][0]) +'\n')
        # f.write(str(img_id[indx]) + ' ' + str(predictions[indx][0]) + ' ' + str(int(predictions[indx][1])) + ' ' + str(int(predictions[indx][2])) + '\n')
    f.close()
    print "\n5. RESULTS"
    print '\nCLASSIFICATION ACCURACY: ', accuracy*100,'%'
    confusion_m(prediction, groundtruth)


#K-NEAREST NEIGHBORS
##COMPUTE ACCURACY & CONFUSION MATRIX
def calculateaccuracy(predictions, groundtruth, image_titles):
    accuracy = 0
    f = open("knn_output.txt","w")
    for index in range(len(predictions)):
        f.write(image_titles[index]+" "+predictions[index]+"\n")
        if predictions[index] == groundtruth[index]:
            accuracy +=1
    print "\n3. RESULTS"
    print "\nCLASSIFICATION ACCURACY: ",(accuracy/len(final_label))*100, '%'
    f.close()
    confusion_m(predictions, groundtruth)

def findKnn(distance, k):
    global train_label
    global test_label
    global final_label
    l = []
    total=0
    for i in range(k):
        d = min(distance)
        # t.append(d)
        index = distance.index(d)
        distance[index] = sys.maxint
        l.append(train_label[index])
    count = Counter(l)
    final_label.append(count.most_common()[0][0])

def traindata(train_data,test_example, k):
    distance = []
    for instance in train_data:
            distance.append(np.sum((abs(np.subtract(test_example,instance)))))
    findKnn(distance,k)

def knn():
    global train_file
    global test_file
    global parameter
    global train_label
    global imageDifferences
    global test_label
    global final_label
    global start_time

    # start_time = time.time()
    k = parameter

    train_file = open(train_file,'r')
    test_file = open(test_file,'r')
    train_data = []
    test_data = []
    imageDifferences = []
    count = 1
    str = ""
    data_list = []
    train_label = []
    test_label = []
    testimages = []
    final_label = []
    for line in test_file:
        data_list = map(int, line.split(' ')[2:])
        test_data.append(np.array(data_list))
        test_label.append(line.split(' ')[1])
        testimages.append(line.split(' ')[0])
    for line in train_file:
        data_list = map(int, line.split(' ')[2:])
        train_data.append(np.array(data_list))
        train_label.append(line.split(' ')[1])
    print "\n1. PRE-PROCESSING"
    # print "\t Pre-Processing Time Required: ", (time.time() - start_time)
    i = 0
    print"\n2. TESTING: "
    l_d = len(test_data)
    F10 = False
    F25 = False
    F50 = False
    F75 = False
    F100 = False
    for ex in test_data:
        i += 1
        p_complete = (i/l_d)*100
        # print p_complete
        if int(p_complete) == 10 and F10 == False:
            F10 = True
            print "\tTesting Progress: ", 10, '%'
        elif int(p_complete) == 25 and F25 == False:
            F25 = True
            print "\tTesting Progress: ", 25, '%'
        elif int(p_complete) == 50 and F50 == False:
            F50 = True
            print "\tTesting Progress: ", 50, '%'
        elif int(p_complete) == 50 and F50 == False:
            F75 = True
            print "\tTesting Progress: ", 75, '%'
        elif int(p_complete) == 100 and F100 == False:
            F100 = True
            print "\tTesting Progress: ", 100, '%'

        if(len(ex) != 0):
            traindata(train_data,ex, k)
    # print"\tTesting | Time Required: ", time.time() - start_time
    calculateaccuracy(final_label,test_label,testimages)
    return

##INPUT
train_file = sys.argv[1]
test_file = sys.argv[2]
l_method = sys.argv[3]
parameter = sys.argv[4]



##CHOOSE ALGORITHMS
if l_method == 'nnet':
    print "\n\nALGORITHM: Neural Networks\n"
    parameter = int(parameter)
    nnet(0)
elif l_method == 'best':
    print "\n\nBEST ALGORITHM: Neural Networks\n"
    parameter = 125
    nnet(1)
elif l_method == 'knn':
    print "\n\nALGORITHM: K Nearest Neighbors\n"
    parameter = int(parameter)
    knn()
else: print 'Algorithm selection not understood. Please try again!'
