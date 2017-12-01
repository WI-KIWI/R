
# For questions and suggestions, please contact

# Author: Zilong Zhao, 
# who works in
# Big Data Analytics Team, 
# CSC Deutschland GmbH, a Next-Gen IT Company, 
# E-Mail: zzhao3@csc.com
#######################################################################################


 
#!/opt/miniconda/bin/python

## import library
##########################
import MPU6050
import math
import time
from dateutil import tz
from datetime import datetime
import scipy.fftpack
import numpy as np
import json
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import mean_squared_error

##########################


# this function read the data from pi accelerometer and save it as a parameter in memory
# to use it, in python: data = readMPU6050(1024), 1024 is the number of sampling data,
# this function will output the sampling time
def readMPU6050(input_Counter):      

    mpu6050 = MPU6050.MPU6050()
    mpu6050.Setup()
#sometimes I need to reset twice
    I = mpu6050.ReadData()
    mpu6050.Setup()
    Counter=0
    while True:

      while not(mpu6050.GotNewData()):
        pass
      I = mpu6050.ReadData()
#  CurrentForce = math.sqrt( (I.Gx * I.Gx) + (I.Gy * I.Gy) +(I.Gz * I.Gz))

      if Counter == 0 :
           FO = []
           Counter = input_Counter
           t1 = time.time()
    
      if Counter > 0:
#     FO.write("{0:6.3f}\t{1:6.3f}\t{2:6.3f}\t{3:6.3f}\n".format(CurrentForce, I.Gx , I.Gy, I.Gz))
#     FO.write("{0:6.4f}\t{1:6.4f}\t{2:6.4f}\n".format(I.Gx , I.Gy, I.Gz))
         FO.append([I.Gx , I.Gy, I.Gz, time.time()])
         Counter = Counter - 1
         if Counter == 0:
           t2 = time.time()
       #FO.close()
           Peak=0
           #print "Elapse for sampling  = {0}".format(t2-t1)
           break   
#print FO
    return FO



# this function split the sensor data into three directions and plus the time information
# to use it, in python: X, Y, Z, timestamp = splitSensorData(data)
def splitSensorData(input_data):
    temp = map(list, zip(*input_data))
    X = temp[0]
    Y = temp[1]
    Z = temp[2]
    timestamp = temp[3]
## avoid loop for pi computing    
#    X = []
#    Y = []
#    Z = []
#    timestamp = []
#    for i in range(0, len(input_data)):
#        X.append(input_data[i][0])
#        Y.append(input_data[i][1])
#        Z.append(input_data[i][2])
#        timestamp.append(input_data[i][3])
    return X, Y, Z, timestamp


# this function use the sensor data for caculating the second norm of 
# three directions and output it with timestamp
def fast_calc_XYZ(input_data):
    temp = map(list, zip(*input_data))
    X = temp[0]
    Y = temp[1]
    Z = temp[2]
    timestamp = temp[3]
    XYZ=[]
    for i in range(0, len(X)):
        XYZ.append(math.sqrt( X[i]**2 + Y[i]**2 + Z[i]**2 ))
    return XYZ, timestamp


# this function compute the frequency spectrum by using FFT algorithm from scipy
# package.
def fastfourier(input_array, time_array):
    # Number of samplepoints
    N = len(input_array)
    freq = N / ( max(time_array) - min(time_array) )
    # sample spacing
    T = 1.0 / freq
    yf = scipy.fftpack.fft(input_array)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    yff = 2.0/N * np.abs(yf[:N/2])
    return xf, yff, N


# this function output the Fourier analysis in JSON format for live
# visualization, the input xf, yff, N come from the function
# fastfourier().
def fft2json(xf, yff, N):
    index_to_smooth = 0
    yff[index_to_smooth] = 0
    fft_python_object = {"title": "Frequency spectrum", 
            "x-label": "Freq [Hz]", "y-label": "Power",
            "x-value": xf.tolist(), "y-value": yff.tolist(), 
            "n-values": N/2 } 
    json_string = json.dumps(fft_python_object)
    return json_string

# output as json file
def fft2jsonfile(xf, yff, N):
    index_to_smooth = 0
    yff[index_to_smooth] = 0
    fft_python_object = {"title": "Frequency spectrum", 
            "x-label": "Freq [Hz]", "y-label": "Power |Y(f)|",
            "x-value": xf.tolist(), "y-value": yff.tolist(), 
            "n-values": N/2 } 
    with open('/srv/http/assets/data/spectrum.txt', 'w') as outfile:
        json.dump(fft_python_object, outfile, sort_keys = True, indent = 4, ensure_ascii=False)
    return


# this function will be used for data sampling for a measurement periode, it use the 
# functions we have above, as parameter, num_rows is how many rows after the fft,
# i.e. seconds, we use json as output format
def datasamplingfft2json(num_rows):
    sampling_python_object={}
    index_to_smooth = 0
    for i in range(0, num_rows):
        data = readMPU6050(1024)
        XYZ, timestamp = fast_calc_XYZ(data)
        xf, yff, N = fastfourier(XYZ, timestamp)
        yff[index_to_smooth] = 0
        sampling_python_object[i] = {'timestart': min(timestamp), 'timestop': max(timestamp), 'x-value': xf.tolist(), 'y-value': yff.tolist()}
    return sampling_python_object

def datasamplingfft2jsonfile(num_rows, file_name='measurement.txt'):
    sampling_python_object={}
    index_to_smooth = 0
    for i in range(0, num_rows):
        data = readMPU6050(1024)
        XYZ, timestamp = fast_calc_XYZ(data)
        xf, yff, N = fastfourier(XYZ, timestamp)
        yff[index_to_smooth] = 0
        sampling_python_object[i] = {'timestart': min(timestamp), 'timestop': max(timestamp), 'x-value': xf.tolist(), 'y-value': yff.tolist()}
    with open('/opt/miniconda/Projects/demobox/data/' + file_name , 'w') as outfile:
        json.dump(sampling_python_object, outfile, sort_keys = True, indent = 4, ensure_ascii=False)
    return


# this function will be used to sample the data in 1 second, transform it with fft and save it local as json for visualization, 
# also give a json variable in memory. 
def datasampling4pred():
    sampling_python_object={}
    index_to_smooth = 0
    data = readMPU6050(1024)
    XYZ, timestamp = fast_calc_XYZ(data)
    xf, yff, N = fastfourier(XYZ, timestamp)
    yff[index_to_smooth] = 0
    sampling_python_object = {'timestart': min(timestamp), 'timestop': max(timestamp), 'x-value': xf.tolist(), 'y-value': yff.tolist()}
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Europe/Berlin')
    utc = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max(timestamp))), '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    fft_python_object = {"title": "Frequency spectrum", 
            "x-label": "Freq [Hz]", "y-label": "Power |Y(f)|",
            "x-value": xf.tolist(), "y-value": yff.tolist(),
            "n-values": N/2, "sampling-time": utc.astimezone(to_zone).strftime('%Y-%m-%d %H:%M:%S %Z%z') }
    with open('/srv/http/assets/data/spectrum.txt', 'w') as outfile:
        json.dump(fft_python_object, outfile, sort_keys = True, indent = 4, ensure_ascii=False)
    return sampling_python_object


# a custom function to read json variable of 1 second fft data from memory to numpy array X, y
# as output, we have X, y, timestart, timestop
def ReadJsonVar2numpyarray(json_var):
    d = json_var
    X_temp = np.array(d['x-value'])
    y_temp = np.array(d['y-value'])
    timestart = np.array(d['timestart'])
    timestop = np.array(d['timestop'])
    return X_temp.reshape(1, len(X_temp)), y_temp.reshape(1, len(y_temp)), timestart, timestop








# a custom function to read one json txt measurement data to numpy array X, y
# as output, we have X, y, timestart, timestop
def ReadJson2numpyarray(onefile):
    with open(onefile) as json_data:
        d = json.load(json_data)
        X_temp=np.zeros((len(d), len(np.array(d[str(0)]['x-value']))), dtype=float)
        y_temp=np.zeros((len(d), len(np.array(d[str(0)]['y-value']))), dtype=float)
        timestart=np.zeros((len(d), 1), dtype=float)
        timestop=np.zeros((len(d), 1), dtype=float)
        for i in range(0, len(d)):
            for j in range (0, len(np.array(d[str(i)]['x-value']))):
                X_temp[i, j] = np.array(d[str(i)]['x-value'])[j]
                y_temp[i, j] = np.array(d[str(i)]['y-value'])[j]
            timestart[i, 0] = d[str(i)]['timestart']
            timestop[i, 0] = d[str(i)]['timestop']

    return X_temp, y_temp, timestart, timestop


# a function for multiple json files read to 2-dim numpy array
def ReadJsonfiles2numpyarray(filelist):
    X_out=[]
    y_out=[]
    timestart_out=[]
    timestop_out=[]
    for i, onefile in enumerate(filelist):
        X_temp, y_temp, timestart_temp, timestop_temp = ReadJson2numpyarray(onefile)
        X_out.append(X_temp)
        y_out.append(y_temp)
        timestart_out.append(timestart_temp)
        timestop_out.append(timestop_temp)
    X_out, y_out, timestart_out, timestop_out = np.array(X_out), np.array(y_out), np.array(timestart_out), np.array(timestop_out)
    X_out = np.vstack(X_out[i,:,:] for i in range(X_out.shape[0]))
    y_out = np.vstack(y_out[i,:,:] for i in range(y_out.shape[0]))
    timestart_out = np.vstack(timestart_out[i,:,:] for i in range(timestart_out.shape[0]))
    timestop_out = np.vstack(timestop_out[i,:,:] for i in range(timestop_out.shape[0]))
    return X_out, y_out, timestart_out, timestop_out






########################################################################################################################################



# Data Preparation


# a function to prepare the data for prediction, 
# y_test is the fft signal from real time measurement
def prep4pred(y_test):
    # import both numpy array from local for time-saving
    X_train = np.load('./X_train.npy')
    X_normal = np.load('./X_normal.npy')

    # using min_max_scaler to scale the data, we pre-saved X_train, X_normal
    # for scaling and calculation
    min_max_scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_normal_scaled = min_max_scaler.transform(X_normal)
    y_normal_actual = X_normal

    # y_test is the fft signal from real time measurement
    X_test_scaled = min_max_scaler.transform(y_test)
    y_test_actual = y_test
    return X_normal_scaled, y_normal_actual, X_test_scaled, y_test_actual




#######################################################################################################################################

# main part

# Use TensorFlow for prediction, as input, X_forPrediction is the real-time fft signal from measurement
def tf4prediction(X_forPrediction):
    n_dim = X_forPrediction.shape[1]

    #learning_rate = 0.001
    #training_epochs = 40

    #batch_size = 4
    #display_step = 10
    #dropout_rate = 0.9
    # Network Parameters
    n_hidden_1 = 100 # 1st layer number of features
    n_input = n_dim
    n_output = n_dim
    #total_len = X_train_scaled.shape[0]

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    # Create model
    def onelayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_output], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_output], 0, 0.1))
    }


    # Construct model
    pred = onelayer_perceptron(x, weights, biases)


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state('./model/ML_model_1')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, './model/ML_model_1/AANN4AnoDet.ckpt')
            y_pred = sess.run(pred , feed_dict={x: X_forPrediction})
#            print ("Prediction Finished!")
        else:
            print 'No pre-saved model can be found.'
        sess.close()
    tf.reset_default_graph()
    return y_pred



#################################################################################################################################

# Function for evaluation

def cal_mse(data_array1, data_array2):
    oneMinute_MSE = []
    if data_array1.shape[0]==data_array2.shape[0]:
        for i in range(0, data_array1.shape[0]):
            oneMinute_MSE.append(mean_squared_error(data_array1[i], data_array2[i]))
    else:
        print 'the dimension of two arrays should be the same!'
    return oneMinute_MSE
