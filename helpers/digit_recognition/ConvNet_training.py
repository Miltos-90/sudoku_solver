import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as k
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import plot_model

def load_dset():
    # Load the datasets
    not_mnist = pd.read_csv('not_mnist.csv')
    mnist = fetch_mldata('MNIST original')
    mnist = pd.DataFrame(data= np.c_[mnist['target'], mnist['data']],
                         columns= not_mnist.columns)
    # Combine the two datasets
    dset = pd.concat([mnist, not_mnist], axis = 0, ignore_index = True)
    # Split targets
    y = dset['label']
    x = dset.drop(['label'], axis = 1)
    return x, y


def reshape_dset(x, rows, cols, depth):
    if k.image_data_format() == 'channels_first':
        x = x.values.reshape(x.shape[0], depth, rows, cols)
        input_shape = (depth, rows, cols)
    else: # channels last
        x = x.values.reshape(x.shape[0], rows, cols, depth)
        input_shape = (rows, cols, depth)        
    return input_shape, x


def create_model(rows, cols, depth):
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), 
                     padding = 'Same', input_shape = (rows, cols, depth)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), 
                     padding = 'Same', input_shape = (rows, cols, depth)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # Final Classifier
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
        
    # Compile the model
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


def load_trained_model(weights_path, rows, cols, depth):
    model = create_model(rows, cols, depth)
    model.load_weights(weights_path)
    return model


def show_results(y_test, y_pred):
    # Reverse categorical tranformation
    true_cls, predicted_cls = [np.argmax(elem, axis = 1) for elem in [y_test, y_pred]]
    target_names = [str(i) for i in range(10)] # define the class labels
    print(classification_report(true_cls, predicted_cls, target_names = target_names))
    target_names = [i for i in range(10)]
    print(confusion_matrix(true_cls, predicted_cls, labels = target_names))
    return 


def show_errors(x_test, y_test, y_pred):
    # Print the erroneous results
    errors = [(x_test[i], y_test[i,:], y_pred[i,:])  for i in range(x_test.shape[0])
                if np.argmax(y_test[i,:]) - np.argmax(y_pred[i,:]) != 0]
    
    plt.figure(figsize=(8, 10))
    for idx, (img, true, prediction) in enumerate(errors):
        plt.subplot(5, 4, idx + 1)
        true_cls = np.argmax(true)
        pred_cls = np.argmax(prediction)
        pred_prob = max(prediction)
        plt.imshow(img[:,:,0])
        plt.title('true:{}, pred:{} ({:1.2f})'.format(true_cls, pred_cls, pred_prob))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return


def show_acc_loss(df):
    # Plot accuracy and loss values
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax1.plot(df['epoch'], df['val_acc'], '.-', label = 'validation set')
    ax1.plot(df['epoch'], df['acc'], '--', label = 'training set')
    ax1.set_ylim([0.97, 1])
    ax1.legend()
    ax1.set_title('Accuracy')
    ax2.plot(df['epoch'], df['val_loss'], '.-', label = 'validation set')
    ax2.plot(df['epoch'], df['loss'], '--', label = 'training set')
    ax2.legend()
    ax2.set_title('Loss')
    plt.show()
    return


if __name__=="__main__":
    
    # Load the datasets
    x, y = load_dset()
    
    # Shape data for the model
    rows = 28; cols=28; depth = 1 # grayscaled images - only 1 channel used 
    shape, x = reshape_dset(x, rows, cols, depth)
    
    # Normalize data
    x /= 255.0
    
    # Categorical Encoding of the targets
    y = to_categorical(y)
    
    # Stratified train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.2)
    
    # Split the train set into a train (again) set and a validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify = y_train, test_size = 0.1)
    
    # Create and plot the model
    model = create_model(rows, cols, depth)
    plot_model(model, to_file='model.png')
    
    # Data augmentation
    augment = ImageDataGenerator(featurewise_center = False, 
                                 samplewise_center = False, 
                                 featurewise_std_normalization = False,  
                                 samplewise_std_normalization = False,  
                                 zca_whitening = False,  
                                 rotation_range = 15,  
                                 zoom_range = 0.1, 
                                 width_shift_range = 0.1,  
                                 height_shift_range = 0.1,  
                                 horizontal_flip = False,  
                                 vertical_flip = False)  
    
    augment.fit(x_train)
    
    # Create callbacks
    tboard = TensorBoard(log_dir = './logs', histogram_freq = 1)
    
    checkpoint = ModelCheckpoint("checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5", 
                                 monitor = 'val_loss')
    
    early_stop = EarlyStopping(monitor='val_loss', patience = 10)
    
    csv_logger = CSVLogger('training.log', append = True)
    
    callbacks=[tboard, checkpoint, early_stop, csv_logger]
    
    print("Run the followig command in cmd: tensorboard --logdir C:/Users/Miltos/Desktop/logs")
    
    # Train the model
    res = model.fit_generator(augment.flow(x_train, y_train, batch_size = 32), epochs = 120, 
                              validation_data = (x_val, y_val), verbose = 1, callbacks = callbacks)
    
    # Check the results
    df = pd.read_table('training.log', sep = ',')

    sns.set() # Set aesthetic parameters
    
    # Load the model and check on the test set
    model_path = 'checkpoints/model.67-0.02.h5' # Get the best model
    model = load_trained_model(model_path, rows, cols, depth)
    y_pred = model.predict(x_test)
    
    # Inverse categorical encoder, and print results / errors
    show_acc_loss(df)
    show_results(y_test, y_pred)
    show_errors(x_test, y_test, y_pred)
    
    