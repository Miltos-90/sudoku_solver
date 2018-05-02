from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import MaxPool2D, Flatten, Dense
from keras import backend as k
import pandas as pd
import numpy as np


class  Digit_Recognizer:
    
    
    def __init__(self):
        # Instantiate the ConvNet
        model = self.create_model()
        # Load the pretrained model
        model.load_weights('./helpers/digit_recognition/checkpoints/model.67-0.02.h5')
        self.model = model
        self.error = False
    
    
    @staticmethod
    def create_model():
        # Create the ConvNet
        rows = 28; cols=28; depth = 1 
        # Instantiate a sequential model
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
        # Compile
        model.compile(loss='categorical_crossentropy', 
                      optimizer='rmsprop', metrics=['accuracy'])
        return model
    
    
    @staticmethod
    def reshape_dset(cells):
        rows = 28; cols=28; depth = 1 
        # Get the position of the non-zero cells
        non_zero_idx = [idx for idx, cell in enumerate(cells) if cell.shape == (rows, cols)]
        # Isolate the non-zero cells to fit to the model
        x = [cell.flatten()  for cell in cells if cell.shape == (rows, cols)] 
        # Reshape the cells accordigly for the model
        x = pd.DataFrame(x)
        if k.image_data_format() == 'channels_first':
            x = x.values.reshape(x.shape[0], depth, rows, cols)
        else: # channels last
            x = x.values.reshape(x.shape[0], rows, cols, depth)
        return non_zero_idx, x
    
    
    def predict(self, frame):
        idx, x = self.reshape_dset(frame.cells)
        if len(x) > 0:
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis = 1)
            y_pred = pd.DataFrame(data = y_pred, index = idx, columns = ['digit'])
            y_pred = y_pred.reindex(range(np.prod(frame.dims))).fillna(0)
            y_pred = y_pred.values.reshape(frame.dims).astype(int)
        else:
            self.error = True
            y_pred = []
        return y_pred
