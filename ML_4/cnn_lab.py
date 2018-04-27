#Convolutional Neural Network

#Building the CNN

#Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialization of CNN
#Building CNN model, setting layers
classifier = Sequential()

#Step 1 Convolution layer - layer for filtering by mask repeatedly and change negative values using Rectified Linear Units
classifier.add(Convolution2D(32,(3,3), padding = 'same', input_shape = (64,64,3), activation = 'relu'))
#Step 2 MaxPooling - shrinking filtered images, taking maximum value inside area
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 Second convolution layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 4 Flattening - get an 1D array of images
classifier.add(Flatten())

#Step 5 Full Connection - connected layer with full connections to all activations in the previous layers. Connects neurons of one layer to another
classifier.add(Dense(128, activation = 'relu'))  #Input Layer
classifier.add(Dense(1, activation = 'sigmoid')) #Output Layer

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

# Here is done real-time data augmentation, generating packages of tensor images data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Take a all pathes in directory, creating batches of data
training_set = train_datagen.flow_from_directory('dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


classifier.fit_generator(training_set,
    steps_per_epoch=8000,
    epochs=3,
    validation_data=test_set,
    validation_steps=2000)

# Prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/training_set/cats/cat.37.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
ind = training_set.class_indices

if(result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Model saved")