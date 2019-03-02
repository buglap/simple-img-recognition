from PIL import Image

# For show and image with PLI
# display_image_pathname= input("Enter pathname: ")
# display_image =  Image.open(display_image_pathname)
# display_image.show()

labels =['airplane', 'automovil', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from keras.datasets import cifar10

#(train images, train labels),(test images, test labels)-arrays
(X_train, y_train),(X_test, y_test) = cifar10.load_data()

# index = int(input('Enter and image index: '))
# #array of numerical (0,1) values that represent pixels in this cases 32x32
# display_image = X_train[index]
# display_label =  y_train[index][0] # the number it self and not an array

from matplotlib import pyplot as plt

#show the array as an image using PLI
# final_img =Image.fromarray(display_image)
# final_img.show()

# Display with a color
# red_img =Image.fromarray(display_image)
# red, green, blue = red_img.split()
# plt.imshow(red, cmap= "Reds")
# plt.show()

#show the array as an image using matplolib
# plt.imshow(display_image)
# plt.show()

#print(labels[display_label])

##Training Data##
from keras.utils import np_utils
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')

#for get values between 0-1
new_X_train /= 255
new_X_test /= 255

#readeble format
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)


##Building a Model##

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

model = Sequential()
#32*32 pixel, 3,3 advance, 3 colors form, helps for the orientation of the images does not affect anything
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#2-dimensional array
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#(32*32)/2 d-array =512
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#learning rate 0.01
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01), metrics=['accuracy'])


#Now for run our model, 10 times
model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

#for save our model already trained
import h5py
model.save('Trained-model.h5')

