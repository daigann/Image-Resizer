# David Mehovic
# CPSC 4430 - Introduction to Machine Learning
# Assignment 3

# The import needed to successfully create and run an autoencoder
import numpy as np
import glob
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

# Here I've created two variants of the training_data and testing_data.
# The first training data contains noise and a smaller image size
# While training_data2 contains a larger image to use as a test of what the image should look like
training_data = []
training_data2 = []

# as with training data above I've done the same with testing data. Its so the smaller image goes
# in as an input while the larger is the desired de-noised output
testing_data = []
testing_data2 = []

# using glob to open both training and testing files
files = glob.glob('train/*.jpg')
files2 = glob.glob('test1/*.jpg')

# the for loop here grabs an image resizes it to 105 then puts it into the array
# after that I resize the same image into a smaller sample then put that into the second array
# this way I've got my target size and the training, input size.
for filepath in files:
    image = Image.open(filepath)
    image = image.resize((105, 105))
    training_data2.append(np.array(image))
    image = image.resize((28, 28))
    training_data.append(np.array(image))

# The same as above applies here. In that I take in the image, resize it to 105x105 and put it in an array
# then I resize it again and put it in the second array
for filepath2 in files2:
    image2 = Image.open(filepath2)
    image2 = image2.resize((105, 105))
    testing_data2.append(np.array(image2))
    image2 = image2.resize((28, 28))
    testing_data.append(np.array(image2))

# the next four blocks of code take my arrays and resizes them to an appropriate size for the encoder
training_data = np.array(training_data)
training_data = np.resize(training_data, (len(training_data), 28, 28, 1))
training_data = training_data.astype('float32') / 255.

training_data2 = np.array(training_data2)
training_data2 = np.resize(training_data2, (len(training_data2), 105, 105, 1))
training_data2 = training_data2.astype('float32') / 255.

testing_data = np.array(testing_data)
testing_data = np.resize(testing_data, (len(testing_data), 28, 28, 1))
testing_data = testing_data.astype('float32') / 255.

testing_data2 = np.array(testing_data2)
testing_data2 = np.resize(testing_data2, (len(testing_data2), 105, 105, 1))
testing_data2 = testing_data2.astype('float32') / 255.

# adding noise to new variables named x_train_noisy and x_test_noisy from the 28x28 arrays.
# this lets me have a non noisy test / desired output array where the encoder can see how the image
# should look
noise_factor = 0.5
x_train_noisy = training_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_data.shape)
x_test_noisy = testing_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=testing_data.shape)

# more noising
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

# I'm not experienced enough in encoder to understand fully how this works
# I basically added, removed and adjusted values here till my cat image appeared as a blob
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

# I added a few more layers with random values to try and get above 56x56 as I'm still
# inexperienced with encoders. Having said that I played around with this enough that I felt
# a 70x70 output layer would suffice for this. I had higher dimensions but the output was
# either impossible to tell what it was or the output was just a solid color. With these
# dimensions my cat image looks like a yellow blob but it is the closest I've gotten to
# something to represent a cat.
x = Conv2D(64, (2, 2), activation='relu', padding='same')(encoded)
x = UpSampling2D((3, 3))(x)
x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(64, (2, 2), activation='relu')(x)
x = UpSampling2D((3, 3))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# the autoencoder comes together here
autoencoder = Model(input_img, decoded)

# leaving this in here as this allowed me to see the output player dimensions to determine
# if I was above 56x56 or not
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, training_data2,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, testing_data2),
                callbacks=[TensorBoard(log_dir='', histogram_freq=0, write_graph=False)])

# after the model is trained I save it into the file model.hdf5
autoencoder.save('model.hdf5')

# this is from #7 where I export my image along with the actual cat image
model = load_model('model.hdf5')
orig_image = Image.open('test1/12000.jpg')

# I added this part in because I trained with a grayscale image and these instructions expected
# a rgb image. Changing the cat image to gray allowed me to resize and print the images
# without any issues
orig_image = ImageOps.grayscale(orig_image)
pred_image = np.array(orig_image.copy().resize((28, 28)))
generated = autoencoder.predict(pred_image.reshape((1, 28, 28, 1)))
truth = orig_image.copy().resize((105, 105))
generated_image = np.reshape(generated, (105, 105, 1))
generated_image = array_to_img(generated_image)
generated_image = np.array(generated_image)
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.imshow(pred_image)
ax2.imshow(generated_image)
ax3.imshow(truth)
plt.show()