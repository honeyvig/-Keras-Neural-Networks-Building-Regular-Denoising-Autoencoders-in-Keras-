# -Keras-Neural-Networks-Building-Regular-Denoising-Autoencoders-in-Keras-
Keras & Neural Networks: Building Regular and Denoising Autoencoders

Autoencoders are a type of neural network used for unsupervised learning. They work by encoding the input data into a compact representation (latent space) and then decoding it back to the original input, minimizing the reconstruction error.
Types of Autoencoders:

    Regular Autoencoder: The simplest form, it learns to compress (encode) and reconstruct (decode) the input data.
    Denoising Autoencoder (DAE): A variation of autoencoders that is trained by adding noise to the input, forcing the network to learn how to remove the noise and reconstruct the clean data.

We'll implement both types of autoencoders in Keras using Python, with examples using the MNIST dataset.
1. Setup and Imports

First, let's import the necessary libraries:

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

2. Loading the MNIST Dataset

We’ll use the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9).

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the images to the range [0, 1] by dividing by 255
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the images to 1D vectors of size 784 (28x28)
x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

# Print the shape of the training and test sets
print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

3. Building the Regular Autoencoder

In this part, we will create a regular autoencoder. The network will consist of an encoder (compressing the data) and a decoder (reconstructing it).

# Size of the encoding (latent) space
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784-dimensional

# Input layer
input_img = Input(shape=(784,))

# Encoder: Compressed representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Decoder: Reconstructing the input
decoded = Dense(784, activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = Model(input_img, decoded)

# Define the encoder model to extract the compressed representation
encoder = Model(input_img, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

4. Evaluating the Regular Autoencoder

We can evaluate the performance of the regular autoencoder by visualizing some of the reconstructed images:

# Encode and decode some test images
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# Visualize the original and reconstructed images
n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

5. Building the Denoising Autoencoder

A Denoising Autoencoder (DAE) is trained by adding noise to the input data, making it more robust. It learns to remove the noise and reconstruct the original input.

# Add random noise to the input images
def add_noise(x, noise_factor=0.5):
    noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    noisy = np.clip(noisy, 0., 1.)  # Make sure the pixel values are between 0 and 1
    return noisy

# Add noise to the training and test images
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# Define the denoising autoencoder architecture (similar to the regular autoencoder)
input_img = Input(shape=(784,))

# Encoder
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Decoder
decoded = Dense(784, activation='sigmoid')(encoded)

# Define the denoising autoencoder model
denoising_autoencoder = Model(input_img, decoded)

# Compile the denoising autoencoder model
denoising_autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# Train the denoising autoencoder with noisy images
denoising_autoencoder.fit(x_train_noisy, x_train,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test))

6. Evaluating the Denoising Autoencoder

Just like with the regular autoencoder, we can visualize the noisy input and the output of the denoising autoencoder.

# Predict the reconstructed images
decoded_imgs_dae = denoising_autoencoder.predict(x_test_noisy)

# Visualize noisy images and reconstructed images
plt.figure(figsize=(20, 4))
for i in range(n):
    # Noisy image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed image from the denoising autoencoder
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs_dae[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Original image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

7. Summary of Differences

    Regular Autoencoder:
        Learns to compress data and reconstruct it with minimal error.
        Used for dimensionality reduction, feature learning, etc.
    Denoising Autoencoder (DAE):
        Trained by adding noise to the input and learning to reconstruct the clean input.
        Useful for noise reduction, anomaly detection, and learning more robust features.

8. Use Cases for Autoencoders

    Dimensionality Reduction: Autoencoders can be used to reduce the dimensions of data, similar to techniques like PCA (Principal Component Analysis), but they can often capture more complex structures.
    Data Denoising: Denoising autoencoders can help to clean noisy data.
    Anomaly Detection: By training autoencoders on normal data, anomalies or outliers can be detected when the reconstruction error is high.
    Image Compression: Autoencoders can learn compact representations of images and compress them for storage or transmission.
    Feature Learning: Autoencoders are widely used for unsupervised feature learning, where they learn meaningful features from raw data without supervision.

Conclusion

We’ve built both regular and denoising autoencoders using Keras for the MNIST dataset. The regular autoencoder learns to compress and reconstruct the input data, while the denoising autoencoder adds noise to the input during training to make the model more robust. Both types of autoencoders have numerous applications in fields such as data compression, anomaly detection, and image processing.
