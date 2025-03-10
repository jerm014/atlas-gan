#!/usr/bin/env python3
"""Deep Convolutional Generative Adversarial Network"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.layers import Input, Dense, Reshape  # type: ignore
from tensorflow.keras.layers import Flatten, Dropout  # type: ignore
from tensorflow.keras.layers import BatchNormalization  # type: ignore
from tensorflow.keras.layers import Activation, LeakyReLU  # type: ignore
from tensorflow.keras.layers import Conv2D, Conv2DTranspose  # type: ignore
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf
import os


class DCGAN():
    """Deep Convolutional Generative Adversarial Network, bet."""
    def __init__(self, img_rows=28, img_cols=28, channels=1, latent_dim=100):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # G: take noise as input and generate images
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train G
        self.discriminator.trainable = False

        # D take generated images as input and determine validity
        validity = self.discriminator(img)

        # combined model (stacked G and D)
        # Train the G to fool the D
        self.combined = Model(z, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )

        os.makedirs('images', exist_ok=True)

    def load(self):
        """
        Load the Modified National Institute of Standards and
        Technology dataset
        """
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Print shapes to show data was loaded
        print(f"Loaded data shapes:")
        print(f"x_train: {self.x_train.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"x_test:  {self.x_test.shape}")
        print(f"y_test:  {self.y_test.shape}")

    def preprocess(self):
        """Preprocess"""
        # 1. normalize the pixel values (from 0-255 to 0-1)
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # 2. reshape the data for the model (CNN)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        # 3. one-hot encode the labels
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

        # 4. split training data to create a validation set
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(self.x_train, self.y_train,
                             test_size=0.1, random_state=42)
        
        # print preprocessed ndarray shapes
        print(f"Preprocessed data shapes:")
        print(f"x_train: {self.x_train.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"x_val:   {self.x_val.shape}")
        print(f"y_val:   {self.y_val.shape}")
        print(f"x_test:  {self.x_test.shape}")
        print(f"y_test:  {self.y_test.shape}")

    def build_generator(self):
        """Build the G"""
        
        # Dense layer: A fully connected layer with 7 * 7 * 256 = 12,544
        # neurons, no bias terms, and receiving input from a latent vector of
        # dimension self.latent_dim
        x1 = Dense(7 * 7 * 256, use_bias=False, input_dim=self.latent_dim)

        # BatchNormalization layer: Normalizes the activations from the Dense
        # layer, helping with training stability
        x2 = BatchNormalization()

        # LeakyReLU layer: An activation function with a small positive slope
        # (0.2) for negative inputs, adding non-linearity to the model
        x3 = LeakyReLU(alpha=0.2)

        # Reshape layer: Transforms the flat output from the previous layers
        # into a 3D tensor with shape (7, 7, 256)
        x4 = Reshape((7, 7, 256))

        # upsample -> 14x14
        x5 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',
                             use_bias=False)
        x6 = BatchNormalization()
        x7 = LeakyReLU(alpha=0.2)

        # Upsample -> 28x28
        x8 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',
                             use_bias=False)
        x9 = BatchNormalization()
        x10 = LeakyReLU(alpha=0.2)

        # add output layer
        x11 = Conv2D(self.channels, (5, 5), padding='same', activation='tanh')

        model = Sequential([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """Build the D"""

        # convolutional layer 1
        x1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                    input_shape=self.img_shape)
        x2 = LeakyReLU(alpha=0.2)
        x3 = Dropout(0.3)

        # convoltional layer 2
        x4 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        x5 = LeakyReLU(alpha=0.2)
        x6 = Dropout(0.3)

        # flatten
        x7 = Flatten()
        # add output layer
        x8 = Dense(1, activation='sigmoid')

        model = Sequential([x1, x2, x3, x4, x5, x6, x7, x8])
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        """train the G and the D"""
        # adversarial ground truths:
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # lets train that D

            # select a random batch of images
            idx = np.random.randint(0, self.x_train.shape[0], batch_size)
            imgs = self.x_train[idx]

            # sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # lets train that G

            g_loss = self.combined.train_on_batch(noise, valid)

            # print our progress
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: "
                  f"{100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

            # if at a save interval save those generated image samples
            if epoch % save_interval == 0:
                self.save_images(epoch)

    def save_images(self, epoch):
        """Save images for an epoch"""
        r, c = 5, 5  # 5x5 grid of images
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # rescale images to 0-1 range
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        
        # save the figure
        fig.savefig(f"images/mnist_{epoch}.png")
        plt.close()

    def show_sample_images(self, n_samples=5):
        """Display sample images to verify data is loaded correctly"""
        if self.x_train is None:
            print("Data not loaded. Please call load() first?!")
            return
            
        # make a figure with n_samples columns
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples*2, 2))
        
        # indices: random indices from training set
        indices = np.random.choice(self.x_train.shape[0],
                                   n_samples,
                                   replace=False)
        
        # for each of the indices plot the image
        for i, idx in enumerate(indices):
            # if data has been preprocessed (reshaped) squeeze out the channel
            # dimension
            if len(self.x_train.shape) == 4:
                axes[i].imshow(self.x_train[idx].squeeze(), cmap='gray')
            else:
                axes[i].imshow(self.x_train[idx], cmap='gray')
            
            # if labels have been one-hot encoded find the index of the 1
            if len(self.y_train.shape) == 2:
                label = np.argmax(self.y_train[idx])
            else:
                label = self.y_train[idx]
                
            axes[i].set_title(f"Digit: {label}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

    def generate_images(self, n_samples=25):
        """Generate and display images using the trained generator"""
        # generate random noise
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))

        # generate images from noise
        gen_imgs = self.generator.predict(noise, verbose=0)

        # rescale images to 0-1 range
        gen_imgs = 0.5 * gen_imgs + 0.5

        # create a grid to display the images
        rows = int(np.sqrt(n_samples))
        cols = int(np.sqrt(n_samples))

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                cnt += 1

        plt.tight_layout()
        fig.savefig(f"images/mnist_trained.png")
        plt.show()


if __name__ == "__main__":
    # Create an instance of DCGAN
    dcgan = DCGAN()
    
    # Load the data
    print("Loading MNIST dataset...")
    dcgan.load()
    print("\nDisplaying sample images before preprocessing:")
    dcgan.show_sample_images()
    print("\nPreprocessing data...")
    dcgan.preprocess()
    print("\nDisplaying sample images after preprocessing:")
    dcgan.show_sample_images()
    print("\nSetup complete! DCGAN is ready for training.")
    # train the thing
    print("\nStarting DCGAN training...")
    dcgan.train(epochs=10000, batch_size=32, save_interval=400)
    # generate some images after training
    print("\nGenerating images using trained model:")
    dcgan.generate_images()