#!/usr/bin/env python3
"""Deep Convolutional Generative Adversarial Network - Modified Architecture"""

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
    """Deep Convolutional Generative Adversarial Network with Modified Architecture"""
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

        os.makedirs('1_images', exist_ok=True)

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
        """Build the Modified Generator with more layers and different filter sizes"""
        
        # Modified: Starting with a smaller dense layer (7x7x128 instead of 7x7x256)
        x1 = Dense(7 * 7 * 128, use_bias=False, input_dim=self.latent_dim)
        x2 = BatchNormalization()
        x3 = LeakyReLU(alpha=0.2)
        x4 = Reshape((7, 7, 128))

        # First upsampling block (7x7 -> 14x14)
        # Modified: Using 3x3 filter size instead of 5x5
        x5 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                            use_bias=False)
        x6 = BatchNormalization()
        x7 = LeakyReLU(alpha=0.2)

        # Added: Intermediate convolutional layer without upsampling
        x7a = Conv2D(64, (3, 3), padding='same', use_bias=False)
        x7b = BatchNormalization()
        x7c = LeakyReLU(alpha=0.2)

        # Second upsampling block (14x14 -> 28x28)
        # Modified: Using 3x3 filter size and 32 filters instead of 5x5 and 64 filters
        x8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',
                            use_bias=False)
        x9 = BatchNormalization()
        x10 = LeakyReLU(alpha=0.2)

        # Added: Final refinement layer
        x10a = Conv2D(16, (3, 3), padding='same', use_bias=False)
        x10b = BatchNormalization()
        x10c = LeakyReLU(alpha=0.2)

        # Output layer
        # Modified: Using 3x3 filter size instead of 5x5
        x11 = Conv2D(self.channels, (3, 3), padding='same', activation='tanh')

        model = Sequential([x1, x2, x3, x4, x5, x6, x7, 
                          x7a, x7b, x7c, 
                          x8, x9, x10,
                          x10a, x10b, x10c,
                          x11])
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """Build the Modified Discriminator with deeper architecture"""

        # First convolutional layer
        # Modified: 32 filters with 3x3 kernel instead of 64 filters with 5x5 kernel
        x1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                   input_shape=self.img_shape)
        x2 = LeakyReLU(alpha=0.2)
        x3 = Dropout(0.3)

        # Added: Intermediate convolutional layer
        x3a = Conv2D(64, (3, 3), padding='same')
        x3b = LeakyReLU(alpha=0.2)
        x3c = Dropout(0.3)

        # Second convolutional layer
        # Modified: 128 filters with 3x3 kernel
        x4 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')
        x5 = LeakyReLU(alpha=0.2)
        x6 = Dropout(0.3)

        # Added: Third convolutional layer
        x6a = Conv2D(256, (3, 3), padding='same')
        x6b = LeakyReLU(alpha=0.2)
        x6c = Dropout(0.3)

        # Flatten and output
        x7 = Flatten()
        
        # Added: Dense layer before final output
        x7a = Dense(128)
        x7b = LeakyReLU(alpha=0.2)
        x7c = Dropout(0.3)
        
        x8 = Dense(1, activation='sigmoid')

        model = Sequential([x1, x2, x3, 
                          x3a, x3b, x3c,
                          x4, x5, x6,
                          x6a, x6b, x6c,
                          x7,
                          x7a, x7b, x7c,
                          x8])
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
        fig.savefig(f"1_images/mnist_{epoch}.png")
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
        fig.savefig(f"1_images/mnist_trained.png")
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
    print("\nSetup complete! Modified DCGAN is ready for training.")
    # train the thing
    print("\nStarting Modified DCGAN training...")
    # Modified: Using smaller number of epochs and larger batch size for comparison
    dcgan.train(epochs=10000, batch_size=64, save_interval=400)
    # generate some images after training
    print("\nGenerating images using trained model:")
    dcgan.generate_images()