#!/usr/bin/env python3
"""Much Better GAN - now with stability upgrades!"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
tf.config.optimizer.set_experimental_options({'remapping': False})


# make dir for pretty pictures and model saving
os.makedirs("generated_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


def load_cifar10():
    """ loading dem pixels """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = (
        x_train.astype("float32") - 127.5
    ) / 127.5  # Normalize to [-1, 1]
    return x_train


def build_generator(latent_dim=100):
    """ generatoring dem fake images """
    noise_input = layers.Input(shape=(latent_dim,))
    
    # First dense layer to project noise
    x = layers.Dense(4 * 4 * 512)(noise_input)
    x = layers.Reshape((4, 4, 512))(x)
    
    # Block 1: 4x4 -> 8x8
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same",
                               kernel_initializer='glorot_uniform')(x)
    
    # Block 2: 8x8 -> 16x16
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",
                               kernel_initializer='glorot_uniform')(x)
    
    # Block 3: 16x16 -> 32x32
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same",
                               kernel_initializer='glorot_uniform')(x)
    
    # Final conv to get 3 RGB channels with tanh activation
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(3, kernel_size=3, padding="same",
                      activation="tanh",
                      kernel_initializer='glorot_uniform')(x)
    
    # Build and return model
    model = models.Model(noise_input, x, name="super_generator")
    return model


def build_discriminator(img_shape=(32, 32, 3)):
    """ discriminatoring dem images """
    img_input = layers.Input(shape=img_shape)
    
    # Block 1: Initial conv with no batchnorm
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same",
                      kernel_initializer='glorot_uniform')(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same",
                      kernel_initializer='glorot_uniform')(x)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)  # Fix odd dims
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same",
                      kernel_initializer='glorot_uniform')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(512, kernel_size=3, strides=2, padding="same",
                      kernel_initializer='glorot_uniform')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Output block
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid',
                     kernel_initializer='glorot_uniform')(x)
    
    # Build and return model
    model = models.Model(img_input, x, name="awesome_discriminator")
    return model


def save_generated_images(epoch, G, latent_dim=100, examples=16):
    """ saving dem pretty pictures """
    noise = np.random.normal(0, 1, (examples, latent_dim))
    gen_images = G.predict(noise, verbose=0)
    gen_images = (gen_images * 127.5 + 127.5).astype(np.uint8)

    # Grid layout for much organization
    rows = cols = int(np.sqrt(examples))
    _, axs = plt.subplots(rows, cols, figsize=(rows, cols))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_images[idx])
            axs[i, j].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()


def train(epochs=250, batch_size=128, latent_dim=100, save_interval=50):
    """ training dem networks """
    # Load and prepare data
    x_train = load_cifar10()
    print(f"Training on {x_train.shape[0]} images of shape {x_train.shape[1:]}")
    
    # Build and compile models
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    
    # Use advanced optimizer settings for stability
    d_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    g_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=d_optimizer,
        metrics=["accuracy"]
    )
    
    # Build the combined model for generator training (freeze discriminator)
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_img = generator(gan_input)
    validity = discriminator(generated_img)
    combined = models.Model(gan_input, validity, name="very_cool_gan")
    combined.compile(
        loss="binary_crossentropy",
        optimizer=g_optimizer
    )
    
    # Track progress metrics
    d_losses = []
    g_losses = []
    
    # Dat training loop
    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Get a random batch of real images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        
        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)
        
        # Label smoothing for much stability
        real_labels = np.ones((batch_size, 1)) * 0.9  # Not quite 1.0
        fake_labels = np.zeros((batch_size, 1)) + 0.1  # Not quite 0.0
        
        # Train discriminator on real and fake batches separately
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size * 2, latent_dim))
        valid_labels = np.ones((batch_size * 2, 1))  # We want D to think these are real
        
        # Train generator to fool discriminator
        g_loss = combined.train_on_batch(noise, valid_labels)
        
        # Keep history for plotting
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} "
                  f"[D loss: {d_loss[0]:.4f}, acc: {d_loss[1]*100:.2f}%] "
                  f"[G loss: {g_loss:.4f}]")
        
        # Save images at specified intervals
        if epoch % save_interval == 0:
            save_generated_images(epoch, generator, latent_dim)
            
            # Save models periodically
            if epoch % (save_interval * 5) == 0:
                generator.save(f"saved_models/generator_epoch_{epoch}.keras")
                discriminator.save(f"saved_models/discriminator_epoch_{epoch}.keras")
            
            # Plot loss history
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label="Discriminator Loss")
            plt.plot(g_losses, label="Generator Loss")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"generated_images/loss_history_epoch_{epoch}.png")
            plt.close()

    # Save final models
    generator.save("saved_models/final_generator.keras")
    discriminator.save("saved_models/final_discriminator.keras")


if __name__ == "__main__":
    # Let's make some beautiful art
    train(epochs=500, batch_size=128, latent_dim=100, save_interval=50)