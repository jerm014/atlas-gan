#!/usr/bin/env python3
"""Super Robust GAN - WGAN-GP Implementation"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import time
tf.config.optimizer.set_experimental_options({'remapping': False})


# make dirs for saving pretty stuff
os.makedirs("generated_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


def load_cifar10():
    """ loading dem beautiful training pixels """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    # Resize to power of 2 for more stability
    x_train = (
        x_train.astype("float32") - 127.5
    ) / 127.5  # Normalize to [-1, 1]
    return x_train


def wasserstein_loss(y_true, y_pred):
    """ much better loss function for stability """
    return tf.reduce_mean(y_true * y_pred)


def build_generator(latent_dim=128):
    """ generatoring dem fancy images """
    noise_input = layers.Input(shape=(latent_dim,))

    # Fully connected layer with lots of params
    x = layers.Dense(4 * 4 * 256)(noise_input)
    x = layers.Reshape((4, 4, 256))(x)

    # Block 1: 4x4 -> 8x8
    x = layers.Conv2DTranspose(256,
        kernel_size=5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Block 2: 8x8 -> 16x16
    x = layers.Conv2DTranspose(128,
        kernel_size=5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
 
    # Block 3: 16x16 -> 32x32
    x = layers.Conv2DTranspose(64,
        kernel_size=5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Final conv - no batch norm here.
    x = layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)

    model = models.Model(noise_input, x, name="super_duper_generator")
    print("Generator Summary:")
    model.summary()
    return model


def build_critic(img_shape=(32, 32, 3)):
    """ criticizing dem images (not discriminating anymore!) """
    img_input = layers.Input(shape=img_shape)
    
    # Block 1: No normalization for first layer
    x = layers.Conv2D(64,
        kernel_size=5, strides=2, padding="same")(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)  # Layer norm instead of batch norm!
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 4
    x = layers.Conv2D(512, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Output - no activation! This is WGAN-GP
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    model = models.Model(img_input, x, name="super_duper_critic")
    print("Critic Summary:")
    model.summary()
    return model


def gradient_penalty(critic, real_samples, fake_samples, batch_size):
    """ calculating dem penalties for critic """
    # Get random interpolation between real and fake samples
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Get critic scores for interpolated images
        pred = critic(interpolated, training=True)
        
    # Calculate gradients of critic output w.r.t. interpolated images
    gradients = tape.gradient(pred, interpolated)
    # Compute L2 norm of gradients
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients),
        axis=[1, 2, 3]))
    # Penalize deviation of gradient norm from 1
    gp = tf.reduce_mean(tf.square(gradient_norm - 1.0))
    return gp


def save_generated_images(epoch, G, latent_dim=128, examples=16, set_name="default"):
    """ saving dem beautiful creations """
    noise = np.random.normal(0, 1, (examples, latent_dim))
    gen_images = G.predict(noise, verbose=0)
    gen_images = (gen_images * 127.5 + 127.5).astype(np.uint8)

    # Grid for optimal viewing pleasure
    rows = cols = int(np.sqrt(examples))
    _, axs = plt.subplots(rows, cols, figsize=(rows, cols))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_images[idx])
            axs[i, j].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(f"generated_images/{set_name}/wgan_epoch_{epoch}.png")
    plt.close()


def train_wgan_gp(epochs=300, batch_size=64, latent_dim=128, n_critic=5, 
                  gp_weight=10.0, save_interval=10, learning_rate=0.0001,
                  set_name="default"):
    """ training dem networks with WGAN-GP goodness """

    if not os.path.exists(f"generated_images/{set_name}/"):
        os.makedirs(f"generated_images/{set_name}/")

    if not os.path.exists(f"saved_models/{set_name}/"):
        os.makedirs(f"saved_models/{set_name}/")

    # Load and prepare dem pixels
    x_train = load_cifar10()
    print(f"Training on {x_train.shape[0]} images of shape"
          f" {x_train.shape[1:]}")

    # Build models with fancy WGAN architecture
    generator = build_generator(latent_dim)
    critic = build_critic()

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.0, beta_2=0.9
    )
    critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.0, beta_2=0.9
    )

    # Metrics to track
    gen_losses = []
    critic_losses = []

    # my training loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train critic more times than generator
        # (n_critic times per generator update)
        critic_loss_epoch = []
        gen_loss_epoch = []

        for _ in range(n_critic):
            # Select random batch of real images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            # Generate batch of fake images
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Train critic
            with tf.GradientTape() as tape:
                # Generate fake images
                fake_images = generator(noise, training=True)

                # Get critic outputs
                real_output = critic(real_images, training=True)
                fake_output = critic(fake_images, training=True)

                # WGAN loss (real images should get positive scores, fake
                # negative)
                critic_loss = tf.reduce_mean(fake_output) - \
                    tf.reduce_mean(real_output)

                # Gradient penalty
                gp = gradient_penalty(critic, real_images, fake_images,
                    batch_size)

                # Total critic loss
                total_critic_loss = critic_loss + gp_weight * gp

            # Apply critic gradients
            gradients = tape.gradient(
                total_critic_loss,
                critic.trainable_variables
                )
            critic_optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables)
            )

            critic_loss_epoch.append(float(total_critic_loss))

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = generator(noise, training=True)

            # Get critic scores
            fake_output = critic(fake_images, training=True)

            # Generator wants critic to give high scores to fake images
            gen_loss = -tf.reduce_mean(fake_output)

        # Apply generator gradients
        gradients = tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables)
        )

        gen_loss_epoch.append(float(gen_loss))

        # Calculate average losses for this epoch
        critic_loss_avg = np.mean(critic_loss_epoch)
        gen_loss_avg = np.mean(gen_loss_epoch)

        # Store metrics
        critic_losses.append(critic_loss_avg)
        gen_losses.append(gen_loss_avg)

        # Print progress
        if epoch % 10 == 0:
            time_per_epoch = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} - {time_per_epoch:.1f}s - "
                  f"Critic Loss: {critic_loss_avg:.4f}, "
                  f"Generator Loss: {gen_loss_avg:.4f}, "
                  f"Learning Rate: {learning_rate:.5f}")

        # Save images
        if epoch % save_interval == 0:
            save_generated_images(epoch, generator, latent_dim, set_name=set_name)

            # Also save models occasionally
            if epoch % (save_interval * 5) == 0:
                generator.save(f"saved_models/{set_name}/wgan_generator_epoch_{epoch}.keras")
                critic.save(f"saved_models/{set_name}/wgan_critic_epoch_{epoch}.keras")

            # Plot loss history
            plt.figure(figsize=(10, 5))
            plt.plot(critic_losses, label="Critic Loss")
            plt.plot(gen_losses, label="Generator Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"generated_images/{set_name}/wgan_loss_epoch_{epoch}.png")
            plt.close()

    # Save final models
    generator.save(f"saved_models/{set_name}/wgan_final_generator.keras")
    critic.save(f"saved_models/{set_name}/wgan_final_critic.keras")


if __name__ == "__main__":
    # do some hyperparameters.
    train_wgan_gp(epochs=300, batch_size=64, latent_dim=128, 
        save_interval=10, n_critic=5, gp_weight=10.0, learning_rate=0.0005,
        set_name="5e05_lr")
    train_wgan_gp(epochs=300, batch_size=32, latent_dim=128, 
        save_interval=10, n_critic=5, gp_weight=10.0, learning_rate=0.0001,
        set_name="32_batch")
    train_wgan_gp(epochs=300, batch_size=64, latent_dim=256, 
        save_interval=10, n_critic=5, gp_weight=10.0, learning_rate=0.0001,
        set_name="256_latent-dim")