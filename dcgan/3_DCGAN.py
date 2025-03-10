#!/usr/bin/env python3
"""Deep Convolutional Generative Adversarial Network - Hyperparameter Tuning with W&B"""
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
from tensorflow.keras.optimizers import Adam, RMSprop, SGD  # type: ignore
import tensorflow as tf
import os
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import wandb  # Import Weights & Biases
import io
from PIL import Image

class DCGAN():
    """
    Deep Convolutional Generative Adversarial Network with Hyperparameter
    Tuning and Weights & Biases tracking
    """
    def __init__(self,
                 img_rows=28,
                 img_cols=28,
                 channels=1,
                 latent_dim=100,
                 learning_rate=0.0002,
                 beta1=0.5,
                 optimizer_type='adam',
                 discriminator_dropout=0.3,
                 leaky_alpha=0.2,
                 experiment_name="default",
                 show_generated=False,
                 use_wandb=True):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        # Model parameters
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.show_generated = show_generated
        self.use_wandb = use_wandb
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.optimizer_type = optimizer_type
        self.discriminator_dropout = discriminator_dropout
        self.leaky_alpha = leaky_alpha
        # Training metrics
        self.d_losses = []
        self.d_accuracies = []
        self.g_losses = []
        # Set up experiment directory
        self.experiment_name = experiment_name
        self.experiment_dir = f"experiments/{self.experiment_name}"
        self.images_dir = f"{self.experiment_dir}/images"
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize Weights & Biases
        if self.use_wandb:
            self.init_wandb()
            
        # Save hyperparameters to JSON file
        self.save_hyperparameters()
        # Set up optimizers
        self.optimizer_d = self.get_optimizer()
        self.optimizer_g = self.get_optimizer()
        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer_d,
            metrics=['accuracy']
        )
        # Build the generator
        self.generator = self.build_generator()
        # Generator takes noise as input and generates images
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminator takes generated images as input and determines
        # validity
        validity = self.discriminator(img)
        # Combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer_g
        )


    def init_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        # Configure W&B with experiment settings
        config = {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "optimizer_type": self.optimizer_type,
            "discriminator_dropout": self.discriminator_dropout,
            "leaky_alpha": self.leaky_alpha,
            "latent_dim": self.latent_dim,
            "img_shape": self.img_shape,
            "architecture": "DCGAN"
        }
        
        # Initialize W&B project
        wandb.init(
            project="dcgan-mnist",
            name=self.experiment_name,
            config=config,
            reinit=True
        )
    
    def save_hyperparameters(self):
        """Save hyperparameters to a JSON file"""
        hyperparameters = {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "optimizer_type": self.optimizer_type,
            "discriminator_dropout": self.discriminator_dropout,
            "leaky_alpha": self.leaky_alpha,
            "latent_dim": self.latent_dim,
            "img_shape": self.img_shape,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(f"{self.experiment_dir}/hyperparameters.json", 'w') as f:
            json.dump(hyperparameters, f, indent=4)
            
    def get_optimizer(self):
        """Return the appropriate optimizer based on configuration"""
        if self.optimizer_type.lower() == 'adam':
            return Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        elif self.optimizer_type.lower() == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            return SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            print(f"Unknown optimizer {self.optimizer_type}, defaulting to"
                  "Adam")
            return Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
            
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
        
        # Log dataset information to W&B
        if self.use_wandb:
            wandb.log({
                "dataset": "MNIST",
                "train_samples": len(self.x_train),
                "test_samples": len(self.x_test)
            })
            
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
        
        # Log preprocessing details to W&B
        if self.use_wandb:
            wandb.log({
                "preprocessed_train_samples": len(self.x_train),
                "preprocessed_val_samples": len(self.x_val),
                "preprocessed_test_samples": len(self.x_test)
            })
            
            # Log a few sample images from the training set
            example_images = []
            for i in range(min(10, len(self.x_train))):
                example_images.append(wandb.Image(
                    self.x_train[i].reshape(self.img_rows, self.img_cols),
                    caption=f"Sample {i}"
                ))
            wandb.log({"training_examples": example_images})
            
    def build_generator(self):
        """Build the Generator"""
        x1 = Dense(7 * 7 * 256, use_bias=False, input_dim=self.latent_dim)
        x2 = BatchNormalization()
        x3 = LeakyReLU(alpha=self.leaky_alpha)
        x4 = Reshape((7, 7, 256))
        # upsample -> 14x14
        x5 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',
                             use_bias=False)
        x6 = BatchNormalization()
        x7 = LeakyReLU(alpha=self.leaky_alpha)
        # Upsample -> 28x28
        x8 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',
                             use_bias=False)
        x9 = BatchNormalization()
        x10 = LeakyReLU(alpha=self.leaky_alpha)
        # add output layer
        x11 = Conv2D(self.channels, (5, 5), padding='same', activation='tanh')
        model = Sequential([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)
        
    def build_discriminator(self):
        """Build the Discriminator"""
        # convolutional layer 1
        x1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                    input_shape=self.img_shape)
        x2 = LeakyReLU(alpha=self.leaky_alpha)
        x3 = Dropout(self.discriminator_dropout)
        # convolutional layer 2
        x4 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        x5 = LeakyReLU(alpha=self.leaky_alpha)
        x6 = Dropout(self.discriminator_dropout)
        # flatten
        x7 = Flatten()
        # add output layer
        x8 = Dense(1, activation='sigmoid')
        model = Sequential([x1, x2, x3, x4, x5, x6, x7, x8])
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)
        
    def train(self, epochs, batch_size=128, save_interval=50, log_interval=10):
        """Train the GAN"""
        # adversarial ground truths:
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train the discriminator
            # Select a random batch of images
            idx = np.random.randint(0, self.x_train.shape[0], batch_size)
            imgs = self.x_train[idx]
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Store the metrics
            self.d_losses.append(d_loss[0])
            self.d_accuracies.append(d_loss[1])
            self.g_losses.append(g_loss)
            
            # Print progress
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: "
                  f"{100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
                  
            # Log to W&B
            if self.use_wandb and (epoch % log_interval == 0 or epoch == epochs - 1):
                # Log scalar metrics
                metrics = {
                    "epoch": epoch,
                    "d_loss": d_loss[0],
                    "d_accuracy": d_loss[1],
                    "g_loss": g_loss,
                    "d_loss_real": d_loss_real[0],
                    "d_loss_fake": d_loss_fake[0]
                }
                
                # Log gradients and learning rates
                for i, layer in enumerate(self.discriminator.layers):
                    if hasattr(layer, 'kernel'):
                        weight_name = f"discriminator/layer_{i}_weight_norm"
                        weight_norm = np.sqrt(np.sum(np.square(layer.kernel.numpy())))
                        metrics[weight_name] = weight_norm
                
                for i, layer in enumerate(self.generator.layers):
                    if hasattr(layer, 'kernel'):
                        weight_name = f"generator/layer_{i}_weight_norm"
                        weight_norm = np.sqrt(np.sum(np.square(layer.kernel.numpy())))
                        metrics[weight_name] = weight_norm
                
                wandb.log(metrics)
            
            # If at a save interval, save generated image samples
            if epoch % save_interval == 0:
                self.save_images(epoch)
                
        # After training is complete, save the final metrics
        self.save_training_metrics()
        
        # Save model weights to W&B
        if self.use_wandb:
            self.generator.save(f"{self.experiment_dir}/generator.keras")
            self.discriminator.save(f"{self.experiment_dir}/discriminator.keras")
            
            # Log final metrics
            wandb.log({
                "final_d_loss": self.d_losses[-1],
                "final_d_accuracy": self.d_accuracies[-1],
                "final_g_loss": self.g_losses[-1],
                "d_loss_stability": np.std(self.d_losses[-100:]),
                "g_loss_stability": np.std(self.g_losses[-100:])
            })
            
    def save_images(self, epoch):
        """Save images for an epoch"""
        r, c = 5, 5  # 5x5 grid of images
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images to 0-1 range
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Create figure with subplots
        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
                
        # Save the figure
        fig.savefig(f"{self.images_dir}/mnist_{epoch}.png")
        
        # Log to W&B
        if self.use_wandb:
            # Convert the plot to an image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Log the image grid to W&B
            wandb.log({f"generated_images_epoch_{epoch}": wandb.Image(
                Image.open(buffer), 
                caption=f"Generated images at epoch {epoch}"
            )})
            
            # Also log individual images for the GAN evaluation panel
            gan_examples = []
            for i in range(min(9, len(gen_imgs))):
                gan_examples.append(wandb.Image(
                    gen_imgs[i, :, :, 0],
                    caption=f"Generated {i}"
                ))
            wandb.log({f"gan_samples_epoch_{epoch}": gan_examples})
            
        plt.close()
        
    def save_training_metrics(self):
        """Save training metrics to file and plot them"""
        # Save metrics to JSON file
        metrics = {
            "d_losses": self.d_losses,
            "d_accuracies": self.d_accuracies,
            "g_losses": self.g_losses
        }
        with open(f"{self.experiment_dir}/training_metrics.json", 'w') as f:
            json.dump(metrics, f)
            
        # Plot and save the metrics
        plt.figure(figsize=(10, 8))
        
        # Plot discriminator loss
        plt.subplot(3, 1, 1)
        plt.plot(self.d_losses)
        plt.title('Discriminator Loss')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot discriminator accuracy
        plt.subplot(3, 1, 2)
        plt.plot(self.d_accuracies)
        plt.title('Discriminator Accuracy')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot generator loss
        plt.subplot(3, 1, 3)
        plt.plot(self.g_losses)
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/training_metrics.png")
        
        # Log the plot to W&B
        if self.use_wandb:
            wandb.log({"training_metrics_plot": wandb.Image(
                plt, 
                caption="Training metrics over time"
            )})
            
        plt.close()
        
    def generate_images(self, n_samples=25):
        """Generate and display images using the trained generator"""
        # Generate random noise
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate images from noise
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images to 0-1 range
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Create a grid to display the images
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
        fig.savefig(f"{self.experiment_dir}/mnist_final.png")
        
        # Log final generated images to W&B
        if self.use_wandb:
            wandb.log({"final_generated_images": wandb.Image(
                fig,
                caption="Final generated images"
            )})
            
            # Create an animation of generated images (optional)
            if os.path.exists(self.images_dir) and len(os.listdir(self.images_dir)) > 0:
                # Sort the image files by epoch number
                image_files = sorted([f for f in os.listdir(self.images_dir) if f.startswith('mnist_')], 
                                    key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                # Create a list of images for the animation
                animation_frames = []
                for img_file in image_files:
                    img_path = os.path.join(self.images_dir, img_file)
                    epoch_num = int(img_file.split('_')[1].split('.')[0])
                    animation_frames.append(wandb.Image(
                        img_path,
                        caption=f"Epoch {epoch_num}"
                    ))
                
                # Log the animation
                if animation_frames:
                    wandb.log({"generation_progress": animation_frames})
        
        if self.show_generated:
            pass
            # plt.show()

def run_experiments():
    """Run multiple hyperparameter experiments with W&B sweep integration"""
    # Create directory for experiments
    os.makedirs("experiments", exist_ok=True)
    
    # Define hyperparameter configurations to test
    experiments = [
        {
            "name": "baseline",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "higher_lr",
            "learning_rate": 0.001,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "lower_lr",
            "learning_rate": 0.00005,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "large_batch",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 256,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "small_batch",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 64,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "rmsprop_optimizer",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "rmsprop",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "high_dropout",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.5,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        },
        {
            "name": "low_dropout",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.1,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 500,
            "save_interval": 25,
        }
    ]
    
    # Run each experiment
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Starting experiment: {exp['name']}")
        print(f"{'='*50}")
        
        # Create DCGAN with specified hyperparameters
        dcgan = DCGAN(
            learning_rate=exp["learning_rate"],
            beta1=exp["beta1"],
            optimizer_type=exp["optimizer_type"],
            discriminator_dropout=exp["discriminator_dropout"],
            leaky_alpha=exp["leaky_alpha"],
            experiment_name=exp["name"],
            show_generated=False,
            use_wandb=True  # Enable W&B tracking
        )
        
        # Load and preprocess data
        dcgan.load()
        dcgan.preprocess()
        
        # Train the model
        dcgan.train(
            epochs=exp["epochs"],
            batch_size=exp["batch_size"],
            save_interval=exp["save_interval"],
            log_interval=10  # Log to W&B every 10 epochs
        )
        
        # Generate final images
        dcgan.generate_images()
        
        print(f"Completed experiment: {exp['name']}")
        
        # Finish the W&B run
        wandb.finish()

run_experiments()
