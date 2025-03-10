#!/usr/bin/env python3
"""Deep Convolutional Generative Adversarial Network - Hyperparameter Tuning"""

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

class DCGAN():
    """
    Deep Convolutional Generative Adversarial Network with Hyperparameter
    Tuning
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
                 show_generated=False):

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

    def train(self, epochs, batch_size=128, save_interval=50):
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

            # If at a save interval, save generated image samples
            if epoch % save_interval == 0:
                self.save_images(epoch)

        # After training is complete, save the final metrics
        self.save_training_metrics()

    def save_images(self, epoch):
        """Save images for an epoch"""
        r, c = 5, 5  # 5x5 grid of images
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # Rescale images to 0-1 range
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        # Save the figure
        fig.savefig(f"{self.images_dir}/mnist_{epoch}.png")
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
        if self.show_generated:
            plt.show()


def run_experiments():
    """Run multiple hyperparameter experiments"""
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
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "higher_lr",
            "learning_rate": 0.001,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "lower_lr",
            "learning_rate": 0.00005,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "large_batch",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 256,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "small_batch",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 64,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "rmsprop_optimizer",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "rmsprop",
            "discriminator_dropout": 0.3,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "high_dropout",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.5,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 10000,
            "save_interval": 500,
        },
        {
            "name": "low_dropout",
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "optimizer_type": "adam",
            "discriminator_dropout": 0.1,
            "leaky_alpha": 0.2,
            "batch_size": 128,
            "epochs": 10000,
            "save_interval": 500,
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
        )

        # Load and preprocess data
        dcgan.load()
        dcgan.preprocess()

        # Train the model
        dcgan.train(
            epochs=exp["epochs"],
            batch_size=exp["batch_size"],
            save_interval=exp["save_interval"],
        )

        # Generate final images
        dcgan.generate_images()

        print(f"Completed experiment: {exp['name']}")

def analyze_gan_experiments(experiments_data):
    """
    Analyze GAN experiment results to identify optimal configurations and visualize performance.
    
    Parameters:
    -----------
    experiments_data : list of dict
        List of dictionaries containing experiment configurations and results
        
    Returns:
    --------
    dict
        Dictionary containing analysis results including:
        - DataFrame with all experiment data
        - Best experiments by different metrics
        - Summary statistics
        - Correlation analysis
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(experiments_data)
    
    # 1. Summary statistics
    summary_stats = df.describe()
    
    # 2. Find best experiments by different metrics
    best_experiments = {
        'best_discriminator_accuracy': df.loc[df['final_d_acc'].idxmax()]['experiment'],
        'best_discriminator_loss': df.loc[df['final_d_loss'].idxmin()]['experiment'],
        'best_generator_loss': df.loc[df['final_g_loss'].idxmin()]['experiment'],
        'most_stable_discriminator': df.loc[df['d_loss_stability'].idxmin()]['experiment'],
        'most_stable_generator': df.loc[df['g_loss_stability'].idxmin()]['experiment']
    }
    
    # 3. Calculate composite score (optional)
    # Lower is better for losses and stability, higher is better for accuracy
    df['composite_score'] = df['final_d_loss'] + df['final_g_loss'] + \
                           df['d_loss_stability'] + df['g_loss_stability'] - df['final_d_acc']
    best_experiments['best_overall'] = df.loc[df['composite_score'].idxmin()]['experiment']
    
    # 4. Analyze impact of hyperparameters
    hyperparams = ['learning_rate', 'optimizer', 'dropout']
    hyperparameter_impact = {}
    
    for param in hyperparams:
        param_groups = df.groupby(param)
        impact = param_groups[['final_d_loss', 'final_d_acc', 'final_g_loss', 
                               'd_loss_stability', 'g_loss_stability']].mean()
        hyperparameter_impact[param] = impact
    
    # 5. Calculate correlations between metrics
    correlation_matrix = df[['learning_rate', 'dropout', 'final_d_loss', 'final_d_acc', 
                             'final_g_loss', 'd_loss_stability', 'g_loss_stability']].corr()
    
    # 6. Create visualizations (can be displayed or saved)
    def create_visualizations():
        # Set up the matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Experiment comparison (d_loss and g_loss)
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        ax1.bar(x - width/2, df['final_d_loss'], width, label='D Loss')
        ax1.bar(x + width/2, df['final_g_loss']/10, width, label='G Loss (scaled / 10)')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Discriminator vs Generator Loss by Experiment')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['experiment'], rotation=45)
        ax1.legend()
        
        # Plot 2: Discriminator Accuracy
        ax2 = axes[0, 1]
        sns.barplot(x='experiment', y='final_d_acc', data=df, ax=ax2)
        ax2.set_title('Discriminator Accuracy by Experiment')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Stability comparison
        ax3 = axes[1, 0]
        sns.scatterplot(x='d_loss_stability', y='g_loss_stability', 
                         size='final_d_acc', hue='experiment', data=df, ax=ax3)
        ax3.set_title('Training Stability Analysis')
        ax3.set_xlabel('Discriminator Stability (lower is better)')
        ax3.set_ylabel('Generator Stability (lower is better)')
        
        # Plot 4: Hyperparameter impact (e.g., learning rate)
        ax4 = axes[1, 1]
        lr_df = df.copy()
        lr_df['learning_rate'] = lr_df['learning_rate'].astype(str)  # Convert to string for categorical plotting
        sns.boxplot(x='learning_rate', y='final_d_acc', data=lr_df, ax=ax4)
        ax4.set_title('Impact of Learning Rate on Discriminator Accuracy')
        
        plt.tight_layout()
        return fig
    
    # Call the visualization function but don't display yet
    visualization = create_visualizations()
    
    # Return comprehensive analysis results
    results = {
        'data': df,
        'summary_stats': summary_stats,
        'best_experiments': best_experiments,
        'hyperparameter_impact': hyperparameter_impact,
        'correlation_matrix': correlation_matrix,
        'visualization': visualization
    }
    
    return results

if __name__ == "__main__":
    # tensorflow/core/util/port.cc:113] oneDNN custom operations are on. 
    # You may see slightly different numerical results due to floating-point
    # round-off errors from different computation orders. To turn them off,
    # set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Run the hyperparameter experiments
    # run_experiments()

    # Compare the results

    experiments_data = [{'experiment': 'baseline',
            'learning_rate': 0.0002,
            'optimizer': 'adam',
            'dropout': 0.3,
            'final_d_loss': 0.3961116671562195,
            'final_d_acc': 0.80859375,
            'final_g_loss': 2.330787420272827,
            'd_loss_stability': 0.046619000802614566,
            'g_loss_stability': 0.214671605052195
        },
            {'experiment': 'higher_lr',
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'dropout': 0.3,
            'final_d_loss': 0.3737909346818924,
            'final_d_acc': 0.828125,
            'final_g_loss': 6.249948501586914,
            'd_loss_stability': 0.0821166613954061,
            'g_loss_stability': 0.6297349651672374
        },
            {'experiment': 'high_dropout',
            'learning_rate': 0.0002,
            'optimizer': 'adam',
            'dropout': 0.5,
            'final_d_loss': 0.45032499730587006,
            'final_d_acc': 0.8046875,
            'final_g_loss': 1.8981313705444336,
            'd_loss_stability': 0.04398192319260841,
            'g_loss_stability': 0.21318973226550778
        },
            {'experiment': 'large_batch',
            'learning_rate': 0.0002, 
            'optimizer': 'adam', 
            'dropout': 0.3, 
            'final_d_loss': 0.4227825552225113, 
            'final_d_acc': 0.810546875, 
            'final_g_loss': 2.1356353759765625, 
            'd_loss_stability': 0.03631447941819278, 
            'g_loss_stability': 0.18874328597717482
        }, 
            {'experiment': 'lower_lr',
            'learning_rate': 5e-05,
            'optimizer': 'adam',
            'dropout': 0.3,
            'final_d_loss': 0.12198597192764282,
            'final_d_acc': 0.9765625,
            'final_g_loss': 3.2999672889709473,
            'd_loss_stability': 0.03232777145469944,
            'g_loss_stability': 0.6557799915453715
        },
            {'experiment': 'low_dropout',
            'learning_rate': 0.0002,
            'optimizer': 'adam',
            'dropout': 0.1,
            'final_d_loss': 0.30488327145576477,
            'final_d_acc': 0.8671875,
            'final_g_loss': 5.094252109527588,
            'd_loss_stability': 0.05557058330519264,
            'g_loss_stability': 0.5963369682481569
        },
            {'experiment': 'rmsprop_optimizer',
            'learning_rate': 0.0002,
            'optimizer': 'rmsprop',
            'dropout': 0.3,
            'final_d_loss': 0.4395427852869034,
            'final_d_acc': 0.8125,
            'final_g_loss': 4.592486381530762,
            'd_loss_stability': 0.07861955949063126,
            'g_loss_stability': 0.5519023997675302
        },
            {'experiment': 'small_batch',
            'learning_rate': 0.0002,
            'optimizer': 'adam',
            'dropout': 0.3,
            'final_d_loss': 0.19378440082073212,
            'final_d_acc': 0.921875,
            'final_g_loss': 3.7479872703552246,
            'd_loss_stability': 0.06278957872534806,
            'g_loss_stability': 0.42280168206495267
        }]

    results = analyze_gan_experiments(experiments_data)
    # 
    # Print key findings
    print("Best overall experiment:", results['best_experiments']['best_overall'])
    print("Best discriminator accuracy:", results['best_experiments']['best_discriminator_accuracy'])
    print("Most stable generator:", results['best_experiments']['most_stable_generator'])
    # 
    # Display the visualization
    plt.show()
