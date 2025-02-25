import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Disable GPU to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set latent dimension (size of random input to generator)
latent_dim = 100

# Build the Generator
def build_generator():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Build the Discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

# Build the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator for GAN training
    gan_input = keras.Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    gan = keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
    return gan

# Function to train the GAN
def train_gan(generator, discriminator, gan, epochs=5000, batch_size=64):
    # Load MNIST dataset
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)

    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train the Discriminator
        real_images = x_train[np.random.randint(0, x_train.shape[0], half_batch)]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))  # Fool the discriminator

        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
            generate_and_save_images(generator, epoch)

# Function to generate and save images
def generate_and_save_images(generator, epoch):
    noise = np.random.normal(0, 1, (16, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0,1]

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.show()

# Initialize models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, epochs=5000, batch_size=64)
