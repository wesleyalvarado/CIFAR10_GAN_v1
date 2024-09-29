# Step 1: Import the required Python libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint  # Import the ModelCheckpoint callback
from keras.layers import LeakyReLU
import re

# Step 2: Load the data
(X, y), (_, _) = keras.datasets.cifar10.load_data()

# Selecting a single class of images
X = X[y.flatten() == 8]  # Choosing class 8

# Step 3: Define parameters to be used in later processes
image_shape = (32, 32, 3)
latent_dimensions = 100

# Step 4: Define a utility function to build the generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(latent_dimensions,)))  # Using Input as the first layer
    model.add(Dense(128 * 8 * 8, activation="relu"))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

# Step 5: Define a utility function to build the discriminator
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=image_shape))  # Using Input as the first layer
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)

# Step 6: Define a utility function to display the generated images
def display_images(epoch, generated_images):
    r, c = 4, 4
    generated_images = 0.5 * generated_images + 0.5  # Scaling the images
    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[count, :, :, :])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig(f"checkpoints/epoch_{epoch}.png")  # Save the image in the checkpoints folder
    # Remove or comment out plt.show() to prevent pausing
    # plt.show()  
    
    # Close the figure to free up memory
    plt.close()

# Step 7: Utility function to find the latest checkpoint
def get_latest_checkpoint(checkpoint_dir, model_name):
    pattern = re.compile(rf"{model_name}_epoch_(\d+)\.weights\.h5")
    max_epoch = -1
    latest_checkpoint = None
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))  # Extract epoch number from the filename
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = os.path.join(checkpoint_dir, filename)
    return latest_checkpoint, max_epoch

# Step 8: Build the GAN
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

discriminator.trainable = False
generator = build_generator()

z = Input(shape=(latent_dimensions,))
image = generator(z)
valid = discriminator(image)

combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Step 9: Load latest checkpoints if available
checkpoint_dir = "checkpoints"  # Define the checkpoint directory
start_epoch = 0  # Initialize the start epoch
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

gen_checkpoint, gen_epoch = get_latest_checkpoint(checkpoint_dir, 'generator')
disc_checkpoint, disc_epoch = get_latest_checkpoint(checkpoint_dir, 'discriminator')

if gen_checkpoint and disc_checkpoint:
    print(f"Loading generator checkpoint: {gen_checkpoint}")
    generator.load_weights(gen_checkpoint)
    print(f"Loading discriminator checkpoint: {disc_checkpoint}")
    discriminator.load_weights(disc_checkpoint)
    start_epoch = min(gen_epoch, disc_epoch) + 1  # Resume from the latest epoch
else:
    print("No checkpoints found. Starting training from scratch.")

# Step 10: Training settings
num_epochs = 15000
batch_size = 32
display_interval = 1000

# Normalize the input
X = (X / 127.5) - 1.0

# Adversarial ground truths with noise
valid = np.ones((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))
fake = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))

# Training loop
for epoch in range(start_epoch, num_epochs):
    # Select a random batch of images
    indices = np.random.randint(0, X.shape[0], batch_size)
    images = X[indices]

    # Generate noise and images
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    generated_images = generator.predict(noise)

    # Train the Discriminator
    discm_loss_real = discriminator.train_on_batch(images, valid)
    discm_loss_fake = discriminator.train_on_batch(generated_images, fake)
    discm_loss = 0.5 * (discm_loss_real[0] + discm_loss_fake[0])

    # Train the Generator
    genr_loss = combined_network.train_on_batch(noise, valid)
    genr_loss_value = genr_loss[0]

    # Track progress and save checkpoints
    if epoch % display_interval == 0:
        print(f"Epoch {epoch}/{num_epochs} [D loss: {discm_loss:.4f}] [G loss: {genr_loss_value:.4f}]")

        display_images(epoch, generated_images)  # Display generated images
        try:
            generator.save_weights(f'{checkpoint_dir}/generator_epoch_{epoch:04d}.weights.h5')
            discriminator.save_weights(f'{checkpoint_dir}/discriminator_epoch_{epoch:04d}.weights.h5')
            print(f"Checkpoint saved for epoch {epoch}")
        except Exception as e:
            print(f"Error saving checkpoints: {e}")
