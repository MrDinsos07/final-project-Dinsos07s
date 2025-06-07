import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import os
import json
import sys
import numpy as np


class TrainingProgressLogger(Callback):
    def __init__(self, log_file='training_log.json'):
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        # Write training progress to a log file after each epoch
        with open(self.log_file, 'w') as f:
            json.dump({
                'epoch': epoch + 1,
                'accuracy': logs.get('accuracy'),
                'val_accuracy': logs.get('val_accuracy'),
                'loss': logs.get('loss'),
                'val_loss': logs.get('val_loss')
            }, f)


def train_model(dataset_path, epochs, log_file='training_log.json'):
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16

    # Get number of classes from the directory structure
    NUM_CLASSES = len(next(os.walk(dataset_path))[1])

    # Define the data generator with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Data generators for training and validation
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Load the base MobileNetV2 model without the top layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model layers

    # Define the model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for logging and early stopping
    callbacks = [
        TrainingProgressLogger(log_file=log_file),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    # Start training the model
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Save the trained model in TensorFlow's .keras format
    model.save("model.keras")

    # Save the class labels (used for inference) to label.npy
    class_names = list(train_generator.class_indices.keys())
    np.save("label.npy", np.array(class_names))

    print("Training complete. Model saved as 'model.keras' and labels saved as 'label.npy'.")


if __name__ == '__main__':
    # Get dataset path and number of epochs from command-line arguments
    dataset_path = sys.argv[1]
    epochs = int(sys.argv[2])
    log_file = sys.argv[3] if len(sys.argv) > 3 else 'training_log.json'

    # Train the model
    train_model(dataset_path, epochs, log_file)
