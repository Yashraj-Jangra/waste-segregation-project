import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# --- 1. Define Constants & Paths ---
DATASET_PATH = 'dataset-resized'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 6 # cardboard, glass, metal, paper, plastic, trash

# --- 2. Prepare Data Generators ---
# Create an instance of the ImageDataGenerator with rescaling and validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# Training data generator
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Set as training data
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Set as validation data
)

print(f"Found {train_generator.samples} images for training.")
print(f"Found {validation_generator.samples} images for validation.")
print(f"Class indices: {train_generator.class_indices}")

# --- 3. Build the Model (using Transfer Learning with MobileNetV2) ---
# Load the base MobileNetV2 model, pre-trained on ImageNet, without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x) # Add a fully-connected layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Add a final softmax layer for classification

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

print("\n--- Script to setup dataset and build model is ready. ---")
print("Next step will be to train this model (Phase 2).")
