import larq as lq
import tensorflow as tf
from tensorflow.keras import optimizers, losses, callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import numpy as np
import time
import datetime
import sys

# Enable GPU memory growth (prevents out-of-memory errors)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Argument parser
parser = argparse.ArgumentParser(description="Larq quant training on GTSDB")
parser.add_argument("--n_bits", type=int, default=6, help="Number of bits for quantization (default: 6)")
parser.add_argument("--n_bits_1st", type=int, default=6, help="Number of bits for quantization (default: 6)")
parser.add_argument("--load_weights", type=bool, default=6, help="Load model weights (default: False)")

args = parser.parse_args()
num_classes= 43
n_bits = args.n_bits
n_bits_1st = args.n_bits_1st
load_saved_model = args.load_weights

# Quantization settings
weight_quantizer = lq.quantizers.DoReFa(k_bit=n_bits, mode="weights")
activation_quantizer = lq.quantizers.DoReFa(k_bit=n_bits)

weight_quantizer_init = lq.quantizers.DoReFa(k_bit=n_bits_1st, mode="weights")
activation_quantizer_init = lq.quantizers.DoReFa(k_bit=n_bits_1st)

kwargs_init = dict(
    kernel_quantizer=weight_quantizer_init,
    kernel_constraint="weight_clip",
    input_quantizer=activation_quantizer_init,
)
kwargs = dict(
    kernel_quantizer=weight_quantizer,
    kernel_constraint="weight_clip",
    input_quantizer=activation_quantizer,
)

if n_bits != 2:
    kwargs["kernel_regularizer"] = tf.keras.regularizers.l2(1e-5)

# ✅ **1. Load GTSDB Dataset**
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset path
train_dir = os.path.join(os.getcwd(), "GTSDB", "Train")

# Load all image paths and labels
image_paths = []
labels = []

for class_name in sorted(os.listdir(train_dir)):  # Ensure sorted order
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_name)  # Class name as label

# Convert to DataFrame
df = pd.DataFrame({"filename": image_paths, "label": labels})

# Ensure labels are strings for consistency
df["label"] = df["label"].astype(str)

# ✅ **Duplicate classes with ≤2 instances to have at least 4 images**
def duplicate_small_classes(df):
    small_classes = df["label"].value_counts()[df["label"].value_counts() <= 2].index

    augmented_rows = []
    for cls in small_classes:
        class_samples = df[df["label"] == cls]
        first_sample = class_samples.iloc[0]  # Get first image of the class

        # Duplicate it 3 times
        for _ in range(3):
            augmented_rows.append(first_sample.copy())

    if augmented_rows:
        df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    return df

df = duplicate_small_classes(df)

# ✅ **Train-Test Split (Ensuring Every Class Appears in Both)**
# Hold out at least one sample per class for the test set
test_samples = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), 1), random_state=42))

# Remove those samples from the main dataset to ensure no duplicates
train_df = df.drop(test_samples.index)

# Perform normal stratified split on remaining data
train_rest, test_rest = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)

# Combine forced test samples with stratified test split
test_df = pd.concat([test_samples, test_rest])
train_df = train_rest  # Keep the rest as train data

# ✅ Print final class distributions
print("Train class distribution:", train_df["label"].value_counts().to_dict())
print("Test class distribution:", test_df["label"].value_counts().to_dict())

# ✅ **Image Augmentation**
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


# ✅ **Create ImageDataGenerator for Train & Test**
batch_size = 64
img_size = (32, 32)
# ✅ **Image Augmentation for Training Data**
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    workers=10,  # Increase this
    use_multiprocessing=True 
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    workers=10,  # Increase this
    use_multiprocessing=True 
)

print(f"✅ Train Samples: {train_generator.samples}")
print(f"✅ Test Samples: {test_generator.samples}")


# ✅ **2. Create Larq Model**
def create_larq_model():
    model = models.Sequential()

    model.add(lq.layers.QuantConv2D(8, (3, 3), use_bias=False, input_shape=(32, 32, 3), **kwargs_init))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(16, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding="valid"))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(32, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding="valid"))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(lq.layers.QuantConv2D(64, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(64, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding="valid"))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(lq.layers.QuantConv2D(16, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(16, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding="valid"))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(22, (3, 3), padding="same", use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(layers.GlobalAveragePooling2D())
    
    model.add(lq.layers.QuantDense(32, use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantDense(32, use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantDense(32, use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantDense(num_classes, use_bias=False, **kwargs))  # ✅ 43 classes
    model.add(layers.Activation("softmax"))

    return model

# ✅ **3. Train the Model**
bat_size = 64
lr = 1e-5
if n_bits == 2:
    bat_size = 32
    lr = 5e-3

# Learning rate scheduler
def warmup_cosine_decay(epoch, lr):
    warmup_epochs = 50
    max_lr = 1e-2
    min_lr = 5e-5

    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lr
    else:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - warmup_epochs) / (1000 - warmup_epochs) * np.pi))

scheduler = callbacks.LearningRateScheduler(warmup_cosine_decay)

# Custom callback to save best accuracy

class SaveBestAccuracyCallback(callbacks.Callback):
    def __init__(self, n_bits_1st, n_bits, total_epochs, run_number):
        super().__init__()
        self.n_bits_1st = n_bits_1st
        self.n_bits = n_bits
        self.best_accuracy = 0.0
        self.total_epochs = total_epochs
        self.run_number = run_number
        self.start_time = None
        self.filepath = f"{n_bits_1st}-{n_bits}_accuracies.txt"

    def on_train_begin(self, logs=None):
        self.start_time = time.time()  # Record start time when training begins

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy", 0)
        
        # Update best accuracy if improved
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
        
        # Estimated time remaining
        elapsed_time = time.time() - self.start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (self.total_epochs - (epoch + 1))
        eta = datetime.timedelta(seconds=int(remaining_time))

        # Clear console and print updated progress
        sys.stdout.write("\033c")  # Clear console output
        sys.stdout.flush()
        print(f"Current configuration: {self.n_bits_1st}-{self.n_bits}")
        print(f"Run: {self.run_number}")
        print(f"Epoch: {epoch + 1}/{self.total_epochs}")
        print(f"Best Validation Accuracy: {self.best_accuracy:.4f}")
        print(f"Estimated Time Remaining: {eta}")

    def on_train_end(self, logs=None):
        # Save best accuracy at the end of the run
        with open(self.filepath, "a") as f:
            f.write(f"Run {self.run_number}: Best Accuracy: {self.best_accuracy:.4f}\n")

# Training loop
validation_accuracies = []
total_epochs = 1000  # Set this based on your training plan
num_runs = 10
for run in range(num_runs):

    model = create_larq_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    best_acc_filepath = f"{n_bits}bits_run_{run + 1}_best_accuracy.txt"
    best_model_filepath = f"{n_bits}bits_run_{run + 1}_best_model.h5"

    callbacks_list = [
        SaveBestAccuracyCallback(
            filepath=best_acc_filepath,
            model_filepath=best_model_filepath,
            total_epochs=total_epochs,
            run_number=run + 1
        ),
        scheduler
    ]

    train_run = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=total_epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    best_val_acc = max(train_run.history["val_accuracy"])
    validation_accuracies.append(best_val_acc)
    print(f"Run {run + 1} completed with best val accuracy: {best_val_acc:.4f}")


average_accuracy = np.mean(validation_accuracies)
avg_acc_filepath = f"avg_{n_bits}_accuracy.txt"
print(f"Val accuracies: {validation_accuracies}")
print("\a") 