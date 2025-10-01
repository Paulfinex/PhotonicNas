import larq as lq
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, losses
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import numpy as np

# gpu mem growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# arg parser
parser = argparse.ArgumentParser(description="Larq quant training")
parser.add_argument("--n_bits", type=int, default=6, help="Number of bits for quantization (default 6)")
parser.add_argument("--load_weights", type=bool, default=False, help="Boolean to load model weights (default False)")
args = parser.parse_args()

n_bits = args.n_bits
load_saved_model = args.load_weights

# quantization 
weight_quantizer = lq.quantizers.DoReFa(k_bit=n_bits, mode='weights')
activation_quantizer = lq.quantizers.DoReFa(k_bit=n_bits)

weight_quantizer_init = lq.quantizers.DoReFa(k_bit=6, mode='weights')
activation_quantizer_init = lq.quantizers.DoReFa(k_bit=6)

# fare 2 kwargs , per primo layer 6 bit e tutto 2 e 6 - 4 
kwargs_init = dict(
    kernel_quantizer=weight_quantizer_init,
    kernel_constraint="weight_clip",
    input_quantizer=activation_quantizer_init
)
kwargs = dict(
    kernel_quantizer=weight_quantizer,
    kernel_constraint="weight_clip",
    input_quantizer=activation_quantizer
)

if n_bits !=2:
    kwargs = dict(
        kernel_quantizer=weight_quantizer,
        kernel_constraint="weight_clip",
        #kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        input_quantizer=activation_quantizer
    )


# best acc save
class SaveBestAccuracyCallback(callbacks.Callback):
    def __init__(self, filepath, model_filepath):
        super().__init__()
        self.filepath = filepath
        self.model_filepath = model_filepath
        self.best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy", 0)
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            with open(self.filepath, "w") as f:
                f.write(f"Best Validation Accuracy: {self.best_accuracy:.4f}\n")
            self.model.save(self.model_filepath)
            print(f"\n_______________________________________________________________")
            print(f"Best accuracy updated to: {self.best_accuracy:.4f}\n")
            print(f"Model saved to: {self.model_filepath}\n")
            print(f"_______________________________________________________________")

# model
def create_larq_model():
    model = models.Sequential()

    model.add(lq.layers.QuantConv2D(8, (3, 3),
                                    kernel_quantizer=weight_quantizer_init,
                                    kernel_constraint="weight_clip",
                                    input_quantizer=activation_quantizer_init,
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                    use_bias=False,
                                    input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantConv2D(32, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantConv2D(16, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2, padding='valid'))
    
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(lq.layers.QuantConv2D(64, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(lq.layers.QuantConv2D(64, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())    
    
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2, padding='valid'))
    
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(lq.layers.QuantConv2D(16, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantConv2D(16, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantConv2D(22, (3, 3), padding='same', use_bias=False, **kwargs))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=1, padding='valid'))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2, padding='valid'))

    model.add(layers.GlobalAveragePooling2D())
    
    model.add(lq.layers.QuantDense(198 , use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())

    model.add(lq.layers.QuantDense(32, use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantDense(32, use_bias=False, **kwargs))
    model.add(layers.BatchNormalization(scale=False))
    model.add(layers.ReLU())
    
    model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
    model.add(layers.Activation("softmax"))

    return model

# cifar10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Data aug
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

#Train
num_runs = 10
validation_accuracies = []

bat_size = 128
lr = 1e-3
if n_bits == 2:
    bat_size = 32
    lr= 5e-3
    
# Define learning rate schedule with warmup
def warmup_cosine_decay(epoch, lr):
    warmup_epochs = 15
    max_lr = 1e-2  # Peak LR
    min_lr = 5e-4  # Final min LR

    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lr  # Linear warmup
    else:
        cosine_decay = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - warmup_epochs) / (200 - warmup_epochs) * np.pi))
        return cosine_decay

# Keras Learning Rate Scheduler Callback
scheduler = callbacks.LearningRateScheduler(warmup_cosine_decay)


print(f"\n_______________________________________________________________")
print(f"Running with quantization bits {n_bits} ")
for run in range(num_runs):
    print(f"\n_______________________________________________________________")
    model = create_larq_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    best_acc_filepath = f"{n_bits}bits_run_{run + 1}_best_accuracy.txt"
    best_model_filepath = f"{n_bits}bits_run_{run + 1}_best_model.h5"
    callbacks_list = [
        callbacks.EarlyStopping(patience=30, restore_best_weights=True),
        SaveBestAccuracyCallback(filepath=best_acc_filepath, model_filepath = best_model_filepath)
    ]
    if n_bits == 2:
        callbacks_list = [
            callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            SaveBestAccuracyCallback(filepath=best_acc_filepath, model_filepath = best_model_filepath),
            scheduler
        ]
    print(f"Run {run + 1}/{num_runs}...")
    train_run = model.fit(
        datagen.flow(x_train, y_train, batch_size=bat_size),
        validation_data=(x_test, y_test),
        epochs=200,
        callbacks=callbacks_list,
        verbose=1
    )
    best_val_acc = max(train_run.history["val_accuracy"])
    validation_accuracies.append(best_val_acc)
    print(f"Run {run + 1} completed with best val accuracy: {best_val_acc:.4f}")

average_accuracy = np.mean(validation_accuracies)
avg_acc_filepath = f"avg_{n_bits}_accuracy.txt"
with open(avg_acc_filepath, "w") as f:
    f.write(f"Accuracy: {average_accuracy:.4f}\n")
print(f"Val accuracies: {validation_accuracies}")