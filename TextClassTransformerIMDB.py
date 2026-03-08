import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import time  # Added for timing

# Global variable
max_length_value = 1024 # Fixed amount of characters to use

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit memory use to 20.48 out of 24GB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)]  # 20.48 GB
            )

            # Float32 used for better accuracy
            keras.mixed_precision.set_global_policy('float32')

            print(f"GPU configured: {gpus[0]}")

        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, stopping program...")
        exit()


# Load and prepare the dataset
def load_imdb_data():
    df = pd.read_csv('IMDB Dataset.csv')

    # Convert sentiments to binary labels
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    texts = df['review'].values
    labels = df['sentiment'].values.astype(np.float32)

    return texts, labels


# Character-level tokenization
def create_char_encoder():
    # Create character vocabulary
    chars = [chr(i) for i in range(32, 127)]  # Printable ASCII characters
    chars.extend(['<PAD>', '<UNK>', '<START>', '<END>'])  # Special tokens

    char_to_id = {char: i for i, char in enumerate(chars)}
    id_to_char = {i: char for i, char in enumerate(chars)}

    return char_to_id, id_to_char, len(chars)


# Preprocess text to character sequences
def text_to_char_sequence(text, char_to_id, max_length=max_length_value):
    text = text.lower()[:max_length - 2]  # Reserve space for start/end special tokens

    # Add start and end tokens
    sequence = [char_to_id['<START>']]
    sequence.extend([char_to_id.get(char, char_to_id['<UNK>']) for char in text])
    sequence.append(char_to_id['<END>'])

    # Pad or shorten to max_length
    if len(sequence) < max_length:
        sequence = sequence + [char_to_id['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
        sequence[-1] = char_to_id['<END>']  # Ensure end token at the end

    return sequence


# Batch preprocessing the dataset
def preprocess_dataset(texts, labels, char_to_id, max_length=max_length_value, batch_size=32):
    char_sequences = [text_to_char_sequence(text, char_to_id, max_length) for text in texts]

    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        np.array(char_sequences, dtype=np.int32),
        np.array(labels, dtype=np.float32)
    ))

    # Optimize dataset performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()  # Cache the preprocessed data

    return dataset


# Transformer components
def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]  # (seq, 1)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth  # (1, depth)

    angle_rates = 1.0 / tf.pow(10000.0, depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_length=max_length_value):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_length, d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # Scale embeddings and add positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.05):
        super().__init__()

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-head attention with residual connection
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

def build_model(vocab_size, max_length=max_length_value):
    d_model = 256
    num_heads = 6  # Attention heads
    dff = 128  # Feed-forward dimension
    dropout_rate = 0.05

    # Input layer
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_layer')

    # Character embeddings with positional encoding
    embedding = PositionalEmbedding(vocab_size, d_model, max_length)(inputs)

    # Transformer blocks
    x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(embedding)
    x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x)

    # Global average pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu')(x)  # Larger dense layer
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output_layer')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# GPU usage monitoring
class GPUMonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if tf.config.list_physical_devices('GPU'):
            gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU Memory - Current: {gpu_stats['current'] / 1e9:.2f}GB, "
                  f"Peak: {gpu_stats['peak'] / 1e9:.2f}GB")


# Function to evaluate model on test data
def evaluate_on_test_data(model, char_to_id, batch_size=32):
    print("\nLoading test data...")
    texts, labels = load_imdb_data()
    
    # Use last 25k rows as test data
    X_test = texts[25000:]
    y_test = labels[25000:]
    
    print(f"Test data size: {len(X_test)} samples")
    
    # Create test dataset
    test_dataset = preprocess_dataset(X_test, y_test, char_to_id, max_length_value, batch_size)
    
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    
    return test_loss, test_accuracy


# Format time function
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


# Main training script
def main():
    # Start timing the entire process
    total_start_time = time.time()
    training_time = 0

    configure_gpu()

    print("Loading IMDB dataset...")
    texts, labels = load_imdb_data()

    print("Creating character encoder...")
    char_to_id, id_to_char, vocab_size = create_char_encoder()
    print(f"Vocabulary size: {vocab_size}")

    batch_size = 32
    max_length = max_length_value  # Set as a global variable

    print("Arranging the dataset...")
    # Use first 20k for training, next 5k for validation, exclude last 25k for test
    X_train = texts[:20000]
    y_train = labels[:20000]
    
    X_val = texts[20000:25000]
    y_val = labels[20000:25000]
    
    # The last 25k (25000:50000) are excluded for later test use

    # Create TensorFlow datasets
    train_dataset = preprocess_dataset(X_train, y_train, char_to_id, max_length, batch_size)
    val_dataset = preprocess_dataset(X_val, y_val, char_to_id, max_length, batch_size)

    print("Building model...")
    model = build_model(vocab_size, max_length)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks for training
    callbacks = [
        GPUMonitorCallback(),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_gpu_transformer.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    print("Starting training...")
    training_start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - training_start_time

    print("Evaluating model...")
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    print(f"Final Validation accuracy: {val_acc:.4f}")

    # Save the final model
    model.save('gpu_char_transformer_imdb.keras')
    print("Final model saved!")

    # Evaluate on test data (last 25k rows)
    test_loss, test_accuracy = evaluate_on_test_data(model, char_to_id, batch_size)
    
    total_time = time.time() - total_start_time

    # Print final results with timing information
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"Test Loss:           {test_loss:.4f}")
    print(f"{'-'*60}")
    print(f"Training Time:       {format_time(training_time)}")
    print(f"Total Time:          {format_time(total_time)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Verify GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()
