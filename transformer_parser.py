import json
import numpy as np

def process_json(json_content):
    file_content = ""

    imports = """
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split

"""

    file_content = file_content + imports

    if isinstance(json_content, str):
        config = json.loads(json_content)
    else:
        config = json_content

    components = config['architecture']['components']
    connections = config['architecture']['connections']

    # Extract component parameters
    components_dict = {comp['id']: comp for comp in components}
    
    # Find each component type
    data_component = None
    input_component = None
    encoder_component = None
    attention_component = None
    decoder_component = None
    output_component = None
    
    for comp in components:
        if comp['type'] == 'data':
            data_component = comp
        elif comp['type'] == 'input':
            input_component = comp
        elif comp['type'] == 'encoder':
            encoder_component = comp
        elif comp['type'] == 'attention':
            attention_component = comp
        elif comp['type'] == 'decoder':
            decoder_component = comp
        elif comp['type'] == 'output':
            output_component = comp

    # Determine architecture type
    has_encoder = encoder_component is not None
    has_decoder = decoder_component is not None
    
    if has_encoder and has_decoder:
        arch_type = "encoder-decoder"
    elif has_encoder:
        arch_type = "encoder-only"
    elif has_decoder:
        arch_type = "decoder-only"
    else:
        arch_type = "encoder-decoder"  # fallback

    # Extract parameters
    vocab_size = int(data_component['attributes']['vocabSize'])
    embedding_dim = int(input_component['attributes']['embeddingDim'])
    max_seq_length = int(input_component['attributes']['maxSeqLength'])
    dropout_rate = float(input_component['attributes']['dropout'])
    
    encoder_layers = int(encoder_component['attributes']['numLayers']) if encoder_component else 0
    encoder_heads = int(encoder_component['attributes']['numHeads']) if encoder_component else 0
    encoder_dff = int(encoder_component['attributes']['feedforwardDim']) if encoder_component else 0
    encoder_dropout = float(encoder_component['attributes']['dropout']) if encoder_component else 0.0
    
    decoder_layers = int(decoder_component['attributes']['numLayers']) if decoder_component else 0
    decoder_heads = int(decoder_component['attributes']['numHeads']) if decoder_component else 0
    decoder_dff = int(decoder_component['attributes']['feedforwardDim']) if decoder_component else 0
    decoder_dropout = float(decoder_component['attributes']['dropout']) if decoder_component else 0.0
    
    attention_heads = int(attention_component['attributes']['numHeads']) if attention_component else (encoder_heads or decoder_heads)
    attention_dropout = float(attention_component['attributes']['dropout']) if attention_component else 0.0
    masked = attention_component['attributes']['masked'].lower() == 'true' if attention_component else False
    
    activation = output_component['attributes']['activation']
    temperature = float(output_component['attributes']['temperature'])

    # Add helper functions
    helper_functions = f"""
def build_positional_encoding(max_seq_length, embedding_dim):
    \"\"\"Create positional encoding for transformer\"\"\"
    pos_encoding = np.zeros((max_seq_length, embedding_dim))
    
    for pos in range(max_seq_length):
        for i in range(0, embedding_dim, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
            if i + 1 < embedding_dim:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
    
    return pos_encoding

def build_transformer_encoder_layer(d_model, num_heads, dff, dropout_rate):
    \"\"\"Build a single transformer encoder layer\"\"\"
    inputs = keras.Input(shape=(None, d_model))
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,
        dropout=dropout_rate
    )(inputs, inputs)
    
    # Add & Norm 1
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward Network
    ffn = keras.Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(d_model),
    ])
    ffn_output = ffn(out1)
    
    # Add & Norm 2
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return keras.Model(inputs=inputs, outputs=out2)

def build_transformer_decoder_layer(d_model, num_heads, dff, dropout_rate, use_cross_attention=True):
    \"\"\"Build a single transformer decoder layer\"\"\"
    inputs = keras.Input(shape=(None, d_model))
    if use_cross_attention:
        enc_outputs = keras.Input(shape=(None, d_model))
    
    # Masked Multi-head self-attention
    attention1 = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,
        dropout=dropout_rate
    )(inputs, inputs, use_causal_mask=True)
    
    attention1 = layers.Dropout(dropout_rate)(attention1)
    out1 = layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    
    if use_cross_attention:
        # Multi-head cross-attention
        attention2 = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(out1, enc_outputs)
        
        attention2 = layers.Dropout(dropout_rate)(attention2)
        out2 = layers.LayerNormalization(epsilon=1e-6)(attention2 + out1)
    else:
        out2 = out1
    
    # Feed Forward Network
    ffn = keras.Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(d_model),
    ])
    ffn_output = ffn(out2)
    
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    out3 = layers.LayerNormalization(epsilon=1e-6)(ffn_output + out2)
    
    if use_cross_attention:
        return keras.Model(inputs=[inputs, enc_outputs], outputs=out3)
    else:
        return keras.Model(inputs=inputs, outputs=out3)

print(f"Building {{'{arch_type}'}} transformer model...")

"""

    file_content += helper_functions

    # Build the model based on architecture type
    if arch_type == "encoder-decoder":
        model_building_code = f"""
# Model parameters
VOCAB_SIZE = {vocab_size}
EMBEDDING_DIM = {embedding_dim}
MAX_SEQ_LENGTH = {max_seq_length}
DROPOUT_RATE = {dropout_rate}
ENCODER_LAYERS = {encoder_layers}
ENCODER_HEADS = {encoder_heads}
ENCODER_DFF = {encoder_dff}
DECODER_LAYERS = {decoder_layers}
DECODER_HEADS = {decoder_heads}
DECODER_DFF = {decoder_dff}

# Input layers
encoder_inputs = keras.Input(shape=(MAX_SEQ_LENGTH,), name="encoder_inputs")
decoder_inputs = keras.Input(shape=(MAX_SEQ_LENGTH,), name="decoder_inputs")

# Embedding layers
encoder_embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
decoder_embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)

# Positional encoding
pos_encoding = build_positional_encoding(MAX_SEQ_LENGTH, EMBEDDING_DIM)

# Add positional encoding to embeddings
encoder_embedding += pos_encoding[:MAX_SEQ_LENGTH, :]
decoder_embedding += pos_encoding[:MAX_SEQ_LENGTH, :]

# Apply dropout
encoder_embedding = layers.Dropout(DROPOUT_RATE)(encoder_embedding)
decoder_embedding = layers.Dropout(DROPOUT_RATE)(decoder_embedding)

# Encoder stack
encoder_output = encoder_embedding
for i in range(ENCODER_LAYERS):
    encoder_layer = build_transformer_encoder_layer(
        EMBEDDING_DIM, ENCODER_HEADS, ENCODER_DFF, {encoder_dropout}
    )
    encoder_output = encoder_layer(encoder_output)

# Decoder stack
decoder_output = decoder_embedding
for i in range(DECODER_LAYERS):
    decoder_layer = build_transformer_decoder_layer(
        EMBEDDING_DIM, DECODER_HEADS, DECODER_DFF, {decoder_dropout}, use_cross_attention=True
    )
    decoder_output = decoder_layer([decoder_output, encoder_output])

# Final output layer
outputs = layers.Dense(VOCAB_SIZE, name="outputs")(decoder_output)

model_inputs = [encoder_inputs, decoder_inputs]
"""
    elif arch_type == "encoder-only":
        model_building_code = f"""
# Model parameters
VOCAB_SIZE = {vocab_size}
EMBEDDING_DIM = {embedding_dim}
MAX_SEQ_LENGTH = {max_seq_length}
DROPOUT_RATE = {dropout_rate}
ENCODER_LAYERS = {encoder_layers}
ENCODER_HEADS = {encoder_heads}
ENCODER_DFF = {encoder_dff}

# Input layer
inputs = keras.Input(shape=(MAX_SEQ_LENGTH,), name="inputs")

# Embedding layer
embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)

# Positional encoding
pos_encoding = build_positional_encoding(MAX_SEQ_LENGTH, EMBEDDING_DIM)

# Add positional encoding to embeddings
embedding += pos_encoding[:MAX_SEQ_LENGTH, :]

# Apply dropout
embedding = layers.Dropout(DROPOUT_RATE)(embedding)

# Encoder stack
encoder_output = embedding
for i in range(ENCODER_LAYERS):
    encoder_layer = build_transformer_encoder_layer(
        EMBEDDING_DIM, ENCODER_HEADS, ENCODER_DFF, {encoder_dropout}
    )
    encoder_output = encoder_layer(encoder_output)

# Final output layer
outputs = layers.Dense(VOCAB_SIZE, name="outputs")(encoder_output)

model_inputs = inputs
"""
    else:  # decoder-only
        model_building_code = f"""
# Model parameters
VOCAB_SIZE = {vocab_size}
EMBEDDING_DIM = {embedding_dim}
MAX_SEQ_LENGTH = {max_seq_length}
DROPOUT_RATE = {dropout_rate}
DECODER_LAYERS = {decoder_layers}
DECODER_HEADS = {decoder_heads}
DECODER_DFF = {decoder_dff}

# Input layer
inputs = keras.Input(shape=(MAX_SEQ_LENGTH,), name="inputs")

# Embedding layer
embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)

# Positional encoding
pos_encoding = build_positional_encoding(MAX_SEQ_LENGTH, EMBEDDING_DIM)

# Add positional encoding to embeddings
embedding += pos_encoding[:MAX_SEQ_LENGTH, :]

# Apply dropout
embedding = layers.Dropout(DROPOUT_RATE)(embedding)

# Decoder stack (without cross-attention)
decoder_output = embedding
for i in range(DECODER_LAYERS):
    decoder_layer = build_transformer_decoder_layer(
        EMBEDDING_DIM, DECODER_HEADS, DECODER_DFF, {decoder_dropout}, use_cross_attention=False
    )
    decoder_output = decoder_layer(decoder_output)

# Final output layer
outputs = layers.Dense(VOCAB_SIZE, name="outputs")(decoder_output)

model_inputs = inputs
"""

    file_content += model_building_code

    # Apply output activation
    if activation.lower() == 'softmax':
        file_content += "outputs = layers.Softmax()(outputs)\n"
    elif activation.lower() == 'sigmoid':
        file_content += "outputs = layers.Activation('sigmoid')(outputs)\n"

    file_content += "\n# Create the model\n"
    file_content += f"model = keras.Model(model_inputs, outputs, name='{arch_type.replace('-', '_')}_transformer')\n"

    # Extract training parameters from workflow
    train_params = {}
    analytics_params = {}
    webhook_url = ""
    notification_text = ""
    custom_code = ""
    
    for action in config['workflow']['actions']:
        if action['type'] == 'train':
            attrs = action['attributes']
            train_params = {
                'epochs': int(attrs['epochs']),
                'batchSize': int(attrs['batchSize']),
                'optimizer': attrs['optimizer'],
                'learningRate': float(attrs['learningRate'])
            }
        elif action['type'] == 'analytics':
            attrs = action['attributes']
            analytics_params = {
                'bleu_score': attrs.get('bleuScore', 'false') == 'true',
                'perplexity': attrs.get('perplexity', 'false') == 'true'
            }
        elif action['type'] == 'notify':
            attrs = action['attributes']
            webhook_url = attrs.get('webhookUrl', '')
            notification_text = attrs.get('notificationText', '')
        elif action['type'] == 'custom':
            attrs = action['attributes']
            custom_code = attrs.get('customCode', '')

    # Compile model
    optimizer_code = f"""
optimizer_name = '{train_params.get('optimizer', 'Adam')}'.lower()
if optimizer_name == 'adamw':
    optimizer = keras.optimizers.AdamW(learning_rate={train_params.get('learningRate', 0.001)})
elif optimizer_name == 'sgd':
    optimizer = keras.optimizers.SGD(learning_rate={train_params.get('learningRate', 0.001)})
else:
    optimizer = keras.optimizers.Adam(learning_rate={train_params.get('learningRate', 0.001)})

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits={str(activation.lower() != 'softmax').lower()}),
    metrics=['accuracy']
)

model.summary()

"""

    file_content += optimizer_code

    # Add data loading and preprocessing
    if data_component:
        filepath = data_component['attributes']['filepath']
        if arch_type == "encoder-decoder":
            data_code = f"""
# Load and preprocess text data
print("Loading data from {filepath}...")

import os
import re
from collections import Counter

def load_and_preprocess_text(filepath):
    \"\"\"Load and preprocess text data for transformer training\"\"\"
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def build_vocabulary(sentences, vocab_size):
    \"\"\"Build vocabulary from sentences\"\"\"
    # Tokenize sentences
    all_words = []
    for sentence in sentences:
        words = re.findall(r'\\w+', sentence.lower())
        all_words.extend(words)
    
    # Build vocabulary
    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, count in word_counts.most_common(vocab_size - 4)]
    
    word_to_idx = {{word: idx for idx, word in enumerate(vocab)}}
    idx_to_word = {{idx: word for word, idx in word_to_idx.items()}}
    
    return word_to_idx, idx_to_word

def encode_sentences(sentences, word_to_idx, max_length):
    \"\"\"Convert sentences to sequences of token indices\"\"\"
    encoded = []
    for sentence in sentences:
        words = re.findall(r'\\w+', sentence.lower())
        # Add START token
        sequence = [word_to_idx['<START>']]
        # Add words (or UNK if not in vocab)
        for word in words:
            sequence.append(word_to_idx.get(word, word_to_idx['<UNK>']))
        # Add END token
        sequence.append(word_to_idx['<END>'])
        
        # Pad or truncate to max_length
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        encoded.append(sequence)
    
    return np.array(encoded)

sentences = load_and_preprocess_text('{filepath}')
print(f"Loaded {{len(sentences)}} sentences")

word_to_idx, idx_to_word = build_vocabulary(sentences, VOCAB_SIZE)
actual_vocab_size = len(word_to_idx)
print(f"Built vocabulary of size: {{actual_vocab_size}}")

if actual_vocab_size != VOCAB_SIZE:
    print(f"Updating VOCAB_SIZE from {{VOCAB_SIZE}} to {{actual_vocab_size}}")
    VOCAB_SIZE = actual_vocab_size

encoded_sentences = encode_sentences(sentences, word_to_idx, MAX_SEQ_LENGTH)
print(f"Encoded sentences shape: {{encoded_sentences.shape}}")

encoder_data = encoded_sentences.copy()
decoder_input = np.zeros_like(encoded_sentences)
decoder_target = np.zeros_like(encoded_sentences)

decoder_input[:, :-1] = encoded_sentences[:, :-1]  # Remove last token for input
decoder_target[:, :-1] = encoded_sentences[:, 1:]   # Remove first token for target

from sklearn.model_selection import train_test_split

train_enc, val_enc, train_dec_in, val_dec_in, train_dec_out, val_dec_out = train_test_split(
    encoder_data, decoder_input, decoder_target, 
    test_size=0.2, random_state=42
)

print(f"Training data: {{train_enc.shape[0]}} samples")
print(f"Validation data: {{val_enc.shape[0]}} samples")

# Rename for consistency with the rest of the code
encoder_train = train_enc
decoder_train = train_dec_in
targets_train = train_dec_out
encoder_val = val_enc
decoder_val = val_dec_in
targets_val = val_dec_out

print(f"Training data shape: encoder {{encoder_train.shape}}, decoder {{decoder_train.shape}}, targets {{targets_train.shape}}")
print(f"Validation data shape: encoder {{encoder_val.shape}}, decoder {{decoder_val.shape}}, targets {{targets_val.shape}}")

"""
        else:  # encoder-only or decoder-only
            data_code = f"""
# Load and preprocess text data
print("Loading data from {filepath}...")

import os
import re
from collections import Counter

def load_and_preprocess_text(filepath):
    \"\"\"Load and preprocess text data for transformer training\"\"\"
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def build_vocabulary(sentences, vocab_size):
    \"\"\"Build vocabulary from sentences\"\"\"
    # Tokenize sentences
    all_words = []
    for sentence in sentences:
        words = re.findall(r'\\w+', sentence.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, count in word_counts.most_common(vocab_size - 4)]
    
    word_to_idx = {{word: idx for idx, word in enumerate(vocab)}}
    idx_to_word = {{idx: word for word, idx in word_to_idx.items()}}
    
    return word_to_idx, idx_to_word

def encode_sentences(sentences, word_to_idx, max_length):
    \"\"\"Convert sentences to sequences of token indices\"\"\"
    encoded = []
    for sentence in sentences:
        words = re.findall(r'\\w+', sentence.lower())
        # Add START token
        sequence = [word_to_idx['<START>']]
        # Add words (or UNK if not in vocab)
        for word in words:
            sequence.append(word_to_idx.get(word, word_to_idx['<UNK>']))
        # Add END token
        sequence.append(word_to_idx['<END>'])
        
        # Pad or truncate to max_length
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        encoded.append(sequence)
    
    return np.array(encoded)

# Load and process the data
sentences = load_and_preprocess_text('{filepath}')
print(f"Loaded {{len(sentences)}} sentences")

# Build vocabulary
word_to_idx, idx_to_word = build_vocabulary(sentences, VOCAB_SIZE)
actual_vocab_size = len(word_to_idx)
print(f"Built vocabulary of size: {{actual_vocab_size}}")

if actual_vocab_size != VOCAB_SIZE:
    print(f"Updating VOCAB_SIZE from {{VOCAB_SIZE}} to {{actual_vocab_size}}")
    VOCAB_SIZE = actual_vocab_size

encoded_sentences = encode_sentences(sentences, word_to_idx, MAX_SEQ_LENGTH)
print(f"Encoded sentences shape: {{encoded_sentences.shape}}")

input_data = encoded_sentences.copy()
target_data = np.zeros_like(encoded_sentences)

target_data[:, :-1] = encoded_sentences[:, 1:]   # Remove first token for target

from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    input_data, target_data, 
    test_size=0.2, random_state=42
)

print(f"Training data: {{train_input.shape[0]}} samples")
print(f"Validation data: {{val_input.shape[0]}} samples")

print(f"Training data shape: input {{train_input.shape}}, targets {{train_target.shape}}")
print(f"Validation data shape: input {{val_input.shape}}, targets {{val_target.shape}}")

"""

        file_content += data_code

    # Training code based on architecture type
    if arch_type == "encoder-decoder":
        training_code = f"""
print(f"\\nTraining {arch_type} transformer for {train_params.get('epochs', 10)} epochs with {train_params.get('optimizer', 'Adam')} optimizer...")

# Train the model
history = model.fit(
    [encoder_train, decoder_train], 
    targets_train,
    validation_data=([encoder_val, decoder_val], targets_val),
    epochs={train_params.get('epochs', 10)},
    batch_size={train_params.get('batchSize', 32)},
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate([encoder_val, decoder_val], targets_val, verbose=0)
print(f"\\nValidation Results:")
print(f"Loss: {{val_loss:.4f}}")
print(f"Accuracy: {{val_accuracy:.4f}}")

"""
    else:  # encoder-only or decoder-only
        training_code = f"""
print(f"\\nTraining {arch_type} transformer for {train_params.get('epochs', 10)} epochs with {train_params.get('optimizer', 'Adam')} optimizer...")

# Train the model
history = model.fit(
    train_input, 
    train_target,
    validation_data=(val_input, val_target),
    epochs={train_params.get('epochs', 10)},
    batch_size={train_params.get('batchSize', 32)},
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_input, val_target, verbose=0)
print(f"\\nValidation Results:")
print(f"Loss: {{val_loss:.4f}}")
print(f"Accuracy: {{val_accuracy:.4f}}")

"""

    file_content += training_code

    # Analytics (only if specified in JSON workflow)
    if analytics_params.get('perplexity'):
        file_content += """
# Calculate perplexity
perplexity = np.exp(val_loss)
print(f"Perplexity: {perplexity:.4f}")

"""

    if analytics_params.get('bleu_score'):
        if arch_type == "encoder-decoder":
            file_content += """
# Calculate BLEU score approximation
def calculate_simple_bleu(references, predictions, idx_to_word):
    \"\"\"Simple BLEU score calculation\"\"\"
    def decode_sequence(sequence):
        words = []
        for idx in sequence:
            if idx in idx_to_word:
                word = idx_to_word[idx]
                if word == '<END>':
                    break
                elif word not in ['<PAD>', '<START>', '<UNK>']:
                    words.append(word)
        return words
    
    total_score = 0
    valid_samples = 0
    
    for ref, pred in zip(references[:100], predictions[:100]):  # Sample 100 for speed
        ref_words = set(decode_sequence(ref))
        pred_words = set(decode_sequence(pred))
        
        if len(ref_words) > 0 and len(pred_words) > 0:
            # Simple word overlap score (approximation of BLEU)
            overlap = len(ref_words.intersection(pred_words))
            total_score += overlap / max(len(ref_words), len(pred_words))
            valid_samples += 1
    
    return total_score / valid_samples if valid_samples > 0 else 0.0

# Calculate BLEU score
val_predictions = model.predict([encoder_val, decoder_val], verbose=0)
val_pred_sequences = np.argmax(val_predictions, axis=2)

bleu_score = calculate_simple_bleu(targets_val, val_pred_sequences, idx_to_word)
print(f"BLEU Score: {bleu_score:.4f}")

"""
        else:
            file_content += """
# Calculate BLEU score approximation
def calculate_simple_bleu(references, predictions, idx_to_word):
    \"\"\"Simple BLEU score calculation\"\"\"
    def decode_sequence(sequence):
        words = []
        for idx in sequence:
            if idx in idx_to_word:
                word = idx_to_word[idx]
                if word == '<END>':
                    break
                elif word not in ['<PAD>', '<START>', '<UNK>']:
                    words.append(word)
        return words
    
    total_score = 0
    valid_samples = 0
    
    for ref, pred in zip(references[:100], predictions[:100]):  # Sample 100 for speed
        ref_words = set(decode_sequence(ref))
        pred_words = set(decode_sequence(pred))
        
        if len(ref_words) > 0 and len(pred_words) > 0:
            # Simple word overlap score (approximation of BLEU)
            overlap = len(ref_words.intersection(pred_words))
            total_score += overlap / max(len(ref_words), len(pred_words))
            valid_samples += 1
    
    return total_score / valid_samples if valid_samples > 0 else 0.0

# Calculate BLEU score
val_predictions = model.predict(val_input, verbose=0)
val_pred_sequences = np.argmax(val_predictions, axis=2)

bleu_score = calculate_simple_bleu(val_target, val_pred_sequences, idx_to_word)
print(f"BLEU Score: {bleu_score:.4f}")

"""

    # Custom code execution
    if custom_code:
        file_content += f"""
# Custom Code Execution
try:
    {custom_code}
    print("Custom code executed successfully")
except Exception as e:
    print(f"Error in custom code: {{e}}")

"""

    # Webhook notification
    if webhook_url and notification_text:
        model_name = config.get('modelName', 'Transformer Model')
        file_content += f"""
# Webhook Notification
try:
    payload = {{
        "content": f"{notification_text.replace('{model_name}', '{{model_name}}')} - Validation Accuracy: {{val_accuracy:.4f}}, Loss: {{val_loss:.4f}}"
    }}
    response = requests.post("{webhook_url}", json=payload)
    if response.status_code == 204:
        print("Notification sent successfully")
    else:
        print(f"Failed to send notification: {{response.status_code}}")
except Exception as e:
    print(f"Error sending notification: {{e}}")

"""

    # Generate text only if specified in workflow
    generate_text_action = next(
        (action for action in config['workflow']['actions'] 
         if action['type'] == 'generate-text'), 
        None
    )
    
    if generate_text_action:
        prompt = generate_text_action['attributes'].get('prompt', 'Once upon a time')
        max_tokens = int(generate_text_action['attributes'].get('maxTokens', 100))
        
        if arch_type == "encoder-decoder":
            file_content += f"""
# Text generation (as specified in workflow)
def decode_sequence(sequence, idx_to_word):
    \"\"\"Convert sequence of indices back to text\"\"\"
    words = []
    for idx in sequence:
        if idx in idx_to_word:
            word = idx_to_word[idx]
            if word == '<END>':
                break
            elif word not in ['<PAD>', '<START>']:
                words.append(word)
    return ' '.join(words)

def generate_text(model, prompt_text, word_to_idx, idx_to_word, max_length):
    \"\"\"Generate text using the trained transformer\"\"\"
    # Encode the prompt
    words = re.findall(r'\\w+', prompt_text.lower())
    encoder_input = np.zeros((1, max_length))
    encoder_input[0, 0] = word_to_idx['<START>']
    
    for i, word in enumerate(words[:max_length-2]):
        encoder_input[0, i+1] = word_to_idx.get(word, word_to_idx['<UNK>'])
    encoder_input[0, min(len(words)+1, max_length-1)] = word_to_idx['<END>']
    
    # Start decoder with START token
    decoder_input = np.zeros((1, max_length))
    decoder_input[0, 0] = word_to_idx['<START>']
    
    for i in range(1, max_length):
        # Predict next token
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        predicted_id = np.argmax(predictions[0, i-1, :])
        
        # Add predicted token to decoder input
        decoder_input[0, i] = predicted_id
        
        # Stop if END token is predicted
        if predicted_id == word_to_idx['<END>']:
            break
    
    return decoder_input[0]

print("\\nGenerating text...")
prompt = "{prompt}"
generated_sequence = generate_text(model, prompt, word_to_idx, idx_to_word, min({max_tokens}, MAX_SEQ_LENGTH))
generated_text = decode_sequence(generated_sequence, idx_to_word)
print(f"Prompt: {{prompt}}")
print(f"Generated: {{generated_text}}")

"""
        else:  # encoder-only or decoder-only
            file_content += f"""
def decode_sequence(sequence, idx_to_word):
    \"\"\"Convert sequence of indices back to text\"\"\"
    words = []
    for idx in sequence:
        if idx in idx_to_word:
            word = idx_to_word[idx]
            if word == '<END>':
                break
            elif word not in ['<PAD>', '<START>']:
                words.append(word)
    return ' '.join(words)

def generate_text(model, prompt_text, word_to_idx, idx_to_word, max_length):
    \"\"\"Generate text using the trained transformer\"\"\"
    # Encode the prompt
    words = re.findall(r'\\w+', prompt_text.lower())
    input_seq = np.zeros((1, max_length))
    input_seq[0, 0] = word_to_idx['<START>']
    
    # Add prompt words
    for i, word in enumerate(words[:max_length-2]):
        input_seq[0, i+1] = word_to_idx.get(word, word_to_idx['<UNK>'])
    
    prompt_len = min(len(words) + 1, max_length - 1)
    
    # Generate tokens one by one
    for i in range(prompt_len, max_length):
        # Predict next token
        predictions = model.predict(input_seq, verbose=0)
        predicted_id = np.argmax(predictions[0, i-1, :])
        
        # Add predicted token to input
        input_seq[0, i] = predicted_id
        
        # Stop if END token is predicted
        if predicted_id == word_to_idx['<END>']:
            break
    
    return input_seq[0]

print("\\nGenerating text...")
prompt = "{prompt}"
generated_sequence = generate_text(model, prompt, word_to_idx, idx_to_word, min({max_tokens}, MAX_SEQ_LENGTH))
generated_text = decode_sequence(generated_sequence, idx_to_word)
print(f"Prompt: {{prompt}}")
print(f"Generated: {{generated_text}}")

"""

    # Final completion message
    file_content += f"""
print(f"\\nTransformer model '{config.get('modelName', 'MyModel')}' processing completed!")
"""

    return file_content
