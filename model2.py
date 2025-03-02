import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
import pickle

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1  # Get the correct vocab size

# Load the preprocessed data
X_train = np.load('Xfin_train.npy')
y_train = np.load('yfin_train.npy')
X_test = np.load('Xfin_test.npy')
y_test = np.load('yfin_test.npy')
# Define the input layer for message
message_input = Input(shape=(100,), dtype='int32', name='message_input')  # Adjust maxlen if needed

# Embedding and LSTM layers for message
message_embedding = Embedding(input_dim=vocab_size, output_dim=128)(message_input)
message_lstm = LSTM(64)(message_embedding)

# Add a dense layer and the output layer
output = Dense(1, activation='sigmoid')(message_lstm)

# Build the model
model = Model(inputs=[message_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

model.fit([X_train], y_train, epochs=5, batch_size=32, validation_data=([X_test], y_test))

# Optionally save the trained model for later use
model.save('finemail_spam_classifier.h5')
