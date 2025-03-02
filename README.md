# Spam Classification using LSTM

## Overview
This project implements a spam classification model using an LSTM neural network. The model is trained to differentiate between spam and non-spam (ham) emails based on text content. The project includes data preprocessing, model training, and a prediction script.

## Features
- Preprocessing of email text (cleaning and tokenization)
- Word embedding using GloVe vectors
- LSTM-based deep learning model for classification
- Model saving and loading for future predictions
- Script for predicting new email messages

## Installation
### Prerequisites
Ensure you have Python 3.7+ installed along with the necessary dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn beautifulsoup4
```

## Dataset Preparation
1. **Dataset Format**:
   - The dataset should be a CSV file (`cleaned_dataset_fin.csv`) with the following columns:
     - `cleaned_text`: The preprocessed email text
     - `labels`: "spam" or "ham" (1 for spam, 0 for ham)

2. **Preprocessing Steps**:
   - Remove HTML tags
   - Convert text to lowercase
   - Remove non-alphabetic characters
   - Tokenize and pad sequences
   - Convert labels to binary (1 for spam, 0 for ham)

## Training the Model
Run the following script to preprocess data and train the model:

```bash
python train.py
```

This script:
- Loads and preprocesses the dataset
- Initializes a tokenizer and saves it (`tokenizer.pkl`)
- Loads pre-trained GloVe embeddings
- Builds and trains an LSTM model
- Saves the trained model as `finemail_spam_classifier.h5`

## Making Predictions
To classify a new email message, use the following script:

```bash
python predict.py
```

### **Example Email Prediction Script (predict.py)**
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

def clean_text(text):
    return text.lower().strip()

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model('finemail_spam_classifier.h5')

# Sample email
new_message = "Congratulations! You won a free lottery. Click here to claim."
cleaned_text = clean_text(new_message)

# Tokenize and pad
maxlen = 100  # Must match training
sequence = tokenizer.texts_to_sequences([cleaned_text])
padded = pad_sequences(sequence, maxlen=maxlen, padding='post')

# Predict
prediction = model.predict(padded)
predicted_label = 'spam' if prediction[0] > 0.5 else 'not spam'
print(f"Predicted Label: {predicted_label}")
```

## Model Details
- **Embedding Layer**: Uses GloVe pre-trained word vectors
- **LSTM Layer**: 64 hidden units
- **Dense Layer**: Single neuron with sigmoid activation
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam

## File Structure
```
├── cleaned_dataset_fin.csv  # Preprocessed dataset
├── tokenizer.pkl            # Saved tokenizer object
├── Xfin_train.npy           # Preprocessed training data
├── yfin_train.npy           # Training labels
├── Xfin_test.npy            # Preprocessed test data
├── yfin_test.npy            # Test labels
├── glove.6B.100d.txt        # GloVe embeddings
├── train.py                 # Training script
├── predict.py               # Prediction script
├── finemail_spam_classifier.h5  # Trained model
├── README.md                # Project documentation
```

## Notes
- Ensure `glove.6B.100d.txt` is downloaded and placed in the project directory.
- The `maxlen` parameter in `pad_sequences` should be the same during training and prediction.
- Adjust batch size and epochs in `train.py` based on available computational resources.

## License
This project is licensed under the soufiane fiha 7ebse .

