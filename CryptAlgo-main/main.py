import binascii
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import math
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import hashlib
import base64
import rsa

# Step 1: Generate Ciphertext Samples
def generate_ciphertext_samples(num_samples=100):
    """Generate ciphertext samples for AES, DES, Blowfish, RSA, SHA-256, MD5, and Caesar Cipher."""
    plaintext = "This is a test message for encryption algorithm identification."

    samples = []
    labels = []

    for _ in range(num_samples):
        # Generate ciphertext for each algorithm
        samples.extend([
            generate_aes_ciphertext(plaintext),
            generate_des_ciphertext(plaintext),
            generate_blowfish_ciphertext(plaintext),
            generate_rsa_ciphertext(plaintext),
            generate_sha256_hash(plaintext),
            generate_md5_hash(plaintext),
            generate_caesar_cipher(plaintext, shift=3)
        ])

        # Corresponding labels
        labels.extend([0, 1, 2, 3, 4, 5, 6])  # Labels for AES, DES, Blowfish, RSA, SHA-256, MD5, and Caesar Cipher

    return samples, labels

# Encryption and Hashing Functions
def generate_aes_ciphertext(plaintext):
    key = get_random_bytes(16)  # AES requires a 16-byte key
    cipher = AES.new(key, AES.MODE_ECB)  # Using ECB mode for simplicity
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_des_ciphertext(plaintext):
    key = get_random_bytes(8)  # DES requires an 8-byte key
    cipher = DES.new(key, DES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), DES.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_blowfish_ciphertext(plaintext):
    key = get_random_bytes(16)  # Blowfish key length can vary
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), Blowfish.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_rsa_ciphertext(plaintext):
    """Generate RSA ciphertext."""
    (public_key, private_key) = rsa.newkeys(512)  # Generate RSA public/private keys
    max_length = (public_key.n.bit_length() + 7) // 8 - 11  # Max size for plaintext with PKCS#1 v1.5 padding
    truncated_plaintext = plaintext[:max_length]  # Ensure plaintext fits within RSA limits
    ciphertext = rsa.encrypt(truncated_plaintext.encode(), public_key)  # Encrypt plaintext with the public key
    return base64.b64encode(ciphertext).decode()

def generate_sha256_hash(plaintext):
    """Generate SHA-256 hash."""
    sha256_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return sha256_hash

def generate_md5_hash(plaintext):
    """Generate MD5 hash."""
    md5_hash = hashlib.md5(plaintext.encode()).hexdigest()
    return md5_hash

def generate_caesar_cipher(plaintext, shift=3):
    """Generate Caesar Cipher ciphertext with a given shift."""
    ciphertext = ''
    for char in plaintext:
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            ciphertext += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            ciphertext += char
    return ciphertext

# Step 2: Feature Extraction
def calculate_entropy(text):
    """Calculate the Shannon entropy of a given text."""
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy





def is_base64(s):
    """Check if a given string is a valid base64 encoded string."""
    try:
        if isinstance(s, str):
            # Convert string to bytes for base64 decoding
            s_bytes = s.encode('utf-8')
        else:
            s_bytes = s

        # Ensure length is a multiple of 4
        if len(s_bytes) % 4 == 0:
            # Check if base64 decoding works
            base64.b64decode(s_bytes, validate=True)
            return True
    except (binascii.Error, ValueError):
        pass
    return False


def extract_features(ciphertext):
    """Extract features like character frequency, entropy, length, and base64 format presence from ciphertext."""
    # Character frequency
    counter = Counter(ciphertext)
    char_freq = [counter.get(chr(i), 0) for i in range(256)]  # ASCII frequencies
    char_freq = np.array(char_freq) / len(ciphertext)  # Normalize by length

    # Entropy
    entropy = calculate_entropy(ciphertext)

    # Check if the ciphertext is base64 encoded
    is_base64_encoded = int(is_base64(ciphertext))

    # Additional features
    features = np.append(char_freq, [entropy, len(ciphertext), is_base64_encoded])
    return features


# Step 3: Train the Machine Learning Model
def train_model(samples, labels):
    """Train a Random Forest classifier on the given samples."""
    # Extract features for each sample
    feature_matrix = np.array([extract_features(sample) for sample in samples])

    # Standardize the feature matrix
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix_scaled, labels, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    model.fit(X_train, y_train)

    # Save the trained model and scaler
    joblib.dump(model, 'crypto_classifier_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Best Parameters:", model.best_params_)
    print(classification_report(y_test, y_pred))

# Step 4: Predict the Algorithm
def predict_algorithm(ciphertext):
    """Predict the encryption algorithm used for the given ciphertext."""
    # Load the trained model and scaler
    model = joblib.load('crypto_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Extract features from the input ciphertext
    features = extract_features(ciphertext)
    features_scaled = scaler.transform([features])

    # Predict the algorithm
    predicted_algo = model.predict(features_scaled)

    # Map numeric prediction to algorithm names
    algo_dict = {0: 'AES', 1: 'DES', 2: 'Blowfish', 3: 'RSA', 4: 'SHA-256', 5: 'MD5', 6: 'Caesar Cipher'}
    predicted_algo_name = algo_dict.get(predicted_algo[0], 'Unknown')

    return predicted_algo_name

# Main Function
def main():
    # Step 1: Generate samples and labels
    samples, labels = generate_ciphertext_samples()

    # Step 2: Train the model
    train_model(samples, labels)

    # Step 3: Take user input for ciphertext
    user_ciphertext = input("Enter the ciphertext: ")

    # Step 4: Predict the algorithm
    result = predict_algorithm(user_ciphertext)
    print(f"The predicted cryptographic algorithm is: {result}")

if __name__ == "__main__":
    main()