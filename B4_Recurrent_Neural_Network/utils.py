import matplotlib.pyplot as plt
import os
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def plot_all_metrics(histories, SAVE_DIR, fileName):
    # Ensure the input is always a dictionary
    if not isinstance(histories, dict):
        histories = {"Model_1": histories}
        
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Extract all unique metrics from histories
    all_metrics = set()
    for history in histories.values():
        all_metrics.update(history.history.keys())
        
    plt.figure(figsize=(15, 5 * len(all_metrics)))
    
    for i, metric in enumerate(all_metrics):
        plt.subplot(len(all_metrics), 2, 2 * i + 1)
        for name, history in histories.items():
            if metric in history.history:
                plt.plot(history.history[metric], label=f'{name} Train {metric.capitalize()}')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label=f'{name} Validation {metric.capitalize()}', linestyle='dashed')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f'Model {metric.capitalize()} Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fileName))
    plt.show()


def encode_review(text, maxlen=20, max_features=10000):
    # Récupérer le dictionnaire mot -> index
    word_index = imdb.get_word_index()
    
    # Tokeniser et convertir en indices
    # (IMDB décale les indices de 3 : 0=padding, 1=début, 2=inconnu)
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    
    # Garder uniquement les mots dans le vocabulaire
    encoded = [idx if idx < max_features else 2 for idx in encoded]
    
    # Padder à la longueur maxlen
    padded = pad_sequences([encoded], maxlen=maxlen)
    
    return padded