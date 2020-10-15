import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math
import json
import pandas as pd

# DATASET PATH = "/home/arun/Jarvis/deepLearningAudio/Data/genres_original"
# Json PATH = "/home/arun/Jarvis/deepLearningAudio/Data/audio_data.json"
## Preprocessing Pipeline
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2046, hop_length=512, num_seg=5):
    """
        dataset_path
        json_path
        n_mfcc
        n_fft
        hop_length
        num_segments
    """
    data = {
        "mapping" : [],
        "mfcc": [],
        "labels": []
    }
    # no of sample per segments
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_seg)
    # expected_num_mfcc_vectors_per_segment
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    # loop throguh all the genres
    genre_index = -1
    for genre in os.listdir(dataset_path):
        genre_index += 1
        print("[{}] Genre processing is  : {}  ".format(genre_index, genre))
        data["mapping"].append(genre)
        for f in os.listdir(os.path.join(dataset_path, genre)):
            if not f.endswith(".wav"):
                continue
            print("[{}] File is --> {}".format(genre, f))
            file_path = os.path.join(dataset_path, genre, f)
            signal, sr = librosa.read(file_path, sr=SAMPLE_RATE)
            # process segments extracting mfcc and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment
                # store mfcc for segment if it has the expected length
                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                           sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length
                                           )
                mfcc = mfcc.T
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(genre_index)
    with open(json_path, 'w') as wfile:
        json.dump(data, wfile, indent=4)



if __name__ == "__main__":
    dataset_path = "/home/arun/Jarvis/deepLearningAudio/Data/genres_original"
    json_path = "/home/arun/Jarvis/deepLearningAudio/Data/audio_data.json"
    save_mfcc(dataset_path, json_path)
