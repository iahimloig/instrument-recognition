import json
import os
import math
import librosa
import librosa.display
import numpy as np
 
DATASET_PATH = "/content/drive/MyDrive/INSTRUMENTE"
JSON_PATH = "data.json"
SAMPLE_RATE = 44100
TRACK_DURATION = 3 # masurat in secunde
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
 
 
def save_mfcc(dataset_path, json_path, num_mfcc=31, n_fft=2048, hop_length=1024, num_segments=6):
    
 
    # dictionar pentru mapare, etichete si MFCC
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
 
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
 
    # trecem prin fiecare director in care avem instrumente
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
 
        # ne asiguram ca suntem intr-un folder cu un tip de instrument
        if dirpath is not dataset_path:
 
            # salvam numele instrumentului 
            semantic_label = dirpath.split("/")[-1]
            semantic_label = os.path.split(dirpath)[-1]
            data["mapping"].append(semantic_label)
            print("\nProcesand: {}".format(semantic_label))
 
 
            # procesam fisierele audio din fiecare folder de instrumente 
            for f in filenames:
 
        # incarcam fisierele audio
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
 
                # procesam fiecare segment
                for d in range(num_segments):
 
                    # calculam inceputul si sfarsitul segmentului actual 
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
 
                    # extragem mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc=mfcc.T
 
                    # incarcam mfcc-uri de lungimea dorita 
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

 
    # salvam mfcc in fisierul json 
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=6)
