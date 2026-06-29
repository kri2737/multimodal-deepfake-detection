import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc.flatten()
    except Exception as e:
        return None

data = []
labels = []
base_dir = 'dataset'

print("Processing Real Samples...")
real_folder = os.path.join(base_dir, 'real_samples')
for file in tqdm(os.listdir(real_folder)):
    if file.endswith('.wav'):
        path = os.path.join(real_folder, file)
        features = extract_mfcc(path)
        if features is not None:
            data.append(features)
            labels.append(0)  # 0 = Real

print("Processing Fake Samples...")
fake_folders = ['FlashSpeech', 'OpenAI', 'VALLE', 'xTTS', 'NaturalSpeech3', 'PromptTTS2', 'seedtts_files', 'VoiceBox']

for folder_name in fake_folders:
    folder = os.path.join(base_dir, folder_name)
    if os.path.exists(folder):
        print(f"Processing {folder_name}...")
        for file in tqdm(os.listdir(folder)):
            if file.endswith('.wav'):
                path = os.path.join(folder, file)
                features = extract_mfcc(path)
                if features is not None:
                    data.append(features)
                    labels.append(1)  # 1 = Fake

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('features.csv', index=False)
print(f"✅ Feature extraction completed! Total samples: {len(df)}")
print(f"Real: {labels.count(0)} | Fake: {labels.count(1)}")
