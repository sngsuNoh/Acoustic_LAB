# feature extraction

import librosa
import numpy as np

filename = 'C:/Users/Administrator/dev/Acoustic_LAB/code/example.wav'
y, sr = librosa.load(filename)

# 512 samples ~=23ms
hop_length = 512

# seperate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

# compute MFCC features from the raw signal
# MFCC: 일정 구간에 대한 스펙트럼을 분석하여 음성의 특징을 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# and from the first-order difference (= delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# stack and synchronize between beat events
# use mean value instead of median (as default)
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

# compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# aggregate chroma features between beat events
# use median
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

# stack all beat-synchronous features
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])