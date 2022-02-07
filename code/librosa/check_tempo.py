# the original audio example.wav is from file-examples.com
import librosa

# absolute route used
filename = 'C:/Users/Administrator/dev/Acoustic_LAB/code/example.wav'

# audio is mixed to mono and resampled to 22050 Hz at load time
y, sr = librosa.load(filename)

# tempo: bpm
# beat_frames: an array of frame numbers
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# beat_times: an array of timestamps(in sec)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)