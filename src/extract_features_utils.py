import numpy as np
import librosa
import librosa.display


# function to get start and end indices for audio sub-sample
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


# This procedure samples the audio file generates log-scaled mel spectrogram for each sample and labels the samples
# it returns to array. The first is N "dimension x dimension" log-scaled mel spectrogram band/ frame
# matrices.
# The second output value is a vector of labels fo each row in in the firs array
#
# windowSize - sample window size in milliseconds
# dimension - dimension of a "band by frame" square
# sampleRate - sample rate
#
# Example
#
# extractFeatures('test.wav', windowSize = 500, dimension = 96)
def extractFeatures(paths, sampleRate=22050, windowSize=1000, dimension=64):
    hop_length = int(windowSize / 1000 * sampleRate / (dimension - 1))
    window_size = hop_length * (dimension - 1)
    log_specgrams_full = []
    class_labels = []
    # for each audio sample
    for path in paths:
        sound_data, sr = librosa.load(path, sr=sampleRate)
        soundSize = len(sound_data) - 1
        if (soundSize < window_size):
            continue
        file_name = path.split('\\')[-1]
        class_label = file_name.split('-')[1]
        # for each audio signal sub-sample window of data
        for (start, end) in windows(sound_data, window_size):
            sample_size = len(sound_data[start:end]);
            tooSmall = sample_size < window_size
            if (tooSmall):  # for the last slice get a window_size piece from the audio file end
                signal = sound_data[(soundSize - window_size):soundSize]
            else:
                signal = sound_data[start:end]

            melspec_full = librosa.feature.melspectrogram(signal, n_mels=dimension,
                                                          sr=sampleRate, hop_length=hop_length)
            logspec_full = librosa.amplitude_to_db(melspec_full)
            logspec_full = logspec_full.T.flatten()[:, np.newaxis].T
            log_specgrams_full.append(logspec_full)
            class_labels.append(class_label)
            if (tooSmall):  # sample size
                break
    # create feature maps
    feature = np.asarray(log_specgrams_full).reshape(len(log_specgrams_full), dimension, dimension, 1)
    feature = np.tile(feature, (1, 1, 1, 3))
    return np.array(feature), np.array(class_labels, dtype=np.int)
