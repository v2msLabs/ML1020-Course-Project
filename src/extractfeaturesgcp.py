import sys
from operator import add
from pyspark.sql import SparkSession

import pyspark
import random
from pyspark.rdd import RDD
from pyspark import SparkContext, SparkConf
import numpy as np
import os
import librosa
import librosa.display
import IPython.display
import soundfile as sf
import glob
import random
import pickle
import io
from google.cloud import storage

# windowSize - sample window size in milliseconds
# dimention - dimention of a "band by frame" square

# function to get start and end indices for audio sub-sample
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


# This procesure samples the audio file generates log-scaled mel spectrogram for each sample and labels the samples
# it reurns to array. The first is N "dimention x dimention" log-scaled mel spectorgam band/ frame matrixes.
# The second output value is a vector of labels fo each row in in the firs array
#
# windowSize - sample window size in milliseconds
# dimention - dimention of a "band by frame" square
# sampleRate - sample rate
#
# Example
#
# extractFeatures('test.wav', windowSize = 500, dimention = 96)
def extractFeatures(path, sampleRate=22050, windowSize=500, dimention=96):
    #sample path from bucket: gs://sampleguysbucket/UrbanSound8K/audio/folder1/7061-6-0-0.wav
    #get bucketname from above and file path string without bucket
    print("extractFeatures START-----")
    print("extractFeatures path-----"+ path + "-------")
    path_parts = path.replace("gs://", "").split("/")
    BUCKET = path_parts.pop(0)
    file_name = "/".join(path_parts)
    print("extractFeatures BUCKET-----" + BUCKET + "-------" + " file_name---" + file_name + "-----")
    hop_length = int(windowSize / 1000 * sampleRate / (dimention - 1))
    window_size = hop_length * (dimention - 1)
    log_specgrams_full = []
    class_labels = []
    # Create a Cloud Storage client.
    gcs = storage.Client()
    # Get the bucket that the file will be read from.
    bucket = gcs.get_bucket(BUCKET)
    # read a blob
    blob = bucket.blob(file_name)
    file_as_string = blob.download_as_string()
    #print(file_as_string)

    # convert the string to bytes and then finally to audio samples as floats and the audio sample rate
    sound_data, sr = sf.read(io.BytesIO(file_as_string))
    sound_data = librosa.resample(sound_data.T, sr, sampleRate)
    sound_data = librosa.to_mono(sound_data)
    soundSize = len(sound_data) - 1
    fn = path.split('/')[-1]
    class_label = fn.split('-')[1]
    # for each audio signal sub-sample window of data
    for (start, end) in windows(sound_data, window_size):
        sample_size = len(sound_data[start:end])
        tooSmall = sample_size < window_size
        if (tooSmall):  # for the last slice get a wnow_size piece from the audo file end
            signal = sound_data[(soundSize - window_size):soundSize]
        else:
            signal = sound_data[start:end]

        # get the log-scaled mel-spectrogram
        melspec_full = librosa.feature.melspectrogram(signal, n_mels=dimention,
                                                      sr=sampleRate, hop_length=hop_length)
        logspec_full = librosa.amplitude_to_db(melspec_full)
        logspec_full = logspec_full.T.flatten()[:, np.newaxis].T
        log_specgrams_full.append(logspec_full)
        class_labels.append(class_label)
        if (tooSmall):  # sample size
            break
    # create the first two feature maps
    feature = np.asarray(log_specgrams_full).reshape(len(log_specgrams_full), dimention,dimention, 1)
    feature = np.tile(feature, (1, 1, 1, 3))
    data1 = np.array(feature)
    labels = np.array(class_labels, dtype=np.int)
    dataWithLabels = np.array(list(zip(data1, labels)))
    # return np.array(feature), np.array(class_labels, dtype = np.int)
    print("extractFeatures END-----")
    return dataWithLabels

if __name__ == "__main__":
    baseDir = "gs://sampleguysbucket/UrbanSound8K"
    metadataDir = "gs://sampleguysbucket/UrbanSound8K/metadata"
    audiodir = "gs://sampleguysbucket/UrbanSound8K/audio"
    subfolders = "**"
    minfold = "folder1"
    separator = "/"
    audioFileExtension = "*.wav"

    conf = SparkConf().setAppName('urbansounds').setMaster('local')
    sc = SparkContext.getOrCreate(conf=conf)
    rdd_binary = sc.binaryFiles(audiodir + separator + minfold + separator + audioFileExtension)
    rdd_binary.collect()
    print("----rdd filecount start----")
    binCount = rdd_binary.count()
    print(binCount)
    print("---rdd filecount end---")
    listOfFiles1 = rdd_binary.keys().take(binCount)
    print(listOfFiles1)
    print(rdd_binary.keys())

    rdd = sc.parallelize(listOfFiles1, 8)
    map_rdd = rdd.map(extractFeatures)
    dataWithLabels = map_rdd.collect()

    pickle_out = open("dataWithLabels.pickle", "wb")
    pickle.dump(dataWithLabels, pickle_out)
    pickle_out.close()

    #pickle_in = open("dataWithLabels.pickle", "rb")
    #dataWithLabels1 = pickle.load(pickle_in)
    #print(dataWithLabels1)
    sc.stop()
