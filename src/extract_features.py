import sys
from sklearn.externals import joblib
import time
from utils import extractFeatures
import glob

print(sys.argv)
print("Command : python extract_features.py input_data_path output_data_path [output_file_prefix]")
input_path = sys.argv[1]
output_path = sys.argv[2]
prefix = ""
if (len(sys.argv)>3):
    prefix = sys.argv[3]
print("Parameters received:\nInput path: {0}\nOutput path: {1}\nOutput files prefix:{2}".format(
    input_path,output_path,prefix
))

paths = glob.glob(input_path)
print("Started processing. Number of files: {0}".format(len(paths)))
start = time.time()
features, labels = extractFeatures(paths,windowSize=500,dimension=96)
if(len(features)>0):
    joblib.dump(features, '{0}/{1}mel_features.pkl'.format(output_path,prefix))
    joblib.dump(labels, '{0}/{1}mel_labels.pkl'.format(output_path,prefix))
else:
    print("The input data directory is empty. No files generated...")

print("feature shape: {0} labels shape:{1}".format(features.shape,labels.shape))
print("Execution time(sec): {0}".format(time.time() - start))

