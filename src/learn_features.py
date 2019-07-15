import sys
from sklearn.externals import joblib
import time
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import *

print(sys.argv)
print("Command : python learn_features.py feature_file_path output_path [output_file_prefix]")
feature_file = sys.argv[1]
output_path = sys.argv[2]
prefix = ""
if (len(sys.argv)>3):
    prefix = sys.argv[3]
print("Parameters received:\nFeature file: {0}\nOutput path: {1}\nOutput files prefix:{2}".format(
    feature_file,output_path,prefix
))
print("Started processing")
start = time.time()
features = joblib.load(feature_file)
print("feature shape: {0}".format(features.shape))

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(96,96,3))
output = vgg.layers[-1].output
output = Flatten()(output)
featureExtractor = Model(vgg.input, output)
featureExtractor.trainable = False

features_learnt = featureExtractor.predict(preprocess_input(features))
if(len(features_learnt)>0):
    joblib.dump(features_learnt, '{0}/{1}lrn_features.pkl'.format(output_path,prefix))
else:
    print("The input data directory is empty. No files generated...")
print("Features shape: {0}".format(np.shape(features_learnt)))

print("Execution time(sec): {0}".format(time.time() - start))