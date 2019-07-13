# This is pretty straightforward, dumb but simple approach to stack outputs
# of the multiple runs of the extract_features.py  into two large pkl files:
# features and labels
from sklearn.externals import joblib
import numpy as np

features_path = "../data/features/{0}"
l1 = features_path.format("fold1_mel_labels.pkl")
l2 = features_path.format("fold2_mel_labels.pkl")
l3 = features_path.format("fold3_mel_labels.pkl")
#f1 = joblib.load(paths[0])
lbl1 = joblib.load(l1)
lbl2 = joblib.load(l2)
lbl3 = joblib.load(l3)

print("Label 1 shape: {0}".format(np.shape(lbl1)))
print("Label 2 shape: {0}".format(np.shape(lbl2)))
print("Label 3 shape: {0}".format(np.shape(lbl3)))

lbl = np.concatenate((lbl1,lbl2,lbl3))

print("Labels shape: {0}".format(np.shape(lbl)))

print("done")