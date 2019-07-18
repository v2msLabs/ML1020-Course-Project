# This is pretty straightforward, dumb but simple approach to stack outputs
# of the multiple runs of the extract_features.py  into two large pkl files:
# features and labels
from sklearn.externals import joblib
import numpy as np
import time

start = time.time()
features_path = "../data/features/{0}"
output_path = "../data/features_final/"

l1 = features_path.format("fold1_mel_labels.pkl")
l2 = features_path.format("fold2_mel_labels.pkl")
l3 = features_path.format("fold3_mel_labels.pkl")
l4 = features_path.format("fold4_mel_labels.pkl")
l5 = features_path.format("fold5_mel_labels.pkl")
l6 = features_path.format("fold6_mel_labels.pkl")
l7 = features_path.format("fold7_mel_labels.pkl")
l8 = features_path.format("fold8_mel_labels.pkl")
l9 = features_path.format("fold9_mel_labels.pkl")
l10 = features_path.format("fold10_mel_labels.pkl")
lau = features_path.format("aug_mel_labels.pkl")

f1 = features_path.format("fold1_lrn_features.pkl")
f2 = features_path.format("fold2_lrn_features.pkl")
f3 = features_path.format("fold3_lrn_features.pkl")
f4 = features_path.format("fold4_lrn_features.pkl")
f5 = features_path.format("fold5_lrn_features.pkl")
f6 = features_path.format("fold6_lrn_features.pkl")
f7 = features_path.format("fold7_lrn_features.pkl")
f8 = features_path.format("fold8_lrn_features.pkl")
f9 = features_path.format("fold9_lrn_features.pkl")
f10 = features_path.format("fold10_lrn_features.pkl")
fau = features_path.format("aug_lrn_features.pkl")

lbl1 = joblib.load(l1)
lbl2 = joblib.load(l2)
lbl3 = joblib.load(l3)
lbl4 = joblib.load(l4)
lbl5 = joblib.load(l5)
lbl6 = joblib.load(l6)
lbl7 = joblib.load(l7)
lbl8 = joblib.load(l8)
lbl9 = joblib.load(l9)
lbl10 = joblib.load(l10)
lblau = joblib.load(lau)

print("Label 1 shape: {0}".format(np.shape(lbl1)))
print("Label 2 shape: {0}".format(np.shape(lbl2)))
print("Label 3 shape: {0}".format(np.shape(lbl3)))
print("Label 4 shape: {0}".format(np.shape(lbl4)))
print("Label 5 shape: {0}".format(np.shape(lbl5)))
print("Label 6 shape: {0}".format(np.shape(lbl6)))
print("Label 7 shape: {0}".format(np.shape(lbl7)))
print("Label 8 shape: {0}".format(np.shape(lbl8)))
print("Label 9 shape: {0}".format(np.shape(lbl9)))
print("Label 10 shape: {0}".format(np.shape(lbl10)))
print("Label aug shape: {0}".format(np.shape(lblau)))

lbl = np.concatenate((lbl1,lbl2,lbl3,lbl4,lbl5,lbl6,lbl7,lbl8,lbl9,lbl10,lblau))

print("------> Label shape: {0}".format(np.shape(lbl)))

ft1 = joblib.load(f1)
ft2 = joblib.load(f2)
ft3 = joblib.load(f3)
ft4 = joblib.load(f4)
ft5 = joblib.load(f5)
ft6 = joblib.load(f6)
ft7 = joblib.load(f7)
ft8 = joblib.load(f8)
ft9 = joblib.load(f9)
ft10 = joblib.load(f10)
ftau = joblib.load(fau)

print("Feature 1 shape: {0}".format(np.shape(ft1)))
print("Feature 2 shape: {0}".format(np.shape(ft2)))
print("Feature 3 shape: {0}".format(np.shape(ft3)))
print("Feature 4 shape: {0}".format(np.shape(ft4)))
print("Feature 5 shape: {0}".format(np.shape(ft5)))
print("Feature 6 shape: {0}".format(np.shape(ft6)))
print("Feature 7 shape: {0}".format(np.shape(ft7)))
print("Feature 8 shape: {0}".format(np.shape(ft8)))
print("Feature 9 shape: {0}".format(np.shape(ft9)))
print("Feature 10 shape: {0}".format(np.shape(ft10)))
print("Feature aug shape: {0}".format(np.shape(ftau)))

ft = np.vstack((ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ftau))

print("-------> Features shape: {0}".format(np.shape(ft)))

joblib.dump(lbl, '{0}all_labels.pkl'.format(output_path))
joblib.dump(ft, '{0}all_features.pkl'.format(output_path))

print("Elapsed time (sec): {0}".format(time.time() - start))
