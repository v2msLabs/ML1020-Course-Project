from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from collections import Counter

features_path = "../data/features_final/{0}"

f = features_path.format("all_features.pkl")
l = features_path.format("all_labels.pkl")

ft = joblib.load(f)
lbl = joblib.load(l)

print("Whole set shapes: {0} {1}".format(ft.shape,lbl.shape))

f1, f2, l1, l2 = train_test_split(ft, lbl, test_size=0.5, random_state=42)
print("50/50 Shapes: {0} {1} {2} {3}".format(f1.shape,l1.shape,f2.shape,l2.shape))

print("Count by category for the first half: {0}".format(Counter(l1)))
print("Count by category for the second half: {0}".format(Counter(l2)))

joblib.dump(l1, features_path.format('half_labels.pkl'))
joblib.dump(f1, features_path.format('half_features.pkl'))

fq1, fq2, lq1, lq2 = train_test_split(f2, l2, test_size=0.5, random_state=482)
print("Quarter Size Shapes: {0} {1} {2} {3}".format(fq1.shape,lq1.shape,fq2.shape,lq2.shape))

print("Count by category for the quarter: {0}".format(Counter(lq1)))
print("Count by category for the quarter: {0}".format(Counter(lq2)))

joblib.dump(lq1, features_path.format('quarter_labels.pkl'))
joblib.dump(fq1, features_path.format('quarter_features.pkl'))