from classifier_model import getClassifierModel
from sklearn.model_selection import train_test_split
from utils import plotModelCurves
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
import utils
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
from datetime import datetime

# load data sets
features_path = "../data/features_final/{0}"
result_path = "../training_result/"

f = joblib.load(features_path.format("all_features.pkl"))
l = joblib.load(features_path.format("all_labels.pkl"))
print("Data set shapes: {0} {1}".format(f.shape, l.shape))

train, test, trainLabel, testLabel = train_test_split(f, l, test_size=0.25, random_state=42)
print("Train shape: {0} Test shape: {1}".format(train.shape, test.shape))

model = getClassifierModel(name='Audio_Classifier_DNN')
# generate Tensorflow board stats for visualization
tb = TensorBoard(log_dir=result_path + datetime.now().strftime("%Y%m%d-%H%M%S") + "/", histogram_freq=2,
                 write_graph=True, write_images=True, write_grads=True)
# train the classification model. Employ Keras automatic validation split
history = model.fit(train, trainLabel, epochs=50, batch_size=256, validation_split=0.25, shuffle=True,
                    verbose=1,callbacks=[tb])

predictions = model.predict(test)
predictedLabel = np.argmax(predictions, axis=1)
report = metrics.classification_report(testLabel, predictedLabel, target_names=utils.classNames)
cf = confusion_matrix(testLabel, predictedLabel)
plotModelCurves(model, history, cf)
with open("../training_result/"+model.name+'.txt', 'w') as f:
     print(report, file=f)
print(report)
