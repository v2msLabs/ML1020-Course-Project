from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from datetime import datetime
import argparse
import os
from . import model
from . import util
from google.cloud import storage
import tensorflow as tf
import io

MODEL_NAME="urbansound.hdf5"
gcs = storage.Client()


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


# blob.download_to_filename("all_labels.pkl")
# f = joblib.load(ffile)
# l = joblib.load("all_labels.pkl")
def train(args):
    bucket = gcs.get_bucket("ml-1020")
    features_path = "features/{0}"
    feature_file = "all_features.pkl"
    label_file = "all_labels.pkl"
    lblob = bucket.blob(features_path.format(label_file))
    labels = lblob.download_as_string()
    fblob = bucket.blob(features_path.format(feature_file))
    features = fblob.download_as_string()
    #
    labels = joblib.load(io.BytesIO(labels))
    features = joblib.load(io.BytesIO(features))
    train, test, trainLabel, testLabel = train_test_split(features, labels, test_size=0.25, random_state=42)
    # train = np.asarray([[0.1, 1, 0.9, 2, 0.6], [0.1, 1, 0.9, 2, 0.6],
    #                     [0.1, 1, 0.9, 2, 0.6], [0.1, 1, 0.9, 2, 0.6],
    #                     [0.1, 1, 0.9, 2, 0.6], [0.1, 1, 0.9, 2, 0.6]])
    # trainLabel = np.asarray([1, 1, 2, 1, 1, 1])
    m = model.getModel()
    # Setup Learning Rate decay.
    decay = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    tb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir,
          'tensorboard_{0}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))),
          histogram_freq=0)
    m.fit(train, trainLabel, epochs=50, batch_size=256, shuffle=True, verbose=1,
          validation_split=0.25, callbacks=[decay, tb])
    # if args.job_dir.startswith('gs://'):
    #     model.save(MODEL_NAME)
    #     util.copy_file_to_gcs(args.job_dir, MODEL_NAME)
    # else:
    #     model.save(os.path.join(args.job_dir, MODEL_NAME))
    # export_path = tf.contrib.saved_model.save_keras_model(
    #     model, os.path.join(args.job_dir, 'keras_export'))
    # export_path = export_path.decode('utf-8')
    # print('Model exported to: ', export_path)

if __name__ == '__main__':
    args = get_args()
    print('the task has started')
    tf.logging.set_verbosity(args.verbosity)
    train(args)
