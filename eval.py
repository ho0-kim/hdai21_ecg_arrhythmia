import argparse

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(777)
import tensorflow as tf
tf.random.set_seed(777)

from read_data import read_data
from preprocess import minmixscale, elevation

parser = argparse.ArgumentParser(prog='eval.py', 
                                description='Evaluate trained model for ECG data')

parser.add_argument('-d', '--data', type=str, required=True,
                    help='File path of validation data')
parser.add_argument('-m', '--model', default='model.h5', type=str,
                    help='File name of saved model')                    
parser.add_argument('-l', '--lead', default=2, type=int,
                    help='Number of leads being trained (default=2)')
parser.add_argument('-v', '--elevation', default=False, action='store_true',
                    help='Option for adjusting elevation')

args = parser.parse_args()

def report(y_val, y_pred):
    print(metrics.classification_report(y_val, (y_pred>0.5)))

    auc = metrics.roc_auc_score(y_val, y_pred)
    print("ROC-AUC Score:", auc)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_val, y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.6f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('roc_curve.png')

if __name__ == '__main__':
    print("reading files and extracting data...")
    X_val, y_val = read_data(args.data, n_leads=args.lead)

    print("applying Min-Max scaier...")
    X_val = minmixscale(X_val)

    if args.elevation:
        print("applying elevation adjustment...")
        X_val = elevation(X_val, n_leads=args.lead)

    model = tf.keras.models.load_model(args.model)
    model.summary()

    y_pred = model.predict(X_val, batch_size=1000)

    report(y_val, y_pred)