# data_utils.py
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def load_raw_data(df, sampling_rate):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def load_and_preprocess_data():
    sampling_rate = 100

    # load and convert annotation data
    df = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    signals = load_raw_data(df, sampling_rate)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv('scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    df.reset_index(inplace=True)
    labels = df.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    x = []
    y = []

    valid_labels = [list(['CD']), list(['HYP']), list(['NORM']), list(['STTC']), list(['MI'])]
    for i in range(len(labels)):
        if labels[i] in valid_labels:
            y.append(labels[i])
            x.append(signals[i])
    
    x = np.array(x)
    y = np.array(y)

    augmented_signals = []
    augmented_labels = []

    for label in ['CD', 'HYP', 'MI', 'STTC']:
        target_indices = np.where(y == label)[0]
        augmentation_factor = (np.count_nonzero(y == 'NORM')  // np.count_nonzero(y == label)) -1
        
        for index in target_indices:
            repeated_signal = np.tile(x[index], (augmentation_factor, 1, 1))
            noisy_signal = repeated_signal + 0.01 * np.random.randn(*repeated_signal.shape)
            augmented_signals.append(noisy_signal)
            augmented_labels.extend([y[index]] * augmentation_factor)
    
    X = np.concatenate([x] + augmented_signals)
    Y = np.concatenate([y, np.array(augmented_labels)])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)
    y_onehot = to_categorical(y_encoded, num_classes=5)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=None)

    return X_train, X_test, Y_train, Y_test, label_encoder

# Example usage:
# X_train, X_test, Y_train, Y_test, label_encoder = load_and_preprocess_data()