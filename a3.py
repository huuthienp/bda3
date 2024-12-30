# %% [markdown]
# # Environment

# %% [markdown]
# ## Functions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-28T03:26:52.900117Z","iopub.execute_input":"2024-10-28T03:26:52.900491Z","iopub.status.idle":"2024-10-28T03:26:52.913016Z","shell.execute_reply.started":"2024-10-28T03:26:52.900448Z","shell.execute_reply":"2024-10-28T03:26:52.911830Z"}}
import json
import os
import subprocess
import sys
import traceback

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-28T03:26:52.914574Z","iopub.execute_input":"2024-10-28T03:26:52.914958Z","iopub.status.idle":"2024-10-28T03:26:52.928973Z","shell.execute_reply.started":"2024-10-28T03:26:52.914917Z","shell.execute_reply":"2024-10-28T03:26:52.927798Z"}}
def ensure_pip() -> None:
    command = [sys.executable, '-m', 'ensurepip']
    print('Ensuring pip is installed...', end=' ', flush=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print('Success.')
    except subprocess.CalledProcessError as e:
        print('Failed.')
        print(e.stderr.strip(), file=sys.stderr)
    except Exception as e:
        print('Unexpected error.')
        traceback.print_exc()


def pip_install(requirements = None) -> None:
    '''
    Install Python packages using pip.

    Args:
        requirements (list): List of package names to install.

    Returns: None.
    '''
    # Check if requirements list is None or empty
    if not requirements:
        print('Error: No requirements provided. Aborting installation.', file=sys.stderr)
        return None

    # Prepare base command for pip install
    command = [sys.executable, '-m', 'pip', 'install', '--quiet']

    print('Starting package installation...')
    for r in requirements:
        try:
            # Attempt to install each package
            print(f'Installing {r}...', end=' ', flush=True)
            subprocess.run(command + [r], check=True, capture_output=True, text=True)
            print('Success.')
        except subprocess.CalledProcessError as e:
            # Handle pip installation errors
            print('Failed.')
            print(e.stderr.strip(), file=sys.stderr)
        except Exception as e:
            # Handle any unexpected errors
            print('Unexpected error.')
            traceback.print_exc()

    return None

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-28T03:26:52.930534Z","iopub.execute_input":"2024-10-28T03:26:52.931241Z","iopub.status.idle":"2024-10-28T03:26:52.948294Z","shell.execute_reply.started":"2024-10-28T03:26:52.931165Z","shell.execute_reply":"2024-10-28T03:26:52.947044Z"}}
def check_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))


def plot_confusion_matrix(cm, title='Confusion Matrix', labels=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels if labels is not None else 'auto',
                yticklabels=labels if labels is not None else 'auto')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_learning_curve(evals_result, metric='MultiClass'):
    plt.figure(figsize=(10, 6))

    # Extract learning curve data for train and validation sets
    train_loss = evals_result['learn'][metric]
    valid_loss = evals_result['validation'][metric]

    # Plotting
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    #Plot Learning Curve for Baseline and Final Models
def plot_learning_curve_2(history, title='Learning Curve'):
    plt.figure(figsize=(10, 6))

    # Extract loss and accuracy data
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    valid_accuracy = history.history['val_accuracy']

    # Plot Training vs Validation Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({title})')
    plt.legend()

    # Plot Training vs Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(valid_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training vs Validation Accuracy ({title})')
    plt.legend()

    plt.tight_layout()
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-28T03:26:52.952205Z","iopub.execute_input":"2024-10-28T03:26:52.952882Z","iopub.status.idle":"2024-10-28T03:26:52.962496Z","shell.execute_reply.started":"2024-10-28T03:26:52.952839Z","shell.execute_reply":"2024-10-28T03:26:52.961389Z"}}
def add_max_prob_and_gap(df, true_lable):
    """
    Adds max probability and probability gap between the top two classes to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with probabilities for each class as columns, and an optional true_label column.

    Returns:
    pd.DataFrame: A new DataFrame with true_label, max_prob, and prob_gap.
    """
    df_copy = df.copy()

    # 1. Calculate the max probability for each row
    df_copy['max_prob'] = df_copy[df_copy.columns[:-1]].max(axis=1)

    # 2. Calculate the gap between the top two probabilities for each row
    df_copy['prob_gap'] = df_copy[df_copy.columns[:-2]].apply(lambda x: x.nlargest(2).diff().iloc[-1], axis=1)

    # 3. Find the column name corresponding to the max probability
    df_copy['pred_label'] = df_copy[df_copy.columns[:-3]].idxmax(axis=1)

    # 4. Get the probability corresponding to the specified column
    df_copy['true label prob'] = df_copy[true_lable]

    return df_copy

# %% [markdown]
# ## Import

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-28T03:26:52.963856Z","iopub.execute_input":"2024-10-28T03:26:52.964379Z"}}
# ensure_pip()
# pip_install(
#     [
#         'catboost==1.2.7',
#         'dask==2024.9.1',
#         'dask-expr==1.1.15',
#         # 'dask_ml==2024.4.4',
#         # 'lightgbm==4.2.0',
#         'optuna==4.0.0',
#         'optuna-integration==4.0.0',
#         'optuna-integration[catboost]',
#         'shap==0.44.1',
#         # 'xgboost==2.0.3',
#     ]
# )

# %% [code] {"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import catboost as cb
from catboost import CatBoostClassifier, Pool

import optuna
from optuna.integration import CatBoostPruningCallback

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# from scikeras.wrappers import KerasClassifier

import shap

# %% [markdown]
# # Main

# %% [markdown]
# ## Preparation

# %% [code] {"jupyter":{"outputs_hidden":false}}
PATHS = {
    'train': 'train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv/finetune/pytorch-image-models-main/train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv',
    'test1': 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv',
    'test2': 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
}
names = {key: os.path.splitext(os.path.basename(path))[0] for key, path in PATHS.items()}
BLOCK_SIZE = '512 MB'

on_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if on_kaggle:
    PATHS = {key: '/kaggle/input/deep-features-eva02/' + path for key, path in PATHS.items()}

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Load datasets
try:
    # Use dask for the large training dataset to handle it efficiently
    print(f'Reading training set with Dask by {BLOCK_SIZE}...', end=' ')
    ddf_tr = dd.read_csv(PATHS['train'], blocksize=BLOCK_SIZE)
    print('Success.')
    print(names['train'],
          f'has {ddf_tr.npartitions} partitions',
          f'and {len(ddf_tr.columns)} columns',
         )

    print('Reading test set 1...', end=' ')
    df_ts1 = pd.read_csv(PATHS['test1'])
    print('Success.')
    print(names['test1'], df_ts1.shape)

    print('Reading test set 2...', end=' ')
    df_ts2 = pd.read_csv(PATHS['test2'])
    print('Success.')
    print(names['test2'], df_ts2.shape)

except FileNotFoundError as e:
    print('Faled.')
    print(e, file=sys.stderr)
    raise

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(names['train'].upper())
print(ddf_tr.head())
print(ddf_tr.dtypes.value_counts())
print('int:', ddf_tr.select_dtypes([np.int_]).columns.tolist())
print('string:', ddf_tr.select_dtypes(['string']).columns.tolist())

# %% [code] {"jupyter":{"outputs_hidden":false}}
# print('Exploring class balance...', end=' ')
# label_counts = ddf_tr['label'].value_counts().compute()
# print('Success.')
# print(label_counts)

# nan_check = ddf_tr.isna().sum().compute()
# print(nan_check)
# if nan_check.sum() == 0: # True
    # print('No missing values')

print('Feature target is balanced.')
print('No missing value.')

# %% [code] {"jupyter":{"outputs_hidden":false}}
scaler = StandardScaler()
X_dtypes = [np.float_]
X_col_i = ddf_tr.select_dtypes(X_dtypes).columns.map(ddf_tr.columns.get_loc)
print('Fitting standard scaler to whole training set, taking 20 minutes...', end=' ')
scaler.fit(ddf_tr.iloc[:, X_col_i])
print('Success.')

# %% [code] {"jupyter":{"outputs_hidden":false}}
print('Transforming...', end=' ')

# Apply scaling to each partition
X_train = ddf_tr.iloc[:, X_col_i].map_partitions(lambda p: pd.DataFrame(scaler.transform(p),
                                                                        columns=ddf_tr.columns),
                                                 meta=ddf_tr._meta).compute()

X_test = scaler.transform(df_ts1.iloc[:, X_col_i])

X_test2 = scaler.transform(df_ts2.iloc[:, X_col_i])

print('Success.')

# %% [code] {"jupyter":{"outputs_hidden":false}}
y_train = ddf_tr['label'].compute()

y_test = df_ts1['label']

y_test2 = df_ts2['label']

# %% [markdown]
# ## CatBoost

# %% [markdown]
# ### Building

# %% [code] {"jupyter":{"outputs_hidden":false}}
batch_size = 5000
ipca = IncrementalPCA()

for start in range(0, X_train.shape[0], batch_size):
    end = min(start + batch_size, X_train.shape[0])
    X_batch = X_train.iloc[start:end].values
    #print(f'Processing batch from index {start} to {end}, shape: {X_batch.shape}')
    ipca.partial_fit(X_batch)

# Calculate cumulative explained variance
explained_variance_ratio = ipca.explained_variance_ratio_.cumsum()
print("Explained Variance Ratios:", ipca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.axhline(y=0.90, color='r', linestyle='--')  # Add a reference line at 90%
plt.show()

# Choose the number of components to retain 90% variance
n_components_90 = (explained_variance_ratio >= 0.90).argmax() + 1
print(f'Number of components to retain 90% variance: {n_components_90}')

# Initialize IncrementalPCA with the number of components determined earlier
ipca = IncrementalPCA(n_components=n_components_90)

#Fit IncrementalPCA in smaller batches
for start in range(0, X_train.shape[0], batch_size):
    end = min(start + batch_size, X_train.shape[0])
    X_batch = X_train.iloc[start:end].values  # Convert to numpy array
    ipca.partial_fit(X_batch)

#Transform X_train in batches and store the results
X_train_transformed = []

for start in range(0, X_train.shape[0], batch_size):
    end = min(start + batch_size, X_train.shape[0])
    X_batch = X_train.iloc[start:end].values  # Convert to numpy array
    X_batch_transformed = ipca.transform(X_batch)
    X_train_transformed.append(X_batch_transformed)

#Concatenate the transformed batches back together
X_train_transformed = np.vstack(X_train_transformed)  # Use vstack for efficient concatenation

print(f'Shape of X_train_transformed: {X_train_transformed.shape}')
# Save the transformed training data
np.save('X_train_transformed.npy', X_train_transformed)

# Transform the entire test sets without batching
X_test_transformed = ipca.transform(X_test)
X_test2_transformed = ipca.transform(X_test2)

print(f'Shape of X_test_transformed: {X_test_transformed.shape}')
# Save the transformed test set 1
np.save('X_test_transformed.npy', X_test_transformed)

print(f'Shape of X_test2_transformed: {X_test2_transformed.shape}')
# Save the transformed test set 2
np.save('X_test2_transformed.npy', X_test2_transformed)

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Stratified Split for Training and Validation
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(split.split(X_train_transformed, y_train))

X_train_full = X_train_transformed[train_idx]
y_train_full = y_train.iloc[train_idx].values.ravel()

X_valid = X_train_transformed[valid_idx]
y_valid = y_train.iloc[valid_idx].values.ravel()

#Sample a Subset of Training Data
sample_fraction = 0.5  # Define the fraction of data to sample (e.g., 50%)
sample_size = int(len(X_train_full) * sample_fraction)

np.random.seed(42)  # Set seed for reproducibility

# Generate random indices to select a subset of the training set
sample_indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)

# Sample from both X_train_full and y_train_full using the same indices
X_train_sampled = X_train_full[sample_indices]
y_train_sampled = y_train_full[sample_indices]

check_gpu_usage()

print(f'Shape of X_train_sampled: {X_train_sampled.shape}')

#Prepare Datasets for CatBoost
train_pool = Pool(data=X_train_sampled, label=y_train_sampled)
valid_pool = Pool(data=X_valid, label=y_valid)

#Set Parameters for CatBoost
catboost_params = {
    'iterations': 100,                   # Number of boosting iterations
    'learning_rate': 0.01,               # Learning rate
    'depth': 4,                          # Depth of each tree
    'loss_function': 'MultiClass',       # Multiclass classification
    'eval_metric': 'MultiClass',         # Evaluation metric
    'task_type': 'GPU',                  # Use GPU if available
    'verbose': 50,                       # Log training every 50 iterations
    'early_stopping_rounds': 20       # Stop if there's no improvement for 20 rounds
}

#Train the Model Using CatBoost
model = CatBoostClassifier(**catboost_params)

model.fit(
    train_pool,
    eval_set=valid_pool,
    early_stopping_rounds=20,  # Stop if validation doesn't improve after 20 rounds
    verbose=50                 # Log every 50 iterations
)
check_gpu_usage()

# Step 6: Evaluate the Model on Validation Set
y_valid_pred = model.predict(X_valid)
y_valid_pred = y_valid_pred.flatten()

print("Baseline Model Evaluation:")
print(classification_report(y_valid, y_valid_pred, zero_division=0))  # zero_division=0 to handle missing classes
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))

# Step 7: Save the Model for Future Use
model.save_model('catboost_baseline_model.cbm')

# %% [markdown]
# ### Tuning

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Define Objective Function for Optuna
def objective(trial):
    # Suggest hyperparameters for tuning
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 6),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.05),
        'random_seed': 42,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'task_type': 'GPU',  # Use GPU if available
    }

    # Train CatBoost model
    train_pool = cb.Pool(X_train_sampled, label=y_train_sampled)
    valid_pool = cb.Pool(X_valid_sampled, label=y_valid_sampled)

    pruning_callback = CatBoostPruningCallback(trial, 'MultiClass')

    model = cb.CatBoostClassifier(**params, verbose=0)

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=20,
        use_best_model=True
    )

    # Calculate validation accuracy
    preds = model.predict(X_valid_sampled)
    accuracy = accuracy_score(y_valid_sampled, preds)

    return accuracy

# %% [code] {"jupyter":{"outputs_hidden":false}}
# #Stratified Split for Training and Validation
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, valid_idx = next(split.split(X_train_transformed, y_train))

# X_train_full = X_train_transformed[train_idx]
# y_train_full = y_train.iloc[train_idx].values.ravel()

# X_valid = X_train_transformed[valid_idx]
# y_valid = y_train.iloc[valid_idx].values.ravel()

# #Sample a Subset of Training Data
# sample_fraction = 0.20  # Define the fraction of data to sample (e.g., 20%)
# sample_size = int(len(X_train_full) * sample_fraction)
# sample_valid_size = int(len(X_valid) * sample_fraction)

# np.random.seed(42)  # Set seed for reproducibility

# # Generate random indices to select a subset of the training set
# sample_indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)
# sample_valid_indices = np.random.choice(len(X_valid), size=sample_valid_size, replace=False)
# # Sample from both X_train_full and y_train_full using the same indices
# X_train_sampled = X_train_full[sample_indices]
# y_train_sampled = y_train_full[sample_indices]
# # Sample from both X_valid and y_valid using the same indices
# X_valid_sampled = X_valid[sample_valid_indices]
# y_valid_sampled = y_valid[sample_valid_indices]

# #Run Optuna Study
# study = optuna.create_study(direction='maximize')  # Maximize accuracy
# study.optimize(objective, n_trials=10)  # Set the number of trials

# #Save the Best Parameters
# best_params = study.best_params
# print("Best Parameters:", best_params)

# #To speed up, The parameters are manually selected from one of trials

# %% [code] {"jupyter":{"outputs_hidden":false}}
best_params = {'iterations': 158, 'depth': 6, 'learning_rate': 0.0031062341322339064}
print("Best Parameters:", best_params)

#Stratified Split for Training and Validation
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(split.split(X_train_transformed, y_train))

X_train_full = X_train_transformed[train_idx]
y_train_full = y_train.iloc[train_idx].values.ravel()

X_valid = X_train_transformed[valid_idx]
y_valid = y_train.iloc[valid_idx].values.ravel()

#Sample a Subset of Training Data
sample_fraction = 0.5  # Define the fraction of data to sample (e.g., 50%)
sample_size = int(len(X_train_full) * sample_fraction)
#sample_valid_size = int(len(X_valid) * sample_fraction)

np.random.seed(42)  # Set seed for reproducibility

# Generate random indices to select a subset of the training set
sample_indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)
#sample_valid_indices = np.random.choice(len(X_valid), size=sample_valid_size, replace=False)
# Sample from both X_train_full and y_train_full using the same indices
X_train_sampled = X_train_full[sample_indices]
y_train_sampled = y_train_full[sample_indices]
# Sample from both X_valid and y_valid using the same indices
X_valid_sampled = X_valid
y_valid_sampled = y_valid

#Train CatBoost with Best Parameters
best_model = cb.CatBoostClassifier(
    **best_params,
    random_seed=42,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    task_type='GPU',
    verbose=50
)

# Train on the full training data
train_pool = cb.Pool(X_train_sampled, label=y_train_sampled)
valid_pool = cb.Pool(X_valid_sampled, label=y_valid_sampled)
best_model.fit(
    train_pool,
    eval_set=valid_pool,
    early_stopping_rounds=20,
    use_best_model=True
)

evals_result = best_model.get_evals_result()

# Call the function to plot the learning curve
plot_learning_curve(evals_result, metric='MultiClass')

# %% [code] {"jupyter":{"outputs_hidden":false}}
with open('best_catboost_params.json', 'w') as f:
    json.dump(best_params, f)

best_model.save_model('best_catboost_model.cbm')

# %% [markdown]
# ### Evaluation

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Test Set 1: First Test Set
y_test_pred_1 = best_model.predict(X_test_transformed)
print("\nModel Evaluation on Test Set 1:")
print(classification_report(y_test, y_test_pred_1, zero_division=0))
print("Confusion Matrix for Test Set 1:")
cm_test_1 = confusion_matrix(y_test, y_test_pred_1)
print(cm_test_1)

# Test Set 2: Second Test Set
y_test_pred_2 = best_model.predict(X_test2_transformed)
print("\nModel Evaluation on Test Set 2:")
print(classification_report(y_test2, y_test_pred_2, zero_division=0))
print("Confusion Matrix for Test Set 2:")
cm_test_2 = confusion_matrix(y_test2, y_test_pred_2)
print(cm_test_2)

# Plot confusion matrices for validation and both test sets
# plot_confusion_matrix(cm_test_1, title='Confusion Matrix - Test Set 1')
# plot_confusion_matrix(cm_test_2, title='Confusion Matrix - Test Set 2')

# %% [markdown]
# ## Fast-Forward Neural Network

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Stratified Split for Training and Validation
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(split.split(X_train_transformed, y_train))

X_train_full = X_train_transformed[train_idx]
y_train_full = y_train.iloc[train_idx].values.ravel()

X_valid = X_train_transformed[valid_idx]
y_valid = y_train.iloc[valid_idx].values.ravel()

# Define the sample size (e.g., 60% of the training data)
sample_fraction = 0.6
sample_size = int(len(X_train_full) * sample_fraction)

np.random.seed(42)  # Set seed for reproducibility

# Sample from the training set
sample_indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)
X_train_sampled = X_train_full[sample_indices]
y_train_sampled = y_train_full[sample_indices]

# %% [markdown]
# ### Baseline

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Define Baseline Model Architecture
baseline_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_sampled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1000, activation='softmax')
])

#Compile the Baseline Model
baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

#Implement Early Stopping for Baseline Model
early_stopping_baseline = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#Train the Baseline Model with Early Stopping
baseline_history = baseline_model.fit(
    X_train_sampled,
    y_train_sampled,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping_baseline],
    verbose=1
)

# Save the entire model
baseline_model.save('baseline_model.h5')

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the Baseline Model
y_valid_pred_baseline = np.argmax(baseline_model.predict(X_valid), axis=1)

print("Baseline Model Evaluation:")
print(classification_report(y_valid, y_valid_pred_baseline, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred_baseline))

#Save the Model for Future Use
baseline_model.save('baseline_model.keras')

# %% [markdown]
# ### Tuned (Best)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define Final Model Architecture with Hyperparameter Tuning
def build_model(learning_rate=0.001, num_neurons=32, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(X_train_full.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(num_neurons/2), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1000, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# # Wrap model for Keras RandomizedSearch
# final_model = KerasClassifier(
#     build_fn=build_model,
#     epochs=50,
#     batch_size=64,
#     verbose=0,
#     learning_rate=0.001,
#     num_neurons=32,
#     dropout_rate=0.2
# )

# #Hyperparameter Tuning using RandomizedSearchCV
# param_dist = {
#     'learning_rate': [0.01, 0.1],  # Learning rate
#     'num_neurons': [16, 32],        # Number of neurons in first layer
#     'dropout_rate': [0.2, 0.3],     # Dropout rate to prevent overfitting
#     'batch_size': [64, 128],        # Batch size for training
#     'epochs': [10, 20]              # Number of epochs for training
# }

# random_search = RandomizedSearchCV(
#     estimator=final_model,
#     param_distributions=param_dist,
#     n_iter=10,
#     cv=3,
#     random_state=42
# )

# random_search.fit(X_train_full, y_train_full)

# # Get the Best Hyperparameters
# best_params = random_search.best_params_
# print("Best Parameters for Final Model:", best_params)

# # Get the best parameters from the RandomizedSearch
# best_params = random_search.best_params_

# # Save the best parameters to a JSON file
# with open('best_params.json', 'w') as f:
#     json.dump(best_params, f)

# %% [code] {"jupyter":{"outputs_hidden":false}}
best_params = {'learning_rate': 0.01, 'num_neurons': 32, 'dropout_rate': 0.2, 'epochs': 10, 'batch_size': 128}
print(best_params)

# Train Final Model with Best Hyperparameters
final_model_best = build_model(
    learning_rate=best_params['learning_rate'],
    num_neurons=best_params['num_neurons'],
    dropout_rate=best_params['dropout_rate']
)

# Implement Early Stopping for the Final Model
early_stopping_final = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the final model with the best parameters
final_history = final_model_best.fit(
    X_train_full,
    y_train_full,
    validation_data=(X_valid, y_valid),
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping_final],
    verbose=1
)

#Save the Model for Future Use
final_model_best.save('final_model_best.h5')
final_model_best.save('final_model_best.keras')

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the final model
y_valid_pred_final = np.argmax(final_model_best.predict(X_valid), axis=1)

print("final Model Evaluation:")
print(classification_report(y_valid, y_valid_pred_final, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred_final))

# %% [markdown]
# ### Custom

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Custom Model Architecture
custom_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_sampled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1000, activation='softmax')
])

#Compile the Baseline Model
custom_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

#Implement Early Stopping for Baseline Model
early_stopping_custom = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
#Train the Baseline Model with Early Stopping
custom_history = custom_model.fit(
    X_train_full,
    y_train_full,
    validation_data=(X_valid, y_valid),
    epochs=10,
    batch_size=128,
    callbacks=[early_stopping_baseline],
    verbose=1
)

#Save the Model for Future Use
custom_model.save('custom_model.h5')

#Save the Model for Future Use
custom_model.save('custom_model.keras')

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the custom model
y_valid_pred_custom = np.argmax(custom_model.predict(X_valid), axis=1)

print("Custom Model Evaluation:")
print(classification_report(y_valid, y_valid_pred_custom, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred_custom))

# %% [markdown]
# ### Learning Curve Analysis

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Call the function to plot the learning curve
plot_learning_curve_2(baseline_history, title='Baseline Model')
plot_learning_curve_2(final_history, title='Final Model')
plot_learning_curve_2(custom_history, title='Customed Model')

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the baseline model on test set 1
y_test_pred_baseline = np.argmax(baseline_model.predict(X_test_transformed), axis=1)

print("baselineModel Evaluation on test set1:")
print(classification_report(y_test, y_test_pred_baseline, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_baseline))

#Evaluate the baseline model on test set 2
y_test2_pred_baseline = np.argmax(baseline_model.predict(X_test2_transformed), axis=1)

print("baseline Model Evaluation on test set1:")
print(classification_report(y_test2, y_test2_pred_baseline, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test2, y_test2_pred_baseline))

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the final model on test set 1
y_test_pred_final = np.argmax(final_model_best.predict(X_test_transformed), axis=1)

print("final Model Evaluation on test set1:")
print(classification_report(y_test, y_test_pred_final, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_final))

#Evaluate the final model on test set 2
y_test2_pred_final = np.argmax(final_model_best.predict(X_test2_transformed), axis=1)

print("final Model Evaluation on test set2:")
print(classification_report(y_test2, y_test2_pred_final, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test2, y_test2_pred_final))

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Evaluate the custom model on test set 1
y_test_pred_custom = np.argmax(custom_model.predict(X_test_transformed), axis=1)

print("Custom Model Evaluation on test set1:")
print(classification_report(y_test, y_test_pred_custom, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_custom))

#Evaluate the custom model on test set 2
y_test2_pred_custom = np.argmax(custom_model.predict(X_test2_transformed), axis=1)

print("Custom Model Evaluation on test set2:")
print(classification_report(y_test2, y_test2_pred_custom, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test2, y_test2_pred_custom))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Save the transformed test set 1
np.save('y_test1_pred_custom.npy', y_test_pred_custom)
np.save('y_test1_pred_final.npy', y_test_pred_final)
np.save('y_test1_pred_baseline.npy', y_test_pred_baseline)

# Save the transformed test set 2
np.save('y_test2_pred_custom.npy', y_test2_pred_custom)
np.save('y_test2_pred_final.npy', y_test2_pred_final)
np.save('y_test2_pred_baseline.npy', y_test2_pred_baseline)

# %% [markdown]
# ## Performance Gap Analysis

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Convert the reports into DataFrames for easier comparison
report1 = classification_report(y_test, y_test_pred_custom, zero_division=0, output_dict=True)
report2 = classification_report(y_test2, y_test2_pred_custom, zero_division=0, output_dict=True)
df_report1 = pd.DataFrame(report1).transpose()
df_report2 = pd.DataFrame(report2).transpose()

#rename cols to be able to perform comparisons
df_report2_renamed = df_report2.rename(columns={
    'precision': 'precision_set2',
    'recall': 'recall_set2',
    'f1-score': 'f1-score_set2'
})

combined_df = pd.concat([df_report1[['precision', 'recall', 'f1-score']],
                          df_report2_renamed[['precision_set2', 'recall_set2', 'f1-score_set2']]],
                          axis=1)
# Calculate the differences
combined_df['precision_diff'] = combined_df['precision'] - combined_df['precision_set2']
combined_df['recall_diff'] = combined_df['recall'] - combined_df['recall_set2']
combined_df['f1-score_diff'] = combined_df['f1-score'] - combined_df['f1-score_set2']

#select a threshold for difference in lable classification
threshold = 0.6

mask = (combined_df['precision_diff'].abs() > threshold) | \
       (combined_df['recall_diff'].abs() > threshold) | \
       (combined_df['f1-score_diff'].abs() > threshold)

# Filter the DataFrame using the mask
significant_diff_df = combined_df[mask]

print(significant_diff_df)

print(significant_diff_df.info())

# %% [markdown]
# ### Visualisation

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Adjust figure size and DPI for higher resolution
plt.figure(figsize=(12, 8))  # Increase width to 24 and height to 8 for a larger plot

# Create the bar plot for precision
sns.barplot(data=significant_diff_df, x=significant_diff_df.index, y='precision_diff', color='lightblue', alpha=0.6, label='Precision difference')

# Annotate each bar with the label only, using larger font size
for index, value in enumerate(significant_diff_df['precision_diff']):
    plt.text(x=index, y=value + 0.02, s=significant_diff_df.index[index], ha='center', va='bottom', fontsize=12)

# Set axis labels with larger font size
plt.xlabel("Label", fontsize=16)
plt.ylabel("Precision Difference", fontsize=16)

# Rotate and increase the size of x-axis tick labels
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Increase legend font size
plt.legend(fontsize=14)

# Show the plot with adjusted layout to fit text properly
plt.tight_layout()
plt.savefig("precision_difference_plot.png", dpi=600, bbox_inches="tight")  # Save as PNG with 600 DPI
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Adjust figure size to accommodate the data
plt.figure(figsize=(15, 8))

# Set width of bars
bar_width = 0.4

# Get the position of the bars on the x-axis
indices = np.arange(len(significant_diff_df))

# Create bar plot for precision from the first set
plt.bar(indices - bar_width / 2, significant_diff_df['precision'], width=bar_width, color='lightblue', alpha=0.6, label='Precision Set 1')

# Create bar plot for precision from the second set
plt.bar(indices + bar_width / 2, significant_diff_df['precision_set2'], width=bar_width, color='lightgreen', alpha=0.6, label='Precision Set 2')

# Annotate each bar for the first set
for index, value in enumerate(significant_diff_df['precision']):
    plt.text(x=index - bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=14)  # Increased fontsize to 12

# Annotate each bar for the second set
for index, value in enumerate(significant_diff_df['precision_set2']):
    plt.text(x=index + bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=14)  # Increased fontsize to 12

# Rotate the x-axis labels for better readability and increase fontsize
plt.xticks(indices, significant_diff_df.index, rotation=45, ha='right', fontsize=14)

# Adding labels, title, and legend with increased font size
plt.xlabel('Labels', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Comparison of Precision Between Two Sets', fontsize=18)
plt.legend(fontsize=14)

# Display the plot with tighter layout
plt.tight_layout()
plt.savefig("comparison_of_precision2.png", dpi=600, bbox_inches="tight")  # Save as PNG with 600 DPI
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Adjust figure size to accommodate the data
plt.figure(figsize=(14, 8))

# Set width of bars
bar_width = 0.4

# Get the position of the bars on the x-axis
indices = np.arange(len(significant_diff_df))

# Create bar plot for recall from the first set
plt.bar(indices - bar_width / 2, significant_diff_df['recall'], width=bar_width, color='lightblue', alpha=0.6, label='Recall Set 1')

# Create bar plot for recall from the second set
plt.bar(indices + bar_width / 2, significant_diff_df['recall_set2'], width=bar_width, color='lightgreen', alpha=0.6, label='Recall Set 2')

# Annotate each bar for the first set (recall)
for index, value in enumerate(significant_diff_df['recall']):
    plt.text(x=index - bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=11)

# Annotate each bar for the second set (recall_set2)
for index, value in enumerate(significant_diff_df['recall_set2']):
    plt.text(x=index + bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=11)

# Rotate the x-axis labels for better readability
plt.xticks(indices, significant_diff_df.index, rotation=45, ha='right')

# Adding labels, title, and legend
plt.xlabel('Labels')
plt.ylabel('Recall')
plt.title('Comparison of Recall Between Two Sets')
plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig("recall_difference_plot.png", dpi=600, bbox_inches="tight")  # Save as PNG with 300 DPI
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Adjust figure size to accommodate the data
plt.figure(figsize=(15, 10))

# Set width of bars
bar_width = 0.4

# Get the position of the bars on the x-axis
indices = np.arange(len(significant_diff_df))

# Create bar plot for precision_diff
plt.bar(indices - bar_width / 2, significant_diff_df['precision_diff'], width=bar_width, color='lightblue', alpha=0.6, label='Precision Difference')

# Create bar plot for recall_diff
plt.bar(indices + bar_width / 2, significant_diff_df['recall_diff'], width=bar_width, color='lightgreen', alpha=0.6, label='Recall Difference')

# Annotate each bar for precision_diff
for index, value in enumerate(significant_diff_df['precision_diff']):
    plt.text(x=index - bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=13)

# Annotate each bar for recall_diff
for index, value in enumerate(significant_diff_df['recall_diff']):
    plt.text(x=index + bar_width / 2, y=value + 0.02, s=f'{value:.2f}', ha='center', va='bottom', fontsize=13)

# Rotate the x-axis labels for better readability
plt.xticks(indices, significant_diff_df.index, rotation=45, ha='right')

# Adding labels, title, and legend with increased font sizes
plt.xlabel('Labels', fontsize=16)  # Increased font size for x-label
plt.ylabel('Difference', fontsize=16)  # Increased font size for y-label
plt.title('Comparison of Precision and Recall Differences Between Two Sets', fontsize=20)  # Increased font size for title
plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig("precision_recall_difference_plot.png", dpi=300, bbox_inches="tight")  # Save as PNG with 300 DPI
plt.show()

# %% [markdown]
# ### Track Mis-Classifications

# %% [code] {"jupyter":{"outputs_hidden":false}}
#load in transformed datasets
# x_test_transformed = np.load("X_test_transformed.npy")
# x_test2_transformed = np.load("X_test2_transformed.npy")

# Load the model from the .keras file
FNN_model = custom_model

prob_val_1 = FNN_model.predict(X_test_transformed)
prob_val_2 = FNN_model.predict(X_test2_transformed)

# Create a DataFrame with true label
df_probs_val_2 = pd.DataFrame(prob_val_2)
df_probs_val_2['true_label'] = y_test2

df_probs_val_1 = pd.DataFrame(prob_val_1)
df_probs_val_1['true_label'] = y_test

#filter rows based on performance
df_label_414_val_1 = df_probs_val_1[df_probs_val_1['true_label'] == 414]
df_label_414_val_2 = df_probs_val_2[df_probs_val_2['true_label'] == 414]

# Filter rows for true label 497
df_label_497_val_1 = df_probs_val_1[df_probs_val_1['true_label'] == 497]
df_label_497_val_2 = df_probs_val_2[df_probs_val_2['true_label'] == 497]

# Filter rows for true label 204
df_label_204_val_1 = df_probs_val_1[df_probs_val_1['true_label'] == 204]
df_label_204_val_2 = df_probs_val_2[df_probs_val_2['true_label'] == 204]

# Filter rows for true label 982
df_label_982_val_1 = df_probs_val_1[df_probs_val_1['true_label'] == 982]
df_label_982_val_2 = df_probs_val_2[df_probs_val_2['true_label'] == 982]

# %% [code] {"jupyter":{"outputs_hidden":false}}
#analyzing which rows were very hard to classify
df_label_414_val_1 = add_max_prob_and_gap(df_label_414_val_1, 414)
df_label_414_val_2 = add_max_prob_and_gap(df_label_414_val_2, 414)
df_label_497_val_1 = add_max_prob_and_gap(df_label_497_val_1, 497)
df_label_497_val_2 = add_max_prob_and_gap(df_label_497_val_2, 497)
df_label_204_val_1 = add_max_prob_and_gap(df_label_204_val_1, 204)
df_label_204_val_2 = add_max_prob_and_gap(df_label_204_val_2, 204)
df_label_982_val_1 = add_max_prob_and_gap(df_label_982_val_1, 982)
df_label_982_val_2 = add_max_prob_and_gap(df_label_982_val_2, 982)

print(df_label_414_val_2[['max_prob', 'prob_gap', "pred_label", "true label prob"]])

print(df_label_204_val_2[['max_prob', 'prob_gap', "pred_label", "true label prob"]])

print(df_label_204_val_1[['max_prob', 'prob_gap', "pred_label", "true label prob"]])

# %% [markdown]
# ### Feature Analysis Between Test Sets

# %% [code] {"jupyter":{"outputs_hidden":false}}
#Sample a Subset of Training Data
sample_fraction = 0.01  # Define the fraction of data to sample (e.g., 0.5%)
sample_size = int(len(X_train_transformed) * sample_fraction)

np.random.seed(42)  # Set seed for reproducibility

# Generate random indices to select a subset of the training set
sample_indices = np.random.choice(len(X_train_transformed), size=sample_size, replace=False)

# Sample from both X_train_full and y_train_full using the same indices
X_train_sampled = X_train_transformed[sample_indices]

explainer = shap.DeepExplainer(FNN_model, X_train_sampled)

shap_values = explainer.shap_values(X_test2_transformed[4140:4150])

print(shap_values)

# %% [markdown]
# ## Clustering Analysis on Test Sets

# %% [code] {"jupyter":{"outputs_hidden":false}}
df_test_1 = pd.DataFrame(X_test_transformed)

# Fit MiniBatchKMeans (assuming 2 clusters)
minibatch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
labels_1 = minibatch_kmeans.fit_predict(df_test_1)

df_test_1_labels = pd.DataFrame(data={'Cluster Lables':labels_1})
df_test_1_labels.value_counts(normalize=True).plot(kind='bar')
plt.title('Test Set 1 - Plot of Cluster Labels')
plt.ylabel('Percentage')
plt.show()

df_test_2 = pd.DataFrame(X_test2_transformed)

# Fit MiniBatchKMeans (assuming 2 clusters)
minibatch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
labels_2 = minibatch_kmeans.fit_predict(df_test_2)

df_test_2_labels = pd.DataFrame(data={'Cluster Lables':labels_2})
df_test_2_labels.value_counts(normalize=True).plot(kind='bar')
plt.ylabel('Percentage')
plt.title('Test Set 2 - Plot of Cluster Labels')
plt.show()

# %% [markdown]
# # End
