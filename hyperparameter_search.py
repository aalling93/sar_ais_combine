import argparse
import keras_tuner as kt
import numpy as np
from .src.drone_em_dl.models import *
from .src.drone_em_dl.models import Fae
from .src.drone_em_dl.data import *
from .src.drone_em_dl.data import Data
from .src.drone_em_dl.hyperparameter.model_builder import model_builder
#python3 hyperparameter_search.py -
parser = argparse.ArgumentParser(description="Finding Hyperparameters FAE  model")
parser.add_argument("-epoch", "--epochs", help="Epoch integer input.", default=100, type=int)
parser.add_argument("-batchSize", "--batchSize", help="Batch size integer input.", default=50, type=int)
parser.add_argument("-seed", "--seed", help="seed value", default=42, type=int)
parser.add_argument("-data", "--data", help="relative path to data", default='../data/raw/falster_data_Kristian.csv', type=str)
parser.add_argument("-data_features", "--date_features", help="list of int. features", default=[1,2,3,4,5,6,7,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], type=list)
parser.add_argument("-data_split", "--data_split", help="split percentage (train)", default=0.8, type=float)
parser.add_argument("-verbose", "--verbose", help="verbose", default=1, type=int)
parser.add_argument("-GPU", "--GPU", help="which GPU", default=0, type=int)
parser.add_argument("-GPU_memory", "--GPU_memory", help="GPU memory", default=60000, type=int)
parser.add_argument("-hyperturner_name", "--hyperturner_name", help="hyperturner_name", default=0, type=int)
args = parser.parse_args()

print(
    f"\nTraining models with the following parms: \nEpochs: {args.epochs} \nBatch size: {args.batchSize} \nSeed: {args.seed} "
)
strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))

np.random.seed(args.seed)

#### Original Data
if args.verbose > 0:
    print("\nloading data")


data = Data()
data.load_data(args.data)
data.get_features([1,2,3,4,5,6,7,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]) 
data.train_test_split(split=args.data_split)
data.norm_data()

if args.verbose > 0:
    print("\nData loaded")

# del train
tuner = kt.Hyperband(
    hypermodel=model_builder,
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=50,
    hyperband_iterations = 7,
    project_name="hyperband_tuner",
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=11)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.2, patience=5, min_lr=0.00000000001
)
callbacks = [stop_early, reduce_lr]


tuner.search(
    data.norm_data.norm_train,
    data.norm_data.norm_train,
    batch_size=args.batchSize,
    epochs=args.epochs,
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=True,
    verbose=args.verbose,
)
