from src.drone_em_dl.models import *
from src.drone_em_dl.models import Fae
from src.drone_em_dl.data import *
from src.drone_em_dl.data import Data

from sklearn.model_selection import KFold
import argparse
from datetime import datetime
import tensorflow as tf

from clearml import Task

#python3 train.py -data_features="23,25,27,29,31" -session_name='Qd_inputs' -notes="This is a training run for AE on Qd data." -GPU_memory=5000
parser = argparse.ArgumentParser(description="Training drone em dl")
parser.add_argument("-epoch", "--epochs", help="Epoch integer input.", default=200, type=int)
parser.add_argument("-batchSize", "--batchSize", help="Batch size integer input.", default=50, type=int)
parser.add_argument("-seed", "--seed", help="seed value", default=42, type=int)
parser.add_argument( "-data",  "--data",  help="relative path to data",  default="data/raw/falster_data_Kristian.csv",  type=str,)
parser.add_argument(  "-data_features",  "--data_features",  help="list of int. features", default="0,1,2,3,6,11,12,13,20,22,23,24,25,26,27,28,29,30,31",  type=str)
parser.add_argument(  "-data_split",  "--data_split",  help="split percentage (train)",  default=0.8,  type=float,)
parser.add_argument("-verbose", "--verbose", help="verbose", default=1, type=int)
parser.add_argument("-GPU", "--GPU", help="which GPU", default=0, type=int)
parser.add_argument( "-GPU_memory", "--GPU_memory", help="GPU memory", default=20000, type=int)
parser.add_argument(  "-hyperturner_name",  "--hyperturner_name",  help="hyperturner_name",  default=0,  type=int,)
parser.add_argument(  "-kfolds",  "--kfolds",  help="Amount of K-folds for CV. (K models will be made)",  default=5,  type=int,)
parser.add_argument(  "-latent_space_dim",  "--latent_space_dim",  help="Latens space dimension for model.",  default=15,  type=int,)
parser.add_argument(  "-neurons", "--neurons", help="neurons", default=[210, 160, 360, 60, 310], type=list)
parser.add_argument(  "-model_folder",  "--model_folder",  help="Folder for model",  default="models/cv",)
parser.add_argument(  "-dropout_prob", "--dropout_prob", help="dropout_prob", default=0.1, type=float)
parser.add_argument( "-session_name",  "--session_name",  help="your name to clear ml",  default="all data",  type=str,)
parser.add_argument( "-notes",  "--notes",  help="notes",  default="NA",  type=str,)
args = parser.parse_args()


if args.verbose > 0:
    print(
        f"\nTraining models with the following parms: \nEpochs: {args.epochs} \nBatch size: {args.batchSize} \nSeed: {args.seed} \nKfolds: {args.kfolds} \nLatens space dim: {args.latent_space_dim}\n"
    )


strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))

#### Original Data
if args.verbose > 0:
    print("\nloading data")
data = Data()
data.load_data(args.data)
data_features = [int(item) for item in args.data_features.split(',')]
data.get_features(data_features)
data.train_test_split(split=args.data_split)
data.norm_data()


strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))
np.random.seed(args.seed)


cvscores = []
kfold = KFold(n_splits=args.kfolds, random_state=None, shuffle=False)
name_append = datetime.now().strftime("%d_%m_%Y_%H")
i = 1
if args.verbose > 0:
    print("\nPreparing K-folds")
    print(f"\nTime: {name_append}")

os.makedirs(f"{args.model_folder}", exist_ok=True)

for train_idx, val_idx in kfold.split(
    data.norm_data.norm_train, data.norm_data.norm_train
):
    tf.keras.backend.clear_session()
    if args.verbose > 0:
        print("training_model ", i)
    # making model for split i
    model_name = "AE_falter" + "_" + name_append + "_" + str(i)
    task = Task.init(project_name="Drone_em_dl", task_name=f"Drone_em_dl_{args.session_name}_{model_name}")

    with Fae() as ae:
        ae = ae.make_model(
            input_size=data.norm_data.norm_train[0].shape,
            latent_space_dim=args.latent_space_dim,
            dense_neurons=args.neurons,
            dropout_prob=args.dropout_prob,
            name=model_name,
        )

    callbacks = get_callbacks(ae)
    model_fit(
        ae,
        data.norm_data.norm_train[train_idx],
        data.norm_data.norm_train[train_idx],
        batch_size=args.batchSize,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks,
    )

    i = i + 1
