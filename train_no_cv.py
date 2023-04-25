from src.drone_em_dl.models import *
from src.drone_em_dl.models import Fae
from src.drone_em_dl.data import *
from src.drone_em_dl.data import Data
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
from clearml import Dataset
from clearml import Task
from sys import platform
#python3 train_no_cv.py -data_features="11,12,13,22,23,24,25,26,27,28,29,30,31" -session_name='Qd_Ip_inputs_pitch_roll_head_2' -notes="This is a training run for AE on Ip, Qd and Pitch Roll Heading." -GPU_memory=5000
#11,12,13,22,23,24,25,26,27,28,29,30,31
parser = argparse.ArgumentParser(description="Training drone em dl")
parser.add_argument("-epoch", "--epochs", help="Epoch integer input.", default=1000, type=int)
parser.add_argument("-batchSize", "--batchSize", help="Batch size integer input.", default=50, type=int)
parser.add_argument("-seed", "--seed", help="seed value", default=42, type=int)
parser.add_argument( "-data",  "--data",  help="relative path to data",  default="data/raw/falster_data_Kristian.csv",  type=str,)
parser.add_argument(  "-data_features",  "--data_features",  help="list of int. features", default="0,1,2,3,6,11,12,13,20,22,23,24,25,26,27,28,29,30,31",  type=str)
parser.add_argument(  "-data_split",  "--data_split",  help="split percentage (train)",  default=0.97,  type=float,)
parser.add_argument("-verbose", "--verbose", help="verbose", default=1, type=int)
parser.add_argument("-GPU", "--GPU", help="which GPU", default=0, type=int)
parser.add_argument( "-GPU_memory", "--GPU_memory", help="GPU memory", default=20000, type=int)
parser.add_argument(  "-hyperturner_name",  "--hyperturner_name",  help="hyperturner_name",  default=0,  type=int,)
parser.add_argument(  "-latent_space_dim",  "--latent_space_dim",  help="Latens space dimension for model.",  default=15,  type=int,)
parser.add_argument(  "-neurons", "--neurons", help="neurons", default=[210, 160, 360, 60, 310], type=list)
parser.add_argument(  "-model_folder",  "--model_folder",  help="Folder for model",  default="models/cv",)
parser.add_argument(  "-dropout_prob", "--dropout_prob", help="dropout_prob", default=0.1, type=float)
parser.add_argument( "-session_name",  "--session_name",  help="your name to clear ml",  default="all_data",  type=str,)
parser.add_argument( "-notes",  "--notes",  help="notes",  default="NA",  type=str,)
parser.add_argument( "-upload_data",  "--upload_data",  help="upload_data",  default=False,  type=bool)
args = parser.parse_args()


name_append = datetime.now().strftime("%d_%m_%Y_%H")
print(name_append)
model_name = f"{args.session_name}" + "_" + name_append 
task = Task.create(project_name="Drone_em_dl", task_name=f"Drone_em_dl_{args.session_name}_{model_name}")



if args.verbose > 0:
    print(
        f"\nTraining models with the following parms: \nEpochs: {args.epochs} \nBatch size: {args.batchSize} \nSeed: {args.seed}  \nLatens space dim: {args.latent_space_dim}\n"
    )

if platform == "linux" or platform == "linux2":
    # linux
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





if args.upload_data ==True:
    ds = Dataset.create(
    dataset_name = f'{args.session_name}',
    dataset_project = 'Drone_em_dl'
    )
    ds.add_files(args.data)
    for i in range(len(data.org_test.columns)):
        ds.get_logger().report_histogram(
            title = f"Histogram of train {data.org_test.columns[i]}",
            series = 'train data',
            values = data.org_test.iloc[:,i].values
        )
        fig1 = plt.figure()
        plt.scatter(data.org_test.X.values,data.org_test.Y.values, c = data.org_test.iloc[:,-i].values,cmap='jet',s=4)
        plt.title(f'{data.org_test.columns[-i]}\n')
        cbar = plt.colorbar()
        cbar.set_label(f'{data.org_test.columns[-i]}', rotation=270)
        plt.xlabel(('X'))
        plt.ylabel(('Y'))
        ds.get_logger().report_matplotlib_figure(
            title = f'Scatter: {data.org_test.columns[-i]}\n',
            series = 'training data',
            figure = fig1,
            report_image= True
        )

    ds.upload()
    ds.finalize()  





fig1 = plt.figure()
plt.scatter(data.org_test.X.values,data.org_test.Y.values, c = data.norm_data.norm_test[:,-1],cmap='jet',s=4)
plt.title(f'{data.org_test.columns[-1]}\n')
cbar = plt.colorbar()
cbar.set_label(f'{data.org_test.columns[-1]}', rotation=270)
plt.xlabel(('X'))
plt.ylabel(('Y'))

task.get_logger().report_matplotlib_figure(title='Debug Samples',
                                series='',
                                figure=fig1)





strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))
np.random.seed(args.seed)


task = Task.init(project_name="Drone_em_dl", task_name=f"Drone_em_dl_test2")
#task.connect(parameters,'hyperparameters')
task.connect(args,'args')




if args.verbose > 0:
    print(f"\nTime: {name_append}")

os.makedirs(f"{args.model_folder}", exist_ok=True)



with Fae() as ae:
    print(model_name)
    print('test')
    ae = ae.make_model(
            input_size=data.norm_data.norm_train[0].shape,
            latent_space_dim=args.latent_space_dim,
            dense_neurons=args.neurons,
            dropout_prob=args.dropout_prob,
            name=model_name,
        )
print(ae.name)
ae._name = model_name
callbacks = get_callbacks(ae)
model_fit(
        ae,
        data.norm_data.norm_train,
        data.norm_data.norm_train,
        batch_size=args.batchSize,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks,
)

