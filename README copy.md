
<img src="figs/test.png" width="750" align="center">


Kristian Aalling Sørensen
kaaso@space.dtu.dk


Tobias 


# Brief Description
<a class="anchor" id="intro"></a>

Her skal vi skrive en fancy beskrivelse


# Table Of Contents
<a class="anchor" id="content"></a>

-  [Introduction](#Introduction)
-  [Requirements](#Requirements)
-  [Install and Run](#Install-and-Run)
-  [Examples ](#use)
-  [EMI ](#emi)
-  [Acknowledgments](#Acknowledgments)



# Requirements
 <a class="anchor" id="Requirements"></a>

- See requirements filde..
# Install and Run
 <a class="anchor" id="Install-and-Run"></a>

Currently, only git is supported. Later pypi will be aded. So, clone the dir.




## Make reconstructions <a class="anchor" id="use"></a>
Go back to [Table of Content](#content)




1. Load module
------------

```python
from src.drone_em_dl import *
```


1. Load data
------------

```python
data = data.Data()
data.load_data('../data/raw/falster_data_Kristian.csv')
data.get_features([11,12,13,22,23,24,25,26,27,28,29,30,31]) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14]  #[1,2,3,4,5,6,7,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
data.train_test_split(split=0.8)
data.norm_data()

```

2. Make reconstructions
----------------
```python
#load model
model = tf.keras.models.load_model('../models/AE_falter_06_04_2023_15_1/best_model_AE_falter_06_04_2023_15_1.h5', custom_objects={'lr': get_lr_metric})

results = Reconstruction()
results.load_model(model)
results.get_reconstructions(data.norm_data.norm_train)

recionstructed_data = data.get_inv(results.reconstructions)
input_data  = data.get_inv(data.norm_data.norm_train)
latent_space = results.latent_space
```


3.Do PCA
```python
pcas = Pca()
pcas.load_data(data.norm_data.norm_train,data.norm_data.norm_test)
pcas.get_pca(pca_amount=8)
train_pcs = data.get_inv(pcas.pca_train_inv)
train_org = data.get_inv(data.norm_data.norm_train)
```




4. comapre results
```python
for i in range(13):
    plt.figure(figsize=(40,10))
    plt.subplot(1,3,1)
    plt.title(f'Reconstructed {data.test.columns[i]}')
    cm = plt.scatter(data.org_test.X,data.org_test.Y,c=recionstructed_data[:,i],cmap='jet',s=4)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title(f'True {data.test.columns[i]}')
    cm = plt.scatter(data.org_test.X,data.org_test.Y,c=input_data[:,i],cmap='jet',s=4)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title(f'PCA {data.test.columns[i]}')
    cm = plt.scatter(data.org_test.X,data.org_test.Y,c=train_pcs[:,i],cmap='jet',s=4)
    plt.colorbar()
    plt.show()
```



------------



## EMI <a class="anchor" id="emi"></a>
Go back to [Table of Content](#content)




# Acknowledgments
 <a class="anchor" id="Acknowledgments"></a>
Us

 # Licence
See License file. In short:

1. Cite us