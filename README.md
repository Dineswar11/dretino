# DRetino

<p align="center">
    <img src="https://res.cloudinary.com/grohealth/image/upload/$wpsize_!_cld_full!,w_1200,h_630,c_scale/v1588090981/Symptoms-of-Diabetic-Retinopathy.png">
</p>

# DRetino 

A python library to create supervised diabetic retinopathy detection neural nets for ordinal regression using different loss functions.
Dretino is build on pytorch lightning and contains four different losses CrossEntropy, MeanSquared, Coral, Corn



- [Quick-start](#quick-start)




## Quick-start

* <a href="" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


* <a href=""><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>


```pycon
from dretino.dataloader.build_features import DRDataModule
from dretino.models.train_model import Model, train

dm = DRDataModule(df_train, df_valid, df_test,
                  train_path,
                  valid_path,
                  test_path,
                  train_transforms,
                  val_transforms,
                  test_transforms,
                  num_workers=4,
                  batch_size=16)

args = dict(
        model_name='resnet50d',
        lr=3e-4,
        loss='mse',
        epochs=50,
        gpus=1,
        project='project_name',
        additional_layers=False
    )

train(Model,dm,**args)
```