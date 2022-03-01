import os
import dotenv
import wandb
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from dretino.dataloader.build_features import DRDataModule
from dretino.models.predict_model import test
from dretino.models.train_model import Model, train
from dretino.visualization.visualize import show_images, cal_mean, plot_metrics
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    PATH = '../data/processed/'
    dfx = pd.read_csv(PATH + '2.Groundtruths/a.IDRiD_Disease_Grading_Training_Labels.csv')
    df_test = pd.read_csv(PATH + '2.Groundtruths/b.IDRiD_Disease_Grading_Testing_Labels.csv')
    df_train, df_valid = train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx['Retinopathy grade'].values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_transforms = A.Compose(
        [
            A.Resize(width=250, height=250),
            A.RandomCrop(height=224, width=224),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.Blur(p=0.3),
            # A.CLAHE(p=0.3),
            # A.ColorJitter(p=0.3),
            # A.Affine(shear=30, rotate=0, p=0.2),
            A.Normalize(
                mean=(0.5237, 0.2542, 0.0853),
                std=(0.2649, 0.1497, 0.0876),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=(0.5237, 0.2542, 0.0853),
                std=(0.2649, 0.1497, 0.0876),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=(0.5237, 0.2542, 0.0853),
                std=(0.2649, 0.1497, 0.0876),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    dm = DRDataModule(df_train, df_valid, df_test,
                      train_path=PATH + 'images_resized',
                      valid_path=PATH + 'images_resized',
                      test_path=PATH + 'test_images_resized',
                      train_transforms=train_transforms,
                      val_transforms=val_transforms,
                      test_transforms=test_transforms,
                      num_workers=4,
                      batch_size=16)

    args = dict(
        model_name='resnet50d',
        num_neurons=512,
        num_layers=2,
        dropout=0.2,
        lr=3e-4,
        loss='mse',
        epochs=5,
        gpus=0,
        project='DRD',
        additional_layers=False
    )

    wab = False
    fast_dev_run = True
    overfit_batches = False

    wandb.login(key=os.getenv('WANDB'))

    file_name, trainer = train(Model, dm,
                               wab=wab,
                               fast_dev_run=fast_dev_run,
                               overfit_batches=overfit_batches,
                               **args)
    if not fast_dev_run:
        plot_metrics('../reports/csv_logs/' + file_name)

        test(Model, dm,
             file_name,
             trainer,
             wab=wab,
             fast_dev_run=fast_dev_run,
             overfit_batches=overfit_batches)
