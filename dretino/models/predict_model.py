import os

import wandb


def predict(Model, dm, file_name, trainer, wab=False, fast_dev_run=False, overfit_batches=False):
    if fast_dev_run:
        return
    elif overfit_batches:
        return
    else:
        model = Model.load_from_checkpoint(os.path.join('../models', file_name) + '.ckpt')
        trainer.test(model, dm)

        if wab:
            wandb.finish()
