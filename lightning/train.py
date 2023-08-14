
import pytorch_lightning as pl
import config
from model import NN
from dataset import MnistDataModule

from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
torch.set_float32_matmul_precision("medium") # to make lightning happy


#### 디버깅을 할수 가 있네.. ㅋㅋㅋㅋ..

def main():    
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")
    
    model = NN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE)
    dm = MnistDataModule(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=3,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)    

if __name__ == "__main__":
    main()
