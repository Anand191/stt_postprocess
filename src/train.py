import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.model import chained_seq2seq
# from src.utils import STTDataModule
from src.data_modules_2 import STTDataModule2



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt_path = "../model/checkpoints/chained_seq2seq_logs/version_3/checkpoints/epoch=9-step=18150.ckpt"
def setup():
    dm = STTDataModule2(num_workers = 12)
    dm.setup("fit")
    dm.setup("test")

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    logger = TensorBoardLogger("../model/checkpoints", name="chained_seq2seq_logs", log_graph=False)
    model = chained_seq2seq()

    seed_everything(42)
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        logger=logger,
        devices=1 if torch.cuda.is_available() else None,
        default_root_dir="../model/checkpoints/"
    )
    return trainer, model, dm


if __name__ == "__main__":
    trainer, model, dm = setup()
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
