import pytorch_lightning as pl
import hydra

from model import MInterface
from data import DInterface
from utils import setup_config
from pathlib import Path


def _maybe_seed(cfg):
    seed = getattr(cfg.experiment, 'seed', None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    _maybe_seed(cfg)
    callbacks = setup_config(cfg)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    data_module = DInterface(cfg.data)
    model = MInterface(cfg.model)

    trainer = pl.Trainer(**cfg.trainer,
                         **callbacks,)
    trainer.test(model, data_module, ckpt_path=cfg.experiment.ckpt_path)

if __name__ == '__main__':
    main()
