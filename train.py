import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
import hydra

from model import MInterface
from data import DInterface
from utils import setup_config
from pathlib import Path


def _num_devices(devices):
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    return 1


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
    trainer_kwargs = {**cfg.trainer, **callbacks}
    if _num_devices(cfg.trainer.devices) > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data_module, ckpt_path=cfg.experiment.resume_ckpt_path)


if __name__ == '__main__':
    main()
