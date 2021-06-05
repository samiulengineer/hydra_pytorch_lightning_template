#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021
import logging

import hydra                                                                        # hydra 
from omegaconf import DictConfig, OmegaConf                                         # dirctconfig and omegaconfig 
import pytorch_lightning as pl                                                      # pytorch lighinging 

log = logging.getLogger(__name__)                                                   # logger


@hydra.main(config_path='config', config_name='train_config')                       # hydra decoretor pointing to folder config and traing_config.yaml file 
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------
    data_module = hydra.utils.instantiate(cfg.data)                                 # creating a instance of mnist_data_module.py 


    # ------------
    # model
    # ------------
    model = hydra.utils.instantiate(cfg.model)                                       # creating instance of dncnn.py 

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(**(cfg.pl_trainer), checkpoint_callback=True)               # creating a instance of Trainer class with gpu = 1
    log.info('run training...')                                                      # store the logging info fo Trainer class
    train_dataloader = data_module.train_dataloader()                                # calling the train dataloader from 
    val_dataloader = data_module.val_dataloader()
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])  # start the training 


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
