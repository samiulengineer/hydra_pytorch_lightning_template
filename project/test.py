#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : test.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021
  
import logging

import hydra                                                                        # hydar 
from omegaconf import DictConfig, OmegaConf                                         # omegaconf and Dictconfig are used for the control of the yaml file 
import pytorch_lightning as pl                                                      # pytorch lightning 
import torch                                                                        # pytorch
import json

log = logging.getLogger(__name__)                                                   # to store the logg info 

#hydar incitialization with with hydar decorator .. pointing to the config folder and test_config.yaml file 
@hydra.main(config_path='config', config_name='test_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)                                                    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition, sets the following environment variables:

    # ------------
    # data
    # ------------
    data_module = hydra.utils.instantiate(cfg.data)                                  # create an instance of class 
    test_dataloader = data_module.test_dataloader()                                  # calling the test_dataloader function  from class to lead data 

    # ------------
    # model
    # ------------
    model = hydra.utils.instantiate(cfg.model)                                       # create an instance of class  
    # model.load_from_checkpoint(cfg.checkpoint)
    model.load_state_dict(torch.load(cfg.checkpoint_path)['state_dict'])             # cfg.chackpoint_path = placeholder for pretrained checkpoint

    trainer = pl.Trainer(**(cfg.pl_trainer))                                         # start tanning  with Trainer function 

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=test_dataloader)                   # run test on trained model and massure the performence on test_dataloader 
    log.info(result)                                                                 # it store all the data of true positive ,  
    with open(f'{trainer.log_dir}/out', 'w') as f:                                   # open log file in log/run/out as write mode 
        f.write(json.dumps(str(result),indent=4))                                    # dumps log info 


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
