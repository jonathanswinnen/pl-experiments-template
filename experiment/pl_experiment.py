"""
File: pl_experiment.py
Author: Jonathan Swinnen
Last Updated: 2023-09-11
Description: A class which represents an experiment. It contains a datamodule, model and trainer. 
             It handles loading the config file and building the objects from the loaded parameters to set up the experiment,
             and creates and calls the trainer to run the experiment.
"""

import traceback
import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import wandb
import experiment.config as config
from datetime import datetime
from datamodule import HDF5DataModule
from pathlib import Path

class PlExperiment:
    def __init__(self, cfg_files, dict_edits={}):
        print("Initializing experiment...")
        self.cfg = config.merge(config.read(cfg_files), dict_edits)
        cfg_txt = config.to_yaml(self.cfg)
        print("Detected parameters:\n-------")
        print(cfg_txt)
        self.cfg_flat = config.flatten(self.cfg)
        
        self.project_name = self.cfg["project_name"]
        self.experiment_name = self.cfg.get("experiment_name", None)
        self.log_wandb = self.cfg.get("log_wandb", False)
        self.ckpt_wandb = self.cfg.get("ckpt_wandb", None)

        self.model_cfg = self.cfg["model"]
        self.optimizers_cfg = self.cfg["optimizers"]
        self.schedulers_cfg = self.cfg.get("schedulers", {})
        self.losses_cfg = config.load_objects(self.cfg["losses"])
        self.metrics_cfg = config.load_objects(self.cfg.get("metrics", {}))
        self.data_module = config.load_objects(self.cfg["data_module"])
        
        self.trainer_params = config.load_objects(self.cfg["trainer"])
        trainer_callbacks = self.trainer_params.get("callbacks", [])
        trainer_callbacks.append(plc.RichProgressBar())
        trainer_callbacks.append(plc.LearningRateMonitor(logging_interval='step'))
        self.ckpt_path = None
        print(trainer_callbacks)
        self.trainer_params["callbacks"] = trainer_callbacks
        print("-------\nInitialized.")
        
    def run(self):
        print("Preparing experiment for run...")
        
        if self.log_wandb:
            print("Creating wandb run...")
            wandb.init(project=self.project_name, name=self.experiment_name, config=config.remove_lists(self.cfg))
        
        pl_module = None
        print("Building model...")
        
        ckpt = self.model_cfg.get("load_ckpt_weights", None)
        if ckpt:
            print("Using model checkpoint:", ckpt)
            ckpt = self._get_ckpt(ckpt)
            print("Loading pl module checkpoint...")
            pl_module = LightningWrapper.load_from_checkpoint(ckpt)
        else:
            pl_module = LightningWrapper(self.model_cfg["module"], self.losses_cfg, self.metrics_cfg, self.optimizers_cfg, self.schedulers_cfg)

        print("Setting up logger...")
        now = datetime.now()
        dt_string = now.strftime("[%Y-%m-%d-%H:%M:%S]")
        exp_name = self.experiment_name + dt_string
        
        if self.log_wandb:
            logger = pl.loggers.WandbLogger(log_model="all")  
            self.trainer_params["logger"] = logger
        
        print("Setting up trainer...")
        
        trainer_params = self.trainer_params
        if trainer_params.get("ckpt_path") and self.ckpt_wandb:
            self.ckpt_path = trainer_params["ckpt_path"]
            print("Resuming trainer from checkpoint:", self.ckpt_path)
            self.ckpt_path = self._get_ckpt(self.ckpt_path)
        
        #trainer param ckpt_path key needs to be removed if it exists (als if value is null)
        if "ckpt_path" in trainer_params:
            del trainer_params["ckpt_path"]
        
        trainer = pl.Trainer(**trainer_params)
        
        print("Setup done! Starting experiment.")

        status = "success"
        print("fit")
        try:
            trainer.fit(pl_module, datamodule=self.data_module, ckpt_path=self.ckpt_path)
            trainer.test(datamodule=self.data_module)
        except Exception as e:
            print(e)
            traceback.print_exc()
            status = "failed"
        finally:
            if self.log_wandb:
                print("ending wandb run")
                logger.finalize(status)
                wandb.finish(0 if status=="success" else 1)
        print("experiment finished.")
            
        
    def export_model(self):
        raise NotImplementedError()

        
    def _get_ckpt(self, ckpt):
        if self.ckpt_wandb:
            print("Downloading checkpoint artifact", ckpt, " from wandb...")
            artifact = wandb.run.use_artifact(ckpt, type="model")
            ckpt = artifact.download() + "/model.ckpt"
            print("Downloaded checkpoint to", ckpt)
        return ckpt
            

class LightningWrapper(pl.LightningModule):
    def __init__(self, model, losses, metrics, optimizers, schedulers):
        super().__init__()
        self.save_hyperparameters()
        self.model = config.load_objects(model)
        self.losses = config.load_objects(losses)
        self.metrics = config.load_objects(metrics)
        self.optimizer_configs = config.load_objects(optimizers)
        self.scheduler_configs = config.load_objects(schedulers)

    def configure_optimizers(self):
        if not type(self.optimizer_configs) == list:
            self.optimizer_configs = [self.optimizer_configs]
        if not type(self.optimizer_configs) == list:
            self.scheduler_configs = [self.scheduler_configs]

        optimizers = []
        for cfg in self.optimizer_configs:
            opt = None
            if cfg["name"] == "Adam":
                opt = torch.optim.Adam(self.parameters(), **cfg["params"])
            elif cfg["name"] == "AdamW":
                print(cfg)
                opt = torch.optim.AdamW(self.parameters(), **cfg["params"])
            else:
                raise NotImplementedError("""
                    Currently, you cannot to create every possible optimizer from the YAML config only. 
                    By default, only 'Adam' and 'AdamW' are included.
                    To add a different optimizer, you also need to add the option to the 'configure_optimizers' 
                    function in the PLModelWrapper class (see pl_trainer.py). 
                    This might be fixed and extended later, but for now it is not a priority.
                """)
            optimizers.append(opt)

        schedulers = []
        for cfg in self.scheduler_configs:
            sched = None
            opt_id = cfg.get("optimizer_id", 0)
            if cfg["name"] == "ReduceLROnPlateau":
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[opt_id], **cfg["params"])
            elif cfg["name"] == "CosineAnnealingLR":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[opt_id], **cfg["params"])
            else:
                raise NotImplementedError("""
                    Currently, you cannot to create every possible scheduler from the YAML config only. 
                    By default, only 'ReduceLROnPlateau' and 'CosineAnnealingLR' are included.
                    To add a different optimizer, you also need to add the option to the 'configure_optimizers' 
                    function in the PLModelWrapper class (see pl_trainer.py). 
                    This might be fixed and extended later, but for now it is not a priority.
                """)
            schedulers.append({"scheduler": sched, "interval": cfg["interval"], "monitor": cfg["monitor"]})

        return optimizers, schedulers


    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)

    def forward(self, x):
        return self.model(x)        

    def training_step(self, batch, batch_idx):
        x, y = batch
        yh = self(x)
        loss = self.calc_loss(yh, y, on_step=True, log_prefix="train")
        self.calc_metrics(yh, y, log_prefix="train")
        return {"loss": loss, "output": yh}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yh = self(x)
        loss = self.calc_loss(yh, y, log_prefix="val")
        self.calc_metrics(yh, y, log_prefix="val")
        return {"loss": loss, "output": yh}

    def test_step(self, batch, batch_idx):
        self.log
        x, y = batch
        yh = self(x)
        loss = self.calc_loss(yh, y, log_prefix="test")
        self.calc_metrics(yh, y, log_prefix="test")
        return {"loss": loss, "output": yh}

    def calc_loss(self, yh, y, log_prefix=None, on_step=False, on_epoch=True):
        loss = 0
        for entry in self.losses:
            l = entry["loss"](yh, y)
            loss += entry["weight"] * l
            log_name = "loss_" + entry["log_name"]
            if log_prefix: log_name = log_prefix + "_" + log_name
            self.log(log_name, l, on_step=on_step, on_epoch=on_epoch)
        log_name = "loss"
        if log_prefix: log_name = log_prefix + "_" + log_name
        self.log(log_name, loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        return loss

    def calc_metrics(self, yh, y, log_prefix=None, on_step=False, on_epoch=True):
        for entry in self.metrics:
            m = entry["metric"](yh, y)
            log_name = entry["log_name"]
            if log_prefix: log_name = log_prefix + "_" + log_name
            self.log(log_name, m, on_step=on_step, on_epoch=on_epoch)
