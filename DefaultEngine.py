import os, sys
sys.path.append(os.getcwd())
sys.path.append("..")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import importlib


class Event(object):
    pass

class Observable(object):
    def __init__(self):
        self.callbacks = {}

    def register(self, name, callback, replace=True):
        if name not in self.callbacks or replace is True:
            self.callbacks[name] = callback
        else:
            raise Exception(f"will overwrite callback {name}")

    def fire(self, name, **attrs):
        if name not in self.callbacks.keys():
            return

        e = Event()
        e.source = self
        for k, v in attrs.iteritems():
            setattr(e, k, v)

        return self.callbacks[name](e)

class DefaultEngine(object):
    def __init__(self, opt):
        self.opt = opt
        self.observer = Observable()

        self.register('on_train_forward', self.default_train_forward)
        self.register('on_train_backward', self.default_train_backward)

    def register(self, name, callback):
        self.observer.register(name, callback)

    def fire(self, name, **attrs):
        return self.observer.fire(name)


    def train(self):
        self.epoch = 1

        self.fire('train_start')

        while self.epoch < self.opt.epochs:
            self.train_epoch()

            self.epoch += 1
            self.fire('train_epoch_end', epoch=self.epoch)


    def train_epoch(self):
        for i, batch in enumerate(self.dataloader):
            self.fire('train_iter_start')

            self.optimizer.zero_grad()

            input, output, target = self.fire('on_train_forward', model=self.model, batch=batch)

            self.fire('on_train_backward', input=input, output=output, target=target, criterion=self.criterion)

            self.optimizer.step()

            self.fire('train_iter_end')

    def preprocess(self):
        try:
            loss_class = getattr(importlib.import_module("torch.nn"), self.opt.loss)
            optim_class = getattr(importlib.import_module("torch.optim"), self.opt.optimizer)
            dataset_class = getattr(importlib.import_module("DGPT.DataLoader.DatasetLoader"), self.opt.dataset.type)
            model_class = getattr(importlib.import_module("DGPT.Model.wgan"), self.opt.model_class)
        except AttributeError:
            sys.exit(1)

        self.fire('on_init_dataset', engine=self, dataset_class=dataset_class, opt=self.opt.dataset)
        if self.dataset is None:
            self.dataset = dataset_class(root_dirs=self.opt.dataset.subsets)

        self.dataloader = DataLoader(self.dataset, batch_size=self.opt.batch_size, shuffle=True)

        self.model = model_class()
        self.criterion = loss_class()
        self.optimizer = optim_class(self.model.parameters(), self.opt.lr)


    def postprocess(self):
        pass


    def default_train_forward(self, e):
        input, target = e.batch
        output = e.model.forward(input)

        return input, output, target


    def default_train_backward(self, e):
        loss = e.criterion(e.output, e.target)
        loss.backward()


    def run(self):
        self.preprocess()

        if self.opt.mode == 'train':
            self.train()

        self.postprocess()

class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)

if __name__ == "__main__":
    options = yaml.load("""
    train:
      batch_size: 4
      lr: 1e-5
      epochs: 100
      image_size: 448
      dataset:
        type: Mixed
        subsets:
          [
          /dataset/1,
          /dataset/2,
          /dataset/3,
          ]
      loss: TotalVariationLoss
      optimizer: Adam
      network_mode: normal
      model_class: EyeDetectionNet
      checkpoints_dir: ./checkpoints
    """)

    train_opt = dict2obj(options.train)
    engine = DefaultEngine(train_opt)

    engine.run()
