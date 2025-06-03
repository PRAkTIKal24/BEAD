# This file is now deprecated. Annealer logic has been moved to helper.py.
# Please use helper.Annealer for hyperparameter annealing.

"""
Utility for flexible hyperparameter annealing during training.
Supports: constant, trigger, and hardcoded schedule strategies.
"""

import torch

class Annealer:
    def __init__(self, config, optimizer=None):
        self.config = config
        self.optimizer = optimizer
        self.state = {}
        self.lr_scheduler = None
        if hasattr(config, 'annealing'):
            for param, settings in config.annealing.items():
                if settings.get('strategy') == 'trigger' and param == 'lr' and optimizer is not None:
                    trig = settings['trigger']
                    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        patience=trig.get('patience', 10),
                        factor=trig.get('factor', 0.5),
                        min_lr=trig.get('min_lr', 1e-6),
                    )

    def step(self, epoch, loss=None):
        """
        Call this at the end of each epoch.
        """
        if not hasattr(self.config, 'annealing'):
            return
        for param, settings in self.config.annealing.items():
            strat = settings.get('strategy')
            if strat == 'constant':
                pace = settings.get('pace', 1.0)
                if param == 'lr' and self.optimizer is not None:
                    for g in self.optimizer.param_groups:
                        g['lr'] *= pace
                else:
                    setattr(self.config, param, getattr(self.config, param) * pace)
            elif strat == 'trigger' and param == 'lr' and self.lr_scheduler is not None and loss is not None:
                self.lr_scheduler.step(loss)
            elif strat == 'hardcoded':
                schedule = settings.get('schedule', {})
                if epoch in schedule:
                    value = schedule[epoch]
                    if param == 'lr' and self.optimizer is not None:
                        for g in self.optimizer.param_groups:
                            g['lr'] = value
                    else:
                        setattr(self.config, param, value)
