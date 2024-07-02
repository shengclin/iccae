from importlib import import_module
import torch.optim as optim
import os


m_mapper = {
    'cae':'CAE',
}
o_mapper = {
    'sgd':'SGD',
    'adam':'Adam',
}


class Model:
    def __init__(self,
                 model='cae',
                 optimizer='sgd',
                 scheduler=False,
                 hpar={'in_length':2048,'num_layers':5},
                 optimhpar={'lr':1e-2},
                 run='train'):

        self.modname   = model
        self.optimname = optimizer
        self.hpar = hpar
        self.optimhpar = optimhpar
        self.scheduler = scheduler
        self.run = run

        self._get_model()

    def _get_model(self):
        m = import_module(
            '.models.' + m_mapper[self.modname],
            package='cae'
        )
        self.encoder = getattr(m, 'encoder')(**self.hpar)
        self.decoder = getattr(m, 'decoder')(**self.hpar)

        if self.run == 'train':
            self._get_optim()

            if self.scheduler:
                self._get_scheduler()
        elif self.run == 'predict':
            pass
        else:
            raise Exception('Run with train or predict modes only')

    def _get_optim(self):
        _opti = getattr(optim, o_mapper[self.optimname])
        self.optimizer = _opti(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            **self.optimhpar
        )

    def _get_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer
        )