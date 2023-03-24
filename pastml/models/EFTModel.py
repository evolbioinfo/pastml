import numpy as np

from pastml.models import ModelWithFrequencies, Model
from pastml.models.F81Model import F81Model

EFT = 'EFT'


class EFTModel(F81Model):

    def __init__(self, states, forest_stats, observed_frequencies, sf=None,
                 tau=0, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        F81Model.__init__(self, states=states, forest_stats=forest_stats, sf=sf, tau=tau,
                          optimise_tau=optimise_tau, frequencies=observed_frequencies,
                          reoptimise=reoptimise, parameter_file=parameter_file, **kwargs)
        self._optimise_frequencies = False
        self._frequency_smoothing = False
        self.name = EFT

    def _print_parameters(self):
        return '{}' \
               '\tfrequencies:\tobserved in the tree\t(fixed)\n'.format(Model._print_parameters(self))
