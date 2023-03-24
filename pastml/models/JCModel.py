import numpy as np

from pastml.models import Model
from pastml.models.F81Model import F81Model

JC = 'JC'


class JCModel(F81Model):

    def __init__(self, states, forest_stats, sf=None, tau=0, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        F81Model.__init__(self, states=states, forest_stats=forest_stats, sf=sf, tau=tau, optimise_tau=optimise_tau,
                          frequencies=np.ones(len(states), dtype=np.float64) / len(states),
                          reoptimise=reoptimise, parameter_file=parameter_file, **kwargs)
        self._optimise_frequencies = False
        self._frequency_smoothing = False
        self.name = JC

    def _print_parameters(self):
        return '{}' \
               '\tfrequencies\tall equal to {:g}\t(fixed)\n'.format(Model._print_parameters(self), 1 / len(self.states))
