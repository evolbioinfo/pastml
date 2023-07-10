import numpy as np

from pastml.models import Model
from pastml.models.F81Model import F81Model

JC = 'JC'


class JCModel(F81Model):

    def __init__(self, states, forest_stats, sf=None, tau=0, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        kwargs['frequency_smoothing'] = False
        F81Model.__init__(self, states=states, forest_stats=forest_stats, sf=sf, tau=tau, optimise_tau=optimise_tau,
                          frequencies=np.ones(len(states), dtype=np.float64) / len(states),
                          reoptimise=reoptimise, parameter_file=parameter_file, **kwargs)
        self._optimise_frequencies = False
        self.name = JC

    def _print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tfrequencies\tall equal to {:g}\t(fixed)\n'.format(Model._print_parameters(self), 1 / len(self.states))

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        JC model can only have the basic parameters (scaling factor and smoothing factor, see pastml.models.Model).

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """
        # This model sets equal frequencies
        # and hence should only read the basic parameters (scaling and smoothing factors) from the input file
        return Model.parse_parameters(self, params, reoptimise)
