from pastml.models import Model
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
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tfrequencies:\tobserved in the tree\t(fixed)\n'.format(Model._print_parameters(self))

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        EFT model can only have the basic parameters (scaling factor and smoothing factor, see pastml.models.Model),
        as its frequencies are set to observed values.

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """
        # This model sets frequencies from the observed values
        # and hence should only read the basic parameters (scaling and smoothing factors) from the input file
        return Model.parse_parameters(self, params, reoptimise)
