from pastml.acr.maxlikelihood.models.SimpleModel import SimpleModel
from pastml.acr.maxlikelihood.models.F81Model import F81Model

EFT = 'EFT'


class EFTModel(F81Model):

    def __init__(self, states, forest_stats, sf=None, tau=0, optimise_tau=False, parameter_file=None,
                 reoptimise=False, **kwargs):
        F81Model.__init__(self, states=states, forest_stats=forest_stats, sf=sf, tau=tau,
                          optimise_tau=optimise_tau, reoptimise=reoptimise, parameter_file=parameter_file, **kwargs)
        self._frequencies = self.observed_frequencies
        self._optimise_frequencies = False
        self._frequency_smoothing = False
        self.name = EFT

    def set_frequency_blocks(self, frequency_blocks):
        if len(frequency_blocks) > 1:
            raise NotImplementedError('You cannot set several frequency blocks on the EFT model, '
                                      'as it has its frequencies fixed (to the values observed at tips). '
                                      'If you are using a Skyline model, '
                                      'note that EFT can only be set for the most recent time interval.')
        else:
            F81Model.set_frequency_blocks(self, frequency_blocks)

    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tfrequencies:\tobserved in the tree\t(fixed)\n'.format(SimpleModel.print_parameters(self))

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
        return SimpleModel.parse_parameters(self, params, reoptimise)
