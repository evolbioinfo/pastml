import logging

import numpy as np

from pastml.models import Model, ModelWithFrequencies

HKY = 'HKY'
HKY_STATES = np.array(['A', 'C', 'G', 'T'])
A, C, G, T = 0, 1, 2, 3
KAPPA = 'kappa'


class HKYModel(ModelWithFrequencies):

    def __init__(self, forest_stats, sf=None, frequencies=None, kappa=4, tau=0,
                 frequency_smoothing=False, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        self._kappa = None
        self._optimise_kappa = True
        kwargs['states'] = HKY_STATES
        ModelWithFrequencies.__init__(self, forest_stats=forest_stats,
                                      sf=sf, tau=tau, optimise_tau=optimise_tau, frequencies=frequencies,
                                      frequency_smoothing=frequency_smoothing, reoptimise=reoptimise,
                                      parameter_file=parameter_file, **kwargs)
        if self._kappa is None:
            self._kappa = kappa
        self.name = HKY

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        if self._optimise_kappa:
            self._kappa = kappa
        else:
            raise NotImplementedError('The kappa value is preset and cannot be changed.')

    @Model.states.setter
    def states(self, states):
        raise NotImplementedError("The HKY model is only implemented for nucleotides: "
                                  "the states are A, C, G, T and cannot be reset")

    def get_Pij_t(self, t, *args, **kwargs):
        """
        Calculates the probability matrix of substitutions i->j over time t,
        with HKY model [Hasegawa, Kishino and Yano 1985], given state frequencies and kappa.
        Is only implemented for 4 nucleotide states.

        :param t: time
        :type t: float
        :return: probability matrix
        :rtype: numpy.ndarray
        """
        t = self.transform_t(t)

        pi_a, pi_c, pi_g, pi_t = self.frequencies
        pi_ag = pi_a + pi_g
        pi_ct = pi_c + pi_t
        beta = .5 / (pi_ag * pi_ct + self.kappa * (pi_a * pi_g + pi_c * pi_t))

        exp_min_beta_t = np.exp(-beta * t)
        exp_ct = np.exp(-beta * t * (1. + pi_ct * (self.kappa - 1.))) / pi_ct
        exp_ag = np.exp(-beta * t * (1. + pi_ag * (self.kappa - 1.))) / pi_ag
        ct_sum = (pi_ct + pi_ag * exp_min_beta_t) / pi_ct
        ag_sum = (pi_ag + pi_ct * exp_min_beta_t) / pi_ag
        p = np.ones((4, 4), dtype=np.float64) * (1 - exp_min_beta_t)
        p *= self.frequencies

        p[T, T] = pi_t * ct_sum + pi_c * exp_ct
        p[T, C] = pi_c * ct_sum - pi_c * exp_ct

        p[C, T] = pi_t * ct_sum - pi_t * exp_ct
        p[C, C] = pi_c * ct_sum + pi_t * exp_ct

        p[A, A] = pi_a * ag_sum + pi_g * exp_ag
        p[A, G] = pi_g * ag_sum - pi_g * exp_ag

        p[G, A] = pi_a * ag_sum - pi_a * exp_ag
        p[G, G] = pi_g * ag_sum + pi_a * exp_ag

        return p

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        return ModelWithFrequencies.get_num_params(self) + (1 if self._optimise_kappa else 0)

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        if self.extra_params_fixed():
            Model.set_params_from_optimised(self, ps, **kwargs)
        else:
            ModelWithFrequencies.set_params_from_optimised(self, ps, **kwargs)
            n_params = ModelWithFrequencies.get_num_params(self)
            if self._optimise_kappa:
                self.kappa = ps[n_params]

    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        if self.extra_params_fixed():
            return Model.get_optimised_parameters(self)
        return np.hstack((ModelWithFrequencies.get_optimised_parameters(self),
                          [self.kappa] if self._optimise_kappa else []))

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        if self.extra_params_fixed():
            return Model.get_bounds(self)
        return np.array((*ModelWithFrequencies.get_bounds(self),
                        *([np.array([1e-6, 20.])] if self._optimise_kappa else [])))

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        For the HKY model, apart from the basic parameters (scaling factor and smoothing factor) and the frequencies
        (see pastml.models.ModelWithFrequencies),
        the input might contain an optional kappa value:
        the key for kappa value is the pastml.models.HKYModel.KAPPA.

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """
        params = ModelWithFrequencies.parse_parameters(self, params, reoptimise)
        logger = logging.getLogger('pastml')
        if KAPPA in params:
            self._kappa = params[KAPPA]
            try:
                self._kappa = np.float64(self._kappa)
                if self._kappa <= 0:
                    logger.error(
                        'Kappa cannot be negative, ignoring the value given in paramaters ({}).'.format(self._kappa))
                    self._kappa = None
                else:
                    self._optimise_kappa = reoptimise
            except:
                logger.error('Kappa ({}) given in parameters is not float, ignoring it.'.format(self._kappa))
                self._kappa = None
        return params

    def _print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tkappa\t{:.6f}\t{}\n'.format(ModelWithFrequencies.get_optimised_parameters(self),
                                          self.kappa, '(optimised)' if self._optimise_kappa else '(fixed)')

    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        ModelWithFrequencies.freeze(self)
        self._optimise_kappa = False

    def save_parameters(self, filehandle):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filehandle: filehandle for the file where the parameter values should be written.
        :return: void
        """
        ModelWithFrequencies.save_parameters(self, filehandle)
        filehandle.write('{}\t{:g}\n'.format(KAPPA, self.kappa))
