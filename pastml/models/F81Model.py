import numpy as np

from pastml.models import ModelWithFrequencies

F81 = 'F81'


class F81Model(ModelWithFrequencies):

    def __init__(self, states, forest_stats, sf=None, frequencies=None, tau=0,
                 frequency_smoothing=False, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        ModelWithFrequencies.__init__(self, states=states, forest_stats=forest_stats,
                                      sf=sf, tau=tau, frequencies=frequencies,
                                      optimise_tau=optimise_tau, frequency_smoothing=frequency_smoothing,
                                      reoptimise=reoptimise, parameter_file=parameter_file, **kwargs)
        self.name = F81

    def get_mu(self):
        """
        Calculates the mutation rate for F81 (and JC that is a simplification of it),
        as \mu = 1 / (1 - sum_i \pi_i^2). This way the overall rate of mutation -\mu trace(\Pi Q) is 1.
        See [Gascuel "Mathematics of Evolution and Phylogeny" 2005] for further details.

        :return: mutation rate \mu = 1 / (1 - sum_i \pi_i^2)
        """
        return 1. / (1. - self.frequencies.dot(self.frequencies))

    def get_Pij_t(self, t, *args, **kwargs):
        """
        Calculate the probability of substitution i->j over time t, given the mutation rate mu:
        For F81 (and JC which is a simpler version of it)
        Pij(t) = \pi_j (1 - exp(-mu t)) + exp(-mu t), if i == j, \pi_j (1 - exp(-mu t)), otherwise
        [Gascuel "Mathematics of Evolution and Phylogeny" 2005],
        where \pi_i is the equillibrium frequency of state i,
        and \mu is the mutation rate: \mu = 1 / (1 - sum_i \pi_i^2).

        :param t: time t
        :type t: float
        :return: probability matrix
        :rtype: numpy.ndarray
        """
        t = self.transform_t(t)
        mu = self.get_mu()
        # if mu == inf (e.g. just one state) and t == 0, we should prioritise mu
        exp_mu_t = 0. if (mu == np.inf) else np.exp(-mu * t)
        return (1 - exp_mu_t) * self.frequencies + np.eye(len(self.frequencies)) * exp_mu_t
