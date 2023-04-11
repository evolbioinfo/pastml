import numpy as np

from pastml.models import Model
from pastml.models.CustomRatesModel import CustomRatesModel

"""
The JTT matrix below is taken from https://www.ebi.ac.uk/goldman-srv/dayhoff/

See the following paper for more details:
Kosiol,C. and Goldman,N. The Different Versions of the Dayhoff Rate Matrix. 
"""

JTT = 'JTT'

AA_NAMES = ['Alanine', 'Arginine', 'Asparagine', 'Aspartic Acid', 'Cysteine', 'Glutamine', 'Glutamic Acid', 'Glycine',
            'Histidine', 'Isoleucine', 'Leucine', 'Lysine', 'Methionine', 'Phenylalanine', 'Proline', 'Serine',
            'Threonine', 'Tryptophan', 'Tyrosine', 'Valine']

AA_3_LETTER_CODES = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
                     'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
                     'Thr', 'Trp', 'Tyr', 'Val']

AA_1_LETTER_CODES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                     'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                     'T', 'W', 'Y', 'V']

JTT_STATES = np.array(AA_1_LETTER_CODES)

NUM_AA = 20

JTT_RATE_MATRIX = np.zeros(shape=(NUM_AA, NUM_AA), dtype=np.float64)
JTT_RATE_MATRIX[np.tril_indices(NUM_AA, k=-1)] = \
    [0.531678,
     0.557967, 0.451095,
     0.827445, 0.154899, 5.549530,
     0.574478, 1.019843, 0.313311, 0.105625,
     0.556725, 3.021995, 0.768834, 0.521646, 0.091304,
     1.066681, 0.318483, 0.578115, 7.766557, 0.053907, 3.417706,
     1.740159, 1.359652, 0.773313, 1.272434, 0.546389, 0.231294, 1.115632,
     0.219970, 3.210671, 4.025778, 1.032342, 0.724998, 5.684080, 0.243768, 0.201696,
     0.361684, 0.239195, 0.491003, 0.115968, 0.150559, 0.078270, 0.111773, 0.053769, 0.181788,
     0.310007, 0.372261, 0.137289, 0.061486, 0.164593, 0.709004, 0.097485, 0.069492, 0.540571, 2.335139,
     0.369437, 6.529255, 2.529517, 0.282466, 0.049009, 2.966732, 1.731684, 0.269840, 0.525096, 0.202562, 0.146481,
     0.469395, 0.431045, 0.330720, 0.190001, 0.409202, 0.456901, 0.175084, 0.130379, 0.329660, 4.831666, 3.856906, 0.624581,
     0.138293, 0.065314, 0.073481, 0.032522, 0.678335, 0.045683, 0.043829, 0.050212, 0.453428, 0.777090, 2.500294, 0.024521, 0.436181,
     1.959599, 0.710489, 0.121804, 0.127164, 0.123653, 1.608126, 0.191994, 0.208081, 1.141961, 0.098580, 1.060504, 0.216345, 0.164215, 0.148483,
     3.887095, 1.001551, 5.057964, 0.589268, 2.155331, 0.548807, 0.312449, 1.874296, 0.743458, 0.405119, 0.592511, 0.474478, 0.285564, 0.943971, 2.788406,
     4.582565, 0.650282, 2.351311, 0.425159, 0.469823, 0.523825, 0.331584, 0.316862, 0.477355, 2.553806, 0.272514, 0.965641, 2.114728, 0.138904, 1.176961, 4.777647,
     0.084329, 1.257961, 0.027700, 0.057466, 1.104181, 0.172206, 0.114381, 0.544180, 0.128193, 0.134510, 0.530324, 0.089134, 0.201334, 0.537922, 0.069965, 0.310927, 0.080556,
     0.139492, 0.235601, 0.700693, 0.453952, 2.114852, 0.254745, 0.063452, 0.052500, 5.848400, 0.303445, 0.241094, 0.087904, 0.189870, 5.484236, 0.113850, 0.628608, 0.201094, 0.747889,
     2.924161, 0.171995, 0.164525, 0.315261, 0.621323, 0.179771, 0.465271, 0.470140, 0.121827, 9.533943, 1.761439, 0.124066, 3.038533, 0.593478, 0.211561, 0.408532, 1.143980, 0.239697, 0.165473]
JTT_RATE_MATRIX = np.maximum(JTT_RATE_MATRIX, JTT_RATE_MATRIX.T)

JTT_FREQUENCIES = np.array(
    [0.076862, 0.051057, 0.042546, 0.051269, 0.020279, 0.041061, 0.061820, 0.074714, 0.022983, 0.052569, 0.091111, 0.059498, 0.023414, 0.040530, 0.050532, 0.068225, 0.058518, 0.014336, 0.032303, 0.066374])
JTT_FREQUENCIES = JTT_FREQUENCIES / JTT_FREQUENCIES.sum()

state_order = np.argsort(JTT_STATES)
JTT_STATES = JTT_STATES[state_order]
JTT_FREQUENCIES = JTT_FREQUENCIES[state_order]
JTT_RATE_MATRIX = JTT_RATE_MATRIX[:, state_order][state_order, :]


class JTTModel(CustomRatesModel):

    def __init__(self, forest_stats, sf=None, tau=0, optimise_tau=False, parameter_file=None, reoptimise=False, **kwargs):
        kwargs['states'] = JTT_STATES
        if 'frequency_smoothing' in kwargs:
            del kwargs['frequency_smoothing']
        CustomRatesModel.__init__(self, forest_stats=forest_stats, sf=sf,
                                  frequencies=JTT_FREQUENCIES, rate_matrix=JTT_RATE_MATRIX,
                                  parameter_file=parameter_file, reoptimise=reoptimise, frequency_smoothing=False,
                                  tau=tau, optimise_tau=optimise_tau, **kwargs)
        self._optimise_frequencies = False
        self.name = JTT

    @CustomRatesModel.states.setter
    def states(self, states):
        raise NotImplementedError('The JTT states are preset and cannot be changed.')

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        JTT model can only have the basic parameters (scaling factor and smoothing factor, see pastml.models.Model).

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """

        # This model sets fixed frequencies
        # and hence should only read the basic parameters (scaling and smoothing factors) from the input file
        return Model.parse_parameters(self, params, reoptimise)

