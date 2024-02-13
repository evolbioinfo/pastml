from pastml import PASTML_VERSION
from pastml.acr.maxlikelihood import MARGINAL_PROBABILITIES, MODEL


METHOD = 'method'
CHARACTER = 'character'

NUM_SCENARIOS = 'num_scenarios'
NUM_UNRESOLVED_NODES = 'num_unresolved_nodes'
NUM_STATES_PER_NODE = 'num_states_per_node_avg'
PERC_UNRESOLVED = 'percentage_of_unresolved_nodes'
NUM_NODES = 'num_nodes'
NUM_TIPS = 'num_tips'


def save_acr_stats(acr_result, out_param_file):
    # Not using DataFrames to speed up document writing
    with open(out_param_file, 'w+') as f:
        f.write('pastml_version\t{}\n'.format(PASTML_VERSION))
        for name in sorted(acr_result.keys()):
            if name not in [MARGINAL_PROBABILITIES, METHOD, MODEL]:
                f.write('{}\t{}\n'.format(name, acr_result[name]))
        f.write('{}\t{}\n'.format(METHOD, acr_result[METHOD]))
        if MODEL in acr_result:
            f.write('{}\t{}\n'.format(MODEL, acr_result[MODEL].name))

