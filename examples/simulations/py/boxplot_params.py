from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, boxplot, title, plot

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_params', type=str, nargs='+')
    parser.add_argument('--output_pdf', required=True, type=str)
    parser.add_argument('--sf', required=True, type=float)
    parser.add_argument('--kappa', required=True, type=float)
    parser.add_argument('--A', required=True, type=float)
    parser.add_argument('--C', required=True, type=float)
    parser.add_argument('--G', required=True, type=float)
    parser.add_argument('--T', required=True, type=float)
    parser.add_argument('--model', required=True, type=str)
    params = parser.parse_args()

    data = []
    for f in params.input_params:
        df = pd.read_csv(f, header=0, index_col=0, sep='\t')
        if 'A' in df.index and 'C' in df.index and 'G' in df.index and 'T' in df.index:
            data.append([float(df.loc['scaling_factor', 'value']), 
                         (float(df.loc['kappa', 'value']) if 'kappa' in df.index else 1), 
                         float(df.loc['A', 'value']), float(df.loc['C', 'value']), 
                         float(df.loc['G', 'value']), float(df.loc['T', 'value'])])
    df = pd.DataFrame(data=data, columns=['SF', 'kappa', 'A', 'C', 'G', 'T'])

    if params.model == 'JC':
        boxplot(df['SF'])
        plot([0, 2], [params.sf, params.sf], 'b-')
        title("SF={:.2f}, K={:.2f}, A={:.2f}, A={:.2f}, A={:.2f}, A={:.2f}, Model {}"
              .format(params.sf, params.kappa, params.A, params.C, params.G, params.T, params.model))
    else:
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].boxplot(df['SF'])
        axs[0, 0].set_title('scaling factor')
        axs[0, 0].plot([0, 2], [params.sf, params.sf], 'b-')
        axs[0, 1].axis('off')
        axs[0, 3].axis('off')
    
        if params.model == 'HKY':
            axs[0, 2].boxplot(df['kappa'])
            axs[0, 2].set_title('kappa')
            axs[0, 2].plot([0, 2], [params.kappa, params.kappa], 'b-')
        elif params.model == 'F81':
            axs[0, 2].axis('off')

        axs[1, 0].boxplot(df['A'])
        axs[1, 0].set_title('A')
        axs[1, 0].plot([0, 2], [params.A, params.A], 'b-')

        axs[1, 1].boxplot(df['C'])
        axs[1, 1].set_title('C')
        axs[1, 1].plot([0, 2], [params.C, params.C], 'b-')

        axs[1, 2].boxplot(df['G'])
        axs[1, 2].set_title('G')
        axs[1, 2].plot([0, 2], [params.G, params.G], 'b-')

        axs[1, 3].boxplot(df['T'])
        axs[1, 3].set_title('T')
        axs[1, 3].plot([0, 2], [params.T, params.T], 'b-')

        fig.suptitle("SF={:.2f}, K={:.2f}, A={:.2f}, A={:.2f}, A={:.2f}, A={:.2f}, Model {}"
                     .format(params.sf, params.kappa, params.A, params.C, params.G, params.T, params.model))

    savefig(params.output_pdf, dpi=300, papertype='a4', orientation='landscape')
