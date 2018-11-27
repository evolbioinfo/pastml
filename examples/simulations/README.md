# Simulated charcter reconstruction

This folder contains a Snakemake [[Köster *et al.*, 2012](https://doi.org/10.1093/bioinformatics/bts480)] pipeline 
for reconstruction of evolutionary history of simulated characters (A, C, G, T).

The nucleotide character values are simulated on a given tree with [Seq-Gen](https://github.com/rambaut/Seq-Gen). 
A PASTML ACR reconstruction is then performed with ML methods and the result is visualised for further comparison.


## Running the pipeline

### Installing the dependencies
1. Install the following python3 packages (e.g. via pip3 install):
    * pastml (>=2.0)
    * snakemake (>=4.3.1)
2. Install the following software:
    * Seq-Gen [[Rambaut and Grassly](https://github.com/rambaut/Seq-Gen)]
3. Once you've installed everything please update the config.yaml file accordingly.

### Running the pipeline
From the snakemake directory run:
```bash
snakemake --keep-going
```


