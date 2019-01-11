# HIV-1 subtype C epidemic

This folder contains a Snakemake [[Köster *et al.*, 2012](https://doi.org/10.1093/bioinformatics/bts480)] pipeline and data 
needed for reconstruction of evolutionary history of Location and drug resistance of HIV-1C epidemic, and assessment of its robustness.

The pipeline steps are detailed below.

## Pipeline

### 1. Sequence alignment
This HIV-1C pol sequence data set is an update of the one used in the study by [Chevenet *et al.*, 2013](https://doi.org/10.1093/bioinformatics/btt010), 
which in turn updated the data from [Jung *et al.*, 2012](https://doi.org/10.1371/journal.pone.0033579).

We extended the alignment used by Chevenet *et al.* with HIV-1C pol sequences from the latest (2017) pol alignment 
in the Los Alamos HIV database [[Kuiken *et al.*, 2003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2613779/)], 
hence adding 583 new sequence. Addition of the new sequences was performed using 
MAFFT multiple sequence alignment program with --add option [[Katoh *et al.*, 2013](http://doi.org/10.1093/molbev/mst010)]. 

The final alignment contains 3,619 HIV-1C pol sequences, plus 35 outgroup reference sequences from the non-C subtypes. 

### 2. Metadata
#### Sampling dates and regions
The data set of Chevenet *et al.* was annotated with sampling years 
and countries grouped into 11 regions: North America, Central America, South America, Europe, Asia, West Africa, 
the Horn of Africa, Central Africa, East Africa, Southern Africa excluding South Africa, and South Africa. 
Los Alamos alignment contained the information of the year and country of sampling. 
We combined them and updated the region information for the Los Alamos sequences to match Chevenet et al.
#### Surveillance Drug Resistance Mutations (SDRMs)
We detected SDRMs in the alignment, using the Sierra web service of 
the Stanford HIV drug resistance database [[Liu *et al.*, 2006](http://doi.org/10.1086/503914)]. 


### 3. Tree reconstruction

#### Reconstruction
We removed the SDRM positions from the alignment 
and used it to reconstruct 5 most parsimonious trees with RAxML [[Stamatakis, 2014](https://doi.org/10.1093/bioinformatics/btu033)]. 
These trees were then used as starting trees for 5 runs of PhyML 3.0 [[Guindon *et al.*, 2010](https://doi.org/10.1093/sysbio/syq010)], 
RAxML-NG v0.7.0-beta [[Kozlov *et al.*, 2018](https://doi.org/10.1101/447110)] and FastTree 2 [[Price *et al.*, 2010](https://doi.org/10.1371/journal.pone.0009490)] 
with GTR+Γ6 substitution model. 

#### Rooting
The resulting trees were rooted with the outgroup sequences, which were subsequently removed from the trees. 

#### Polytomies
The branches of length zero were collapsed into polytomies. 

#### Comparison
We thereby obtained 5x3 ML trees with different topologies. 
To assess the difference we calculated the average normalized bipartition distances = 0.33 (calculated with ETE 3 toolkit [[Huerta-Cepas *et al.*, 2016](http://doi.org/10.1093/molbev/msw046)]), 
and average quartet distance = 0.31 (calculated with tqDist library [[Sand *et al.*, 2014](http://doi.org/10.1093/bioinformatics/btu157)]), 
where 0.0 means identical trees, and 1.0 trees that have no bipartition / no quartet in common. 

### 4. Ancestral character reconstruction (ACR)
We reconstructed ACR for Location and 10 most prevalend SDRMs using pastml.

### 5. Robustness of ACR

The 5 reconstructed trees with different topologies were used to check the robustness of ACR against phylogenetic uncertainty.
We also checked the robustness of the results regarding state sampling variations, 
as some regions were could be sampled more intensively than others. 
For this purpose we resampled the tree by keeping at most 250 tips per region 
(for the regions with less samples all the tips were kept, for those with more samples 250 random tips were kept) 
and performed ACR of the Location for 5 such trees.

## Running the pipeline

### Installing the dependencies

1. Install snakemake [[installation instructions](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)] (version 5.4.0 or higher)
1. Install Singularity [[singularity.lbl.gov](https://singularity.lbl.gov/)] (version 2.6.1 or higher)

### Running the pipeline
Check that the paths/filenames in the config.yaml file are correct.

From the snakemake directory run:
```bash
snakemake --snakefile Snakefile_trees --keep-going --cores <number_of_available_cores_eg_4> --use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~"
```

This will reconstruct the trees.

After tree reconstruction is finished, run:
```bash
snakemake --keep-going --cores <number_of_available_cores_eg_4> --use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~"
```

This will perform ACR analysis.
