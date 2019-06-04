# PastML

__PastML__ provides fast methods for Ancestral Character Reconstruction (ACR) and visualisation
on rooted phylogenetic trees. Given a rooted tree and its node annotations, it can either visualise them as-is, 
or infer ancestral node states based on the tip states, with a selection of maximum likelihood and parsimonious methods. 
The result is then visualised as a zoomable html map.

#### Article

Ishikawa SA, Zhukova A, Iwasaki W, Gascuel O (2019) __A Fast Likelihood Method to Reconstruct and Visualize Ancestral Scenarios__. [_Molecular Biology and Evolution_, msz131](https://doi.org/10.1093/molbev/msz131).

## How it works

For detailed description of the ACR methods, and visualization procedure used in __PastML__ please visit our [help page](https://pastml.pasteur.fr/help).

## Try it online
Try __PastML__ at [pastml.pasteur.fr](https://pastml.pasteur.fr)

# Input data
As an input, one needs to provide a **rooted** phylogenetical tree in [newick](https://en.wikipedia.org/wiki/Newick_format) format,
and a table containing tip states, 
in tab-delimited (by default) or csv format (to be specified with *--data_sep ,* option).

### Example
You can download [HIV1-A in Albania data](examples/Albania/data) as an example.
Let's assume that the tree and annotation files are in the Downloads folder, 
and are named respectively Albanian.tree.152tax.tre	and data.txt.

The data.txt is a comma-separated file, containing tip ids in the first column, 
and Country in the second column, i.e.:

id | Country
----- |  -----
98CMAJ6932 | Africa
98CMAJ6933 | Africa
96CMAJ6134 | Africa
00SEAY5240 | WestEurope
... | ...
02GRAY0303 | Greece
97YUAF9960 | EastEurope

# Installation

There are 3 alternative ways to run __pastml__ on your computer: with [docker](https://www.docker.com/community-edition), [singularity](https://www.sylabs.io/singularity/) or in python3.

## Run with docker

### Basic usage
Once [docker](https://www.docker.com/community-edition) is installed, run the following command:

```bash
docker run -v <path_to_the_folder_containing_the_tree_and_the_annotations>:/data:rw -t evolbioinfo/pastml --tree /data/<tree_file.nwk> --data /data/<annotation_file.csv> --data_sep <separator_eg_,> --columns <column1 column2 ...> --html_compressed /data/<output_map.html>
```

For example, to reconstruct and visualise the ancestral Country states for Albanian data, 
one needs to run the following command:

```bash
docker run -v ~/Downloads:/data:rw -t evolbioinfo/pastml --tree /data/Albanian.tree.152tax.tre --data /data/data.txt --columns Country --html_compressed /data/Albanian_map.html --data_sep , 
```

This will produce a file Albanian_map.html in the Downloads folder, 
that can be viewed with a browser.


#### Help

To see advanced options, run
```bash
docker run -t evolbioinfo/pastml -h
```

## Run with singularity

### Basic usage
Once [singularity](https://www.sylabs.io/guides/2.6/user-guide/quick_start.html#quick-installation-steps) is installed, run the following command:

```bash
singularity run docker://evolbioinfo/pastml --tree <path/to/tree_file.nwk> --data <path/to/annotation_file.csv> --columns <column1 column2 ...> --html_compressed <path/to/output/map.html> --data_sep <separator_eg_,>
```

For example, to reconstruct and visualise the ancestral Country states for Albanian data, 
one needs to run the following command:

```bash
singularity run docker://evolbioinfo/pastml --tree ~/Downloads/Albanian.tree.152tax.tre --data ~/Downloads/data.txt --columns Country --html_compressed ~/Downloads/Albanian_map.html --data_sep , 
```

This will produce a file Albanian_map.html in the Downloads folder, 
that can be viewed with a browser.


#### Help

To see advanced options, run
```bash
singularity run docker://evolbioinfo/pastml -h
```

## Run in python3

### Windows
For **Windows** users, we recommend installing __pastml__ via [Cygwin environment](https://www.cygwin.com/).
First instal python3 and pip3 from the Cygwin packages. Then install __pastml__:
```bash
pip3 install pastml
```

### All other platforms

You can install __pastml__ for python with or without [conda](https://conda.io/docs/), following the procedures described below:

#### Installing with conda

Once you have conda installed create an environment for __pastml__ with python3, here we name it pastmlenv:

```bash
conda create --name pastmlenv python=3
```

Then activate it:
```bash
source activate pastmlenv
```

Then install __pastml__ in it:

```bash
pip install pastml
```

#### Installing without conda

Make sure that pyhon3 and pip3 are installed, then install __pastml__:

```bash
pip3 install pastml
```

### Basic usage in a command line
If you installed __pastml__ via conda, do not forget to first activate the dedicated environment (here named pastmlenv), e.g.

```bash
source activate pastmlenv
```

To run __pastml__:

```bash
pastml --tree <path/to/tree_file.nwk> --data <path/to/annotation_file.csv> --columns <column1 column2 ...> --html_compressed <path/to/output/map.html> --data_sep <separator_eg_,>
```

For example, to reconstruct and visualise the ancestral Country states for Albanian data, 
one needs to run the following command:

```bash
pastml --tree ~/Downloads/Albanian.tree.152tax.tre --data ~/Downloads/data.txt --columns Country --html_compressed ~/Downloads/Albanian_map.html --data_sep , 
```

This will produce a file Albanian_map.html in the Downloads folder, 
that can be viewed with a browser.

#### Help

To see advanced options, run:
```bash
pastml -h
```

### Basic usage in python3
```python
from pastml.acr import pastml_pipeline

# Path to the table containing tip/node annotations, in csv or tab format
data = "~/Downloads/data.txt"

# Path to the tree in newick format
tree = "~/Downloads/Albanian.tree.152tax.tre"

# Columns present in the annotation table,
# for which we want to reconstruct ancestral states
# (for Albanian data we only have one column, but multiple columns are also allowed)
columns = ['Country']

# Path to the output compressed map visualisation
html_compressed = "~/Downloads/Albanian_map.html"

# (Optional) path to the output tree visualisation
html = "~/Downloads/Albanian_tree.html"

pastml_pipeline(data=data, data_sep=',', columns=columns, name_column='Country', tree=tree,
                html_compressed=html_compressed, html=html, verbose=True)
```

### Examples

See the [examples folder](https://github.com/evolbioinfo/pastml/tree/master/examples) for ideas :)
