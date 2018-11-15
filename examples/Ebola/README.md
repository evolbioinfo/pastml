# EBOV from Guinea, Sierra Leone and Liberia from 2014-2016

This data set comes from the study by [Dudas *et al.*, 2017](https://www.nature.com/articles/nature22040).

## Data

It contains a tree with 1610 tips, and an annotation file describing the location for each tree tip among other metadata.
The original Nexus tree provided by Dudas *et al.* is transformed by conversion to Newick format, 
collapsing its inner branches of length zero, and naming inner nodes. 

## Ancestral character reconstruction (ACR) analysis

The main.py script performs the ACR of the location with different methods and models available in PastML.