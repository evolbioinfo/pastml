# Skyline Model

The Skyline model assumes that models and states can change at user-defined time points, 
but stay constant within each time interval.

It hence defines:
* n > 0 time breakpoints: y<sub>0</sub>, ..., y<sub>n-1</sub>,
* n + 1 submodels: sub>0</sub>, ..., M<sub>n</sub>, 
* and n pairs of state mappings: S<sup>(i)</sup> -> S<sup>(i-1)</sup>, 
S<sup>(i-1)</sup> -> S<sup>(i)</sup>, where S<sup>(i)</sup> are states of model M<sub>i</sub>.

In the most simple case, all models have the same states:
S<sup>(i)</sup> = S<sup>(j)</sup> ∀ 0 < i, j ≤ n.

We assume that
* M<sub>0</sub> corresponds to the oldest time interval ]-∞, y<sub>0</sub>[, 
which includes the root of the (oldest forest) tree,
* M<sub>i</sub> (0 < i < n) corresponds to time interval [y<sub>i-1</sub>, y<sub>i</sub>[,
* and M<sub>n</sub> corresponds to the most recent time interval [y<sub>n-1</sub>, +∞[, which inclused the most recent tip.

All the submodels are assumed to be at equilibrium. Therefore, to ensure time-reversibility,
the Skyline model imposes constraints on underlying submodels such that:
* The most recent submodel M<sub>n</sub> does not have any additional constraints
* M<sub>i</sub>'s state frequencies (0 < i < n) are calculated 
based on the frequencies of the corresponding M<sub>i-1</sub>'s states:
  * f<sub>s"</sub> ≥ f<sub>s'</sub> ∀ s": map<sup>(i->i-1)</sup>(s') = {s"};
  * ∀ subset pair S' ⊆ S<sup>(i)</sup>, S" ⊆ S<sup>(i-1)</sup> 
  such that ∀ s' ∈ S' map<sup>(i->i-1)</sup>(s') ⊆ S" and ∀ s" ∈ S" map<sup>(i-1->i)</sup>(s") ⊆ S',
  Σ<sub>s' ∈ S'</sub>f<sub>s'</sub> = Σ<sub>s" ∈ S"</sub>f<sub>s"</sub>.