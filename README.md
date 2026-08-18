# polvista

An interactive visualizer to help get some intuition on complex polarization prescriptions used to model observations of Active Galactic Nuclei (AGN) and other astronomical objects.
Pick a model, customize its parameter, and watch how the degree of polarization and electric vector position angle (EVPA) changes across the spectrum.


Polvista can also fit spectropolarimetric data with simple least-squares regression. First load some data, pick a model and set its parameters to a good initial guess, then just click Fit!

## Included models for the Faraday screen:

- Burn depolarization (external screen)
- Tribble screen (partially resolved external screen)
- Internal uniform Faraday screen
- Partial coverage, external and internal
- Two-component blends: two external screens, two internal screens, or internal + external screens, each with independent spectral parameters!

![Burn depolarization with equation card](assets/partialcover_stokes.png)

![Two external-screen components](assets/2comp_polarization.png)


## Install

Requires Python >= 3.7.

```bash
git clone https://github.com/medoug7/polvista.git
cd polvista
pip install .
```

## Run

```bash
polvista
```

or, without installing a console script:

```bash
python -m polvista
```

## Development

```bash
pip install -e .
```

installs the package in editable mode so code changes are picked up without reinstalling.


### Bayesian sampling (optional)

In addition to least-squares fitting, polvista can run a full Bayesian nested-sampling fit via [pymultinest](https://github.com/JohannesBuchner/PyMultiNest), giving posterior distributions and Bayesian evidence for model comparison instead of a single best-fit point.

This is an optional feature that only becomes available if `pymultinest` is installed. `pymultinest` is a Python wrapper around the [MultiNest](https://github.com/JohannesBuchner/MultiNest) Fortran library. Installation instructions for both `multinest` and `pymultinest` are available in [this link](https://johannesbuchner.github.io/PyMultiNest/install.html).

To visualize the resulting posteriors as a corner plot, also install the [`corner`](https://corner.readthedocs.io/) package:

```bash
pip install .[bayes]
```



