# lake-altimetry-inversions
Author: Aaron Stubblefield (Dartmouth College)

# Overview
This program inverts for a basal vertical velocity anomaly "w_inv" given the observed surface
elevation change data "h_obs" as input by solving a least-squares optimization problem.

The motivation is that the basal vertical velocity anomaly is associated with
subglacial lake activity that produces the elevation anomaly. Once we invert
for the basal vertical velocity anomaly, we can try to estimate the subglacial
water-volume change or the areal extent of the lake.

The main model assumptions are (1) Newtonian viscous ice flow, (2) a linear
basal sliding law, and (3) that all fields are small perturbations from a simple
reference flow state. These assumptions allow for efficient solution of the problem:
2D (map-plane) Fourier transforms and convolution in time are the main
operations. The model and numerical method are described in a forthcoming manuscript.

# Dependencies
As of this commit, this code runs with the latest SciPy (https://www.scipy.org/)
release. Plotting relies on Matplotlib (https://matplotlib.org/).

# Required data
The following data are needed to run the code with ICESat-2 ATL15 data.

(Note: you can run the inversion with synthetic data without this data--see description below)

1. ICESat-2 ATL15 (1 km resolution):

   *website: https://nsidc.org/data/atl15/versions/2

   *filename: ATL15_AA_0311_01km_001_01.nc

2. WAVI model output (for basal drag, ice thickness, viscosity )

   *website: https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=5f0ac285-cca3-4a0e-bcbc-d921734395ab

   *filename: WAVI_5km_Initial.nc

   *reference:
      >Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within
      the Antarctic ice sheet and its controls inferred from satellite observations.
      Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.

3. MEaSUREs Antarctic Ice Velocity (V2)

   *webiste: https://nsidc.org/data/nsidc-0484/versions/2

   *filename: antarctica_ice_velocity_450m_v2.nc

   *reference:
      > Rignot, E., J. Mouginot, and B. Scheuchl. 2011. Ice Flow of the Antarctic
       Ice Sheet. Science. 333. DOI: 10.1126/science.1208336.

4. Antarctic Subglacial Lake inventory

   *website: https://github.com/mrsiegfried/Siegfried2021-GRL

   *filename: SiegfriedFricker2018-outlines.h5

   *references:

   >   Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial
      lake activity in Antarctica from multi-mission satellite altimetry.
      Annals of Glaciology, 59(76pt1), 42-55.

   >   Siegfried, M. R., & Fricker, H. A. (2021). Illuminating active
      subglacial lake processes with ICESat‐2 laser altimetry. Geophysical
      Research Letters, 48(14), e2020GL091089.

# Contents
The inversion code is in the *inverse-model* directory.
## Source files
The model is organized in several python files in the *inverse-model/source* directory as follows.

1. **main.py** is the main file that calls the inverse problem solver and then
plots the solution.

2. **inversion.py** is the inverse problem solver: this defines the normal equations
and calls the conjugate gradient solver.

3. **conj_grad.py** defines the conjugate gradient method that is used to solve
the normal equations.

4. **operators.py** defines the forward and adjoint operators that appear in the
normal equations.

5. **kernel_fcns.py** defines the relaxation and transfer functions that the forward and adjoint operators depend on.

6. **regularizations.py** defines the regularization.

7. **params.py** defines all of the model options and parameters.

There are two auxiliary functions as well:

8. **localization.py** defines a function that remove the off-lake component from fields when necessary.

9. **post_process.py** defines functions to estimate the water-volume change
from the elevation change and inversion.

## Scripts
The *inverse-model/scripts* contains files for running the model, pre-processing
the data, and plotting the results. The main scripts are:

1. **make_synth_data.py** makes synthetic data that can be inverted to verify the code.

2. **proc_icesat_data.py** obtains and interpolates the ICESat-2 ATL15 data around
a given lake location (from the inventory listed above), and also obtains some
auxiliary parameters (ice thickness, viscosity, basal drag coefficient, flow speed).

3. **run_example.py** is a short script that runs the inversion solver and plots
the results.

Plotting can  be modified in the **plot_results.py** scripts.

## Miscellaneous
There is also some FEniCS (https://fenicsproject.org/) code in the *misc* directory
for producing synthetic data with a fully nonlinear finite element model. See the repository
agstub/grounding-line-methods for a description. The version herein assumes
radial symmetry with respect to the horizontal coordinates.

# Running the model
When running the inversion, the data and auxiliary parameters are stored in a directory
called *data_lakename* and the results are stored in a directory called
*results_lakename*.

## Synthetic data
First run **make_synth_data.py** to create the synthetic data and enter the
lake name "synth" when prompted. This just supplies a given basal vertical
velocity anomaly (and some auxiliary parameters) to the linearized model
and adds some noise to the resulting elevation-change anomaly.

To invert the synthetic data, run **run_example.py** and enter the lake name
"synth" when prompted.

## ICESat-2 data
First, make sure that you have all of the data sets outlined above and that
the correct paths to the data sets are set in **proc_icesat_data.py**.  
Then, set the lake name ('lake_name') is **proc_icesat_data.py** from
the inventory and run the program (this will take some time).
Finally, run **run_example.py** and enter the lake name
when prompted.

To see a list of the lake names in the inventory run the python script in IPython:
```
  from load_lakes import gdf
  gdf['name'].to_list()
```
