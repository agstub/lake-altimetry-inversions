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
release.

Plotting relies on Matplotlib (https://matplotlib.org/).

# Required data
The following data are needed to run the code with ICESat-2 ATL15 data.

(Note: you can run the inversion with synthetic data without downloading this data--see description below)

1. ICESat-2 ATL15 (1 km resolution):

   *website: https://nsidc.org/data/atl15/versions/2

   *filename: ATL15_AA_0314_01km_002_02.nc

2. WAVI model output (for basal drag and ice viscosity )

   *website: https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=5f0ac285-cca3-4a0e-bcbc-d921734395ab

   *filename: WAVI_5km_Initial.nc

   *reference:
      >Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within
      the Antarctic ice sheet and its controls inferred from satellite observations.
      Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.

3. MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 (NSIDC-0754)

   *webiste: https://nsidc.org/data/nsidc-0754/versions/1

   *filename: antarctic_ice_vel_phase_map_v01.nc

   *reference:
      > Mouginot, J., E. Rignot, and B. Scheuchl. (2019). MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PZ3NJ5RXRH10. 

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

5. MEaSUREs BedMachine Antarctica, Version 3 (NSIDC-0756)
   *website: https://nsidc.org/data/nsidc-0756/versions/3

   *filename: BedMachineAntarctica-v3.nc

   *reference:
   
   > Morlighem, M. (2022). MEaSUREs BedMachine Antarctica, Version 3 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/FPSU0V1MWUB6. Date Accessed 02-17-2023.




# Contents
The inversion code is in the *inverse-model* directory.
## Source files
The inversion code is organized in several python files in the *inverse-model/source* directory as follows.

1. **inversion.py** is the inverse problem solver: this defines the normal equations
and calls the conjugate gradient solver.

2. **conj_grad.py** defines the conjugate gradient method that is used to solve
the normal equations.

3. **operators.py** defines the forward and adjoint operators that appear in the
normal equations.

4. **kernel_fcns.py** defines the relaxation and transfer functions that the forward and adjoint operators depend on.

5. **regularizations.py** defines the regularization.

6. **params.py** defines all of the model options and parameters.

There are two auxiliary functions as well:

7. **localization.py** defines a function that remove the off-lake component from fields when necessary.

8. **post_process.py** defines functions to estimate the water-volume change
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

Plotting can  be modified in the **plot_results.py** script.

## Miscellaneous
There is also some FEniCS (https://fenicsproject.org/) code in the *misc* directory
for producing synthetic data with a fully nonlinear finite element model. See the repository
agstub/grounding-line-methods for a description. The version herein assumes
radial symmetry.

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

Then, set the lake name ('lake_name') in **proc_icesat_data.py** to one of the options in
the inventory and run the program (this will take some time).
To see a list of the lake names in the inventory run the python script:
```
  from load_lakes import gdf
  gdf['name'].to_list()
```

Finally, run **run_example.py** and enter the lake name
when prompted.
