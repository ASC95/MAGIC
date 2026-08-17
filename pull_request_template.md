# Motivation
- These commits add a copy of the omf with the new cyberInverters.py and cyberInverters.html code, new loadshape upsampling code, and scripts to patch
  PyMAGIC code so that it integrates with the omf. These commits represent the full current state of the MAGIC project at the time of writing
# Modifications
- Removed old upsampling code
- Added new upsampling code that uses 1-second shapelets from the RAE dataset instead of 1-minute shapelets from the AMPds2 dataset. The new interpolation
  code also no longer uses a Variational Autoencoder to generate shapelets. The shapelets are sampled straight from the RAE dataset
- The new upsampling code also includes compressed versions of the 1-second resolution load shapes for feeders rhs2_1247--rdt1262 and rhs2_1247--rdt1264
  that were generated with the loadshape_upsampling package
- Added a copy of omf with new cyberInverters.py and cyberInverters.html code. The copy of the omf also includes 4 scenarios: baseline ieee37, ieee37 with
  a VVVW attack, baseline ieee123, and ieee123 with a VVVW attack. The input configurations for running the scenarios are included in testFiles/pyMAGIC and
  the model results of the pre-run models are available in Model/admin
- Added scripts that patch the PyMAGIC repository once it has been pulled by the user into the root of the MAGIC repository
- Added linux_install.py that creates three virtual environments, one for each package. It also installs required apt packages and attempts to download the
  RAE and SMART-DS datasets. The user must still manually download the PyMAGIC repository and must possess the permissions to do so
- Updated README.md documentation