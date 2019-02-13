# `RandomFields`

(Formerly called `aniso`; python package is called `realization`.)

Code (and iPython notebooks) to generate and manipulate realizations of cosmological random fields.

Originally written to study anisotropic models (hence the original name) but has also been used to study
 * velocity fields
 * sub-pixel 2-D aliasing.
 * PDF of non-Gaussianity (see Matsubara 2008)

Note that `velocites.py` has command-line code.

### TODO:
* move driver code out of package?
* non-Gaussianity:
  * can we reverse engineer a non-Gaussian PDF?
  * Start with anisotropic field? (Use code from `aniso.py`)
  * Can we actually see this from Matsubara's formulae?


### DONE:
* Convert to a proper package format
* rename on github?
* Split off "anisotropic" parts of aniso.py
* Conversion to Python3 (Feb 2019)