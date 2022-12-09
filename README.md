[![Project Status: Inactive – TThe project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)

#  3D Moulds for Ovarian Cancer

Repository for the publication "Lesion-Specific 3D-Printed Moulds for Image-Guided Tissue Multi-Sampling of Ovarian Tumours: a prospective pilot study"

## Dependencies
The code in this repository runs on Python and has been developed and tested using Python 3.8.13. Necessary libraries and versions are in the [requirements.txt](requirements.txt) file. Required libraries can be installed by running:
```bash
pip install -r requirements.txt
```

## Inputs
* [*dicom\_info.yaml*](dicom_info.yaml): yaml file specifying:
  1. Path to the folder that contains both the DICOM images files and the DICOM-RT file.
  2. Names given to the ROIs (tumour, base, reference points) in the DICOM-RT file.
* [*tunable\_parameters.yaml*](tunable_parameters.yaml): yaml file specifying the values each of tunable parameter illustrated in Figure X of the manuscript.

Empty templates (with default suggested values for tunable_parameters.yaml) of both files are provided.

## Structure
### Usage of `run.py`:
This is the main script. It is composed of different parts that handle inputs import, DICOM re-slicing and rotation, transformation from DICOM to World Coordinate System (WCS) and tumour and mould modelling. This script connects all the steps in the pipeline and calls to specific functions defined in [utils](utils).
There are multiple options and flags that can be passed to this script.
