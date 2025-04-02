# OPTVISAPP - Optimized Target Visibilities App

## Description

Target visiblity visualization application including the calculation of satellite orbit day/night boudaries, target 
sun angles, and bright earth angles. Intended for the use with the NICER telescope aboard the ISS.

## Installation

At the moment, OPTVISAPP can be installed locally after cloning the directory or simply downloading 
and untarring it. Then from the OPTVISAPP root directory:

```bash
  python -m pip install .
```

You may add the '-e' option to pip to install as an editable package.

The code has been tested on Python 3.9.16

## Disclaimer

OPTVISAPP is unrelated to [NICERDAS](https://heasarc.gsfc.nasa.gov/docs/nicer/nicer_analysis.html), the official data analysis software package that is part of the HEASoft 
distribution. OPTVISAPP is not intended for any NICER data analysis either. Nothing here should be construed as formally 
recommended by the NICER instrument team.

## License

[MIT](https://choosealicense.com/licenses/mit/)