"""
Generte targets for NICER visibility calculations either randomly across a
patch of the sky (sky backgrounds) or from existing catalogs (4XMM or eROSITA)
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse

def generate_random_pointings(ra_min, ra_max, dec_min, dec_max, outputFile='random_pointings', n_points=10):
    """
    Generate random sky pointings (RA/Dec in degrees, J2000) and write to a file.
    :param ra_min: Minimum right ascension in degrees
    :type ra_min: float
    :param ra_max: Maximum right ascension in degrees
    :type ra_max: float
    :param dec_min: Minimum declination in degrees
    :type dec_min: float
    :param dec_max: Maximum declination in degrees
    :type dec_max: float
    :param outputFile: Name of output csv random pointings
    :type outputFile: str
    :param n_points: Number of random pointings to generate
    :type n_points: int
    :return pointings: Pointings ra and dec in degrees
    :rtype pointings: astropy.coordinates.SkyCoord
    """
    ra_random = np.random.uniform(ra_min, ra_max, n_points)
    dec_random = np.random.uniform(dec_min, dec_max, n_points)

    pointings = SkyCoord(ra=ra_random * u.deg, dec=dec_random * u.deg, frame='icrs')

    with open(outputFile + '.csv', 'w') as f:
        for i, coord in enumerate(pointings, 1):
            line = f"pointing_{i},{coord.ra.deg:.6f},{coord.dec.deg:.6f}\n"
            f.write(line)

    return pointings


def main():
    parser = argparse.ArgumentParser(description="Create NICER target catalogs")
    parser.add_argument("ra_min", help="Minimum right ascension in degrees J2000", type=float)
    parser.add_argument("ra_max", help="Maximum right ascension in degrees J2000", type=float)
    parser.add_argument("dec_min", help="Minimum declination in degrees DE2000", type=float)
    parser.add_argument("dec_max", help="Maximum declination in degrees DE2000", type=float)
    parser.add_argument("-of", "--outputFile", help="Name of output csv random pointings"
                                                    "(default = random_pointings(.csv))", type=str,
                        default='random_pointings')
    parser.add_argument("-np", "--n_points", help="Number of random pointings to generate (default = 10)",
                        type=int, default=10)
    args = parser.parse_args()

    generate_random_pointings(args.ra_min, args.ra_max, args.dec_min, args.dec_max, args.outputFile, args.n_points)

    return None


if __name__ == '__main__':
    main()