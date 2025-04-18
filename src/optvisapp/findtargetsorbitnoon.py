"""
Create scatter plot of target catalog ra and dec and their corresponding sun angles at a certain time
"""
from optvisapp.ags_iss import read_target_catalog
from optvisapp.observing_geometry import sunangle
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import argparse


def findtargetsorbitnoon(targetcat, nicertimemjd):
    # Reading the target catalog
    _, targetcat_df_withsunangle, _ = read_target_catalog(targetcat)

    # Calculate sun angle of each target
    alltargetsunangle = []
    for _, target in enumerate(targetcat_df_withsunangle):
        targetsunangle = sunangle(nicertimemjd, target['RAJ_DEG'], target['DECJ_DEG'])
        alltargetsunangle.append(targetsunangle)

    targetcat_df_withsunangle['SUN_ANGLE'] = alltargetsunangle

    return targetcat_df_withsunangle


def plottargets(targetcat):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ras = targetcat['RAJ_DEG'].to_numpy()
    ras[np.where(ras > 180)] -= 360
    ax.scatter(ras * u.deg.to('rad'), targetcat['DECJ_DEG'] * u.deg.to('rad'), marker='s', s=50, zorder=2)

    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Provide orbit status and Sun angle information to NICER target catalog")
    parser.add_argument("targetcat", help="A NICER source catalog", type=str)
    parser.add_argument("nicertime", help="A NICER time in Y-jTH:M:S, e.g., "
                                          "2025-075T00:00:00", type=str)
    args = parser.parse_args()

    # Derive sun angles to target catalog
    targetcat_df_withsunangle = findtargetsorbitnoon(args.targetcat, args.nicertime)

    # Create a plot of traget catalog
    plottargets(targetcat_df_withsunangle)

    return None


if __name__ == '__main__':
    main()