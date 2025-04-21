"""
Create scatter plot of target catalog ra and dec and their corresponding sun angles at a certain time
"""
from optvisapp.ags_iss import read_target_catalog
from optvisapp.observing_geometry import sunangle
from optvisapp.ags_iss import read_ags3_vis_file

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import argparse
import pandas as pd


def findtargetsorbitnoon(targetcat_df, nicertimemjd):
    # Reading the target catalog
    targetcat_df_withsunangle = targetcat_df.copy()

    # Calculate sun angle of each target
    alltargetsunangle = []
    for _, target in targetcat_df_withsunangle.iterrows():

        targetsunangle = sunangle(nicertimemjd, target['RAJ_DEG'], target['DECJ_DEG'])
        alltargetsunangle.append(targetsunangle)

    targetcat_df_withsunangle['SUN_ANGLE'] = alltargetsunangle

    return targetcat_df_withsunangle


def plottargets(targetcat):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ras = targetcat['RAJ_DEG'].to_numpy()
    ras[np.where(ras > 180)] -= 360
    #ax.scatter(ras * u.deg.to('rad'), targetcat['DECJ_DEG'] * u.deg.to('rad'), marker='s', s=50, zorder=2)

    # scatter with color mapping by SUN_ANGLE
    sc = ax.scatter(
        ras * u.deg.to('rad'), targetcat['DECJ_DEG'] * u.deg.to('rad'),
        c=targetcat['SUN_ANGLE'],  # your third variable
        cmap='viridis',  # or any other matplotlib colormap
        marker='s',
        s=50,
        zorder=2
    )

    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('Sun Angle (degrees)')

    plt.grid()
    plt.show()

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot Provide orbit status and Sun angle information to NICER target catalog")
    parser.add_argument("targetcat", help="A NICER source catalog", type=str)
    parser.add_argument("ags3", help="A NICER AGS3 visibility file", type=str)
    parser.add_argument("start_time", help="Filter visibilities starting from this start_time (same "
                                           "format as visibility windows in AGS3; Y-jTH:M:S, e.g., "
                                           "2025-075T00:00:00)", type=str)
    parser.add_argument("end_time", help="Filter visibilities ending at this end_time (same format as "
                                         "visibility windows in AGS3; Y-jTH:M:S, e.g., "
                                         "2025-075T00:00:00)", type=str)
    args = parser.parse_args()

    # Filter AGS3 according to desired start and end time
    _, df_nicer_vis = read_ags3_vis_file(args.ags3)
    time_data_dt = pd.to_datetime(pd.Series([args.start_time, args.end_time]), format='%Y-%jT%H:%M:%S', utc=True)

    df_nicer_vis_timefilt = df_nicer_vis[
        df_nicer_vis['vis_start'].between(time_data_dt.iloc[0], time_data_dt.iloc[1], inclusive='neither') |
        df_nicer_vis['vis_end'].between(time_data_dt.iloc[0], time_data_dt.iloc[1], inclusive='neither')
        ].reset_index()

    # names=["target_name", "target_id", "vis_start", "vis_end", "Span", "Initial",
    #                                       "Final", "Relative"]

    # Match targetcat sources with the ones that have visibility (from AGS3) during the requested times
    _, targetcat_df, _ = read_target_catalog(args.targetcat)
    targetcat_df_timefilt = targetcat_df[targetcat_df['ID'].isin(df_nicer_vis_timefilt['target_id'])]

    # MJD time of starttime
    timenicermjd = np.array([time_data_dt.iloc[0].to_julian_date() - 2400000.5])

    # Derive sun angles to target catalog
    targetcat_df_withsunangle = findtargetsorbitnoon(targetcat_df_timefilt, timenicermjd)

    # Create a plot of traget catalog
    plottargets(targetcat_df_withsunangle)

    return None


if __name__ == '__main__':
    main()