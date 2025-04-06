"""
Tools to work out the visibility windows of NICER targets, their Sun angles, bright_earth angles, while indicating
ISS orbit-day/night boundaries. All functions are wrapped in a script, plotvisibilities, which produces an interactive
plotly visualization of all.

Warning: bright_earth angle calculation takes time, I highly recommend plotting short time windows if you are
interested in that information as well.
"""

import argparse
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
from matplotlib import colormaps

from astropy.time import Time
import plotly.graph_objects as go

from optvisapp import ags_iss, observing_geometry
from optvisapp.optvisapp_logging import get_logger

import time
import os

import sys

sys.dont_write_bytecode = True

# Log config
############
logger = get_logger(__name__)


def visibilitytargetcat(catalog_name, ags3, start_time, end_time, freq_bound=60, freq_brearth=240, sa_ll=70,
                        sa_ul=180, outputFile='visibilities', saveresults=True, saveplot=False):
    """
    Read-in ISS OEM ephemeris as a dataframe
    :param catalog_name: NICER target catalog file
    :type catalog_name: str
    :param ags3: NICER visibility file
    :type ags3: str
    :param start_time: Start time in the same format as visibility windows in AGS3 ('%Y-%jT%H:%M:%S', e.g., 2025-075T00:00:00)
    :type start_time: str
    :param end_time: End time in the same format as visibility windows in AGS3 ('%Y-%jT%H:%M:%S', e.g., 2025-076T00:00:00)
    :type end_time: str
    :param freq_bound: frequency of timestamps between start_time and end_time in seconds
    :type freq_bound: int
    :param sa_ll: Lower-limit on sun angle for target_id acceptance
    :type sa_ll: float
    :param sa_ul: upper-limit on sun angle for target_id acceptance
    :type sa_ul: float
    :param freq_brearth: frequency of timestamps for bright earth calculation in seconds
    :type freq_brearth: int
    :param outputFile: name of output file (default = "visibilities")
    :type outputFile: str
    :param saveresults: Boolean to save dataframes to csv files (default = "False")
    :type saveresults: bool
    :param saveplot: Boolean to save results to a static html plot (only for debugging, default = False)
    :type saveplot: bool
    """

    logger.info('\n Running function visibilitytargetcat with input parameters: '
                '\n Target catalog: ' + str(catalog_name) +
                '\n AGS3 visibility file: ' + str(ags3) +
                '\n Start time of interest: ' + str(start_time) +
                '\n End time of interest: ' + str(end_time) +
                '\n Sun angle lower-limit: ' + str(sa_ll) +
                '\n Sun angle upper-limit: ' + str(sa_ul) +
                '\n Output files prefix: ' + str(outputFile) +
                '\n Save results to .parquet?: ' + str(saveresults) +
                '\n Save static .html plot: ' + str(saveplot) + '\n')

    # Reading target catalog
    targetcat_df, targetcat_df_nosourceduplicates, _ = ags_iss.read_target_catalog(catalog_name)

    # Reading ISS OEM file
    # Check if iss_orbit_file="ISS.OEM_J2K_EPH.txt" is in current directory, if not download it
    iss_orbit_file = "ISS.OEM_J2K_EPH.txt"
    if os.path.exists(iss_orbit_file):
        issorbitdata = ags_iss.read_iss_oem_ephem(iss_orbit_file)
    else:
        iss_orbit_file = ags_iss.downloadissoemfile()
        issorbitdata = ags_iss.read_iss_oem_ephem(iss_orbit_file)

    # Reading AGS catalog
    # Drop duplicates of exact target_name and visibility windows, keep first
    # warning: these targets have different target_IDs, only first ID kept
    _, df_nicer_vis_nosrcdulpicate = ags_iss.read_ags3_vis_file(ags3)

    # Define times of interest
    start_timeofint = pd.to_datetime(start_time, format='%Y-%jT%H:%M:%S', utc=True)  # Start of x-axis
    end_timeofint = pd.to_datetime(end_time, format='%Y-%jT%H:%M:%S', utc=True)  # End of x-axis

    # filter nicer visibility accoding to times
    df_nicer_vis_timeflt = filtertime_nicervis(start_timeofint, end_timeofint, df_nicer_vis_nosrcdulpicate)

    # If no visibility exists at all after time filtering
    if df_nicer_vis_timeflt.empty:
        logger.error('No targets that satisfy the time selection criteria - Exiting')
        raise Exception('No targets that satisfy the time selection criteria - Exiting')

    # Define time stamps for orbit
    orbit_range = orbittimes(start_timeofint, end_timeofint, freq_bound)

    # Calculating orbit status
    df_day_night, od_windows = orbitstatus(issorbitdata, orbit_range)

    #####################################
    # This is really the core of the code
    #####################################
    # Initialize an empty list for all targets' bright_earth angles
    target_brightearth_all_list = []
    # Initialize an empty list for all targets' bright_earth angles
    target_sunangle_all_list = []
    # Initialize empty list for sun angle filtering (sources outside of range)
    target_outside_sunangle = []
    # Initialize empty list for od start and end times per target
    target_od_startend_times_all_list = []

    for targetid in df_nicer_vis_timeflt['target_id']:

        # Filter target catalog
        targetcat_df_srcflt = targetcat_df_nosourceduplicates[targetcat_df_nosourceduplicates['ID'] ==
                                                              targetid].reset_index(drop=True)

        # Check if target is in target catalog
        if targetcat_df_srcflt.empty:
            logger.error('Target ID {} is not in target catalog, yet appears in visibility files - '
                         'ingest it into target catalog'.format(targetid))
            raise Exception('Target ID {} is not in target catalog, yet appears in visibility files - '
                            'ingest it into target catalog'.format(targetid))

        # Times at which to calculate Sun angles
        time_start_end_allvis = np.array([start_timeofint.to_julian_date() - 2400000.5,
                                          end_timeofint.to_julian_date() - 2400000.5])

        # Calculate sun angles
        sunangles = observing_geometry.sunangle(time_start_end_allvis, targetcat_df_srcflt['RAJ_DEG'].loc[0].item(),
                                                targetcat_df_srcflt['DECJ_DEG'].loc[0].item())

        # Filter out sources with undesired sun angle ranges
        # Done here to avoid expensive bright_earth angle calculation
        if (sunangles[0] < sa_ll) or (sunangles[0] > sa_ul):
            target_outside_sunangle.append(targetid)
            continue

        trend = 'Dec' if (sunangles[0] - sunangles[1]) > 0 else 'Inc'

        # Append results in a dictionary
        target_sunangle_all_list.append({
            'target_id': targetid,  # keep original target ID
            'sunangle_start': sunangles[0],
            'sunangle_end': sunangles[1],
            'sunangle_trend': trend
        })

        # Bright_earth calculation
        targetbright_earth_df, target_od_startend_times = bright_earth_targetvis(issorbitdata, df_nicer_vis_timeflt,
                                                                                 od_windows, targetid,
                                                                                 targetcat_df_srcflt['RAJ_DEG'].loc[
                                                                                     0].item(),
                                                                                 targetcat_df_srcflt['DECJ_DEG'].loc[
                                                                                     0].item(),
                                                                                 freq_brearth=freq_brearth)
        # Concatenate lists to dataframes
        target_brightearth_all_list.append(targetbright_earth_df)
        target_od_startend_times_all_list.append(target_od_startend_times)

    # Filter out the targets that landed outside the user-defined sun angle range
    df_nicer_vis_timeflt = df_nicer_vis_timeflt[~df_nicer_vis_timeflt['target_id'].isin(target_outside_sunangle)]

    if df_nicer_vis_timeflt.empty:
        logger.error('No targets that satisfy the time and sun angle criteria - Exiting')
        raise Exception('No targets that satisfy the time and sun angle criteria - Exiting')

    # Convert the sun angle results list into a DataFrame and append to df_nicer_vis_flt dataframe
    targetsun_angle_results_df = pd.DataFrame(target_sunangle_all_list)

    df_nicer_vis_timeflt = df_nicer_vis_timeflt.merge(targetsun_angle_results_df, on='target_id', how='left')

    # Convert the bright_earth angle results list into a DataFrame
    target_brightearth_all_df = pd.concat(target_brightearth_all_list, axis=0).reset_index(drop=True)

    # Convert the od start-stop times list into a DataFrame
    target_od_startend_times_all = pd.concat(target_od_startend_times_all_list, axis=0).reset_index(drop=True)

    if saveresults and outputFile is not None:
        df_nicer_vis_timeflt.to_parquet(outputFile + '_vis.parquet')
        target_brightearth_all_df.to_parquet(outputFile + '_brightearth.parquet')
        target_od_startend_times_all.to_parquet(outputFile + '_od_startend_times.parquet')
        od_windows.to_parquet(outputFile + '_odbounds.parquet')
    else:
        logger.info('Dataframe results not saved to .parquet output files. If saveresults is set to True, ensure '
                    'outputFile is not None.')

    # Create a plotly html file - interactive plot
    if saveplot and outputFile is not None:
        visibilityplot_plotly(df_nicer_vis_timeflt, target_brightearth_all_df, target_od_startend_times_all,
                              od_windows, start_timeofint, end_timeofint, freq_bound=60, outputFile=outputFile)
    else:
        logger.info('No visibility plot created. If saveplot is set to True, ensure outputFile is not None.')

    return df_nicer_vis_timeflt, target_brightearth_all_df, target_od_startend_times_all, od_windows


def orbittimes(start_time, end_time, freq_bound):
    """
    Create timestamps between start_time and end_time
    :param start_time: Start time in format ('%Y-%jT%H:%M:%S', e.g., 2025-075T00:00:00)
    :type start_time: pandas.Timestamp
    :param end_time: End time in format ('%Y-%jT%H:%M:%S', e.g., 2025-076T00:00:00) -
    best if start_time and end_times are at a 1-minute resolution
    :type end_time: pandas.Timestamp
    :param freq_bound: frequency of timestamps between start_time and end_time in seconds
    :type freq_bound: int
    :return orbit_times: Orbit times
    :rtype: pandas.DatetimeIndex
    """
    # Define time stamps for orbit between start and end times at frequency freq_bound in seconds
    orbit_times = pd.date_range(start=start_time, end=end_time,
                                freq=str(freq_bound) + 's').to_series().reset_index(drop=True)

    return orbit_times


def orbitstatus(issorbitdata, orbit_times):
    """
    Read-in ISS OEM ephemeris as a dataframe
    :param issorbitdata: iss orbit data from an OEM file
    :type issorbitdata: pandas.DataFrame
    :param orbit_times: Orbit times, timestamps in (format='%Y-%jT%H:%M:%S', utc=True)
    :type orbit_times: pandas.DatetimeIndex
    :return df_day_night: dataframe of orbit status (1: day and 0: night)
    :rtype: pandas.DataFrame
    :return od_windows: Start and end times that define orbit day boundaries
    :rtype: pandas.DataFrame
    """
    # Clculating which part of orbit_times are in orbit day
    is_orbitday = observing_geometry.iss_islit(orbit_times, issorbitdata)

    # Merge is_orbitday with orbit_range
    df_day_night = pd.DataFrame({
        'timestamps': orbit_times,  # Assign timestamps
        'inorbitday': is_orbitday  # Assign NumPy array
    })

    # Find start and end timestamps for continuous blocks
    df_day_night['block_change'] = df_day_night['inorbitday'].diff().ne(0).cumsum()  # Group changes
    blocks = df_day_night.groupby('block_change')

    start_end_list = []

    for _, block in blocks:
        # Single out orbitday blocks
        if block['inorbitday'].iloc[0] == 1:  # Only process blocks where 'inorbitday' == 1
            start_time = block['timestamps'].iloc[0]
            end_time = block['timestamps'].iloc[-1]
            start_end_list.append({'start_time': start_time, 'end_time': end_time})

    if start_end_list:
        # Convert list of series into a DataFrame
        od_windows = pd.DataFrame(start_end_list)
    else:
        od_windows = None

    return df_day_night, od_windows


def filtertime_nicervis(start_time, end_time, df_nicer_vis):
    """

    :param start_time: Start time in format ('%Y-%jT%H:%M:%S', e.g., 2025-075T00:00:00)
    :type start_time: pandas.Timestamp
    :param end_time: End time in format ('%Y-%jT%H:%M:%S', e.g., 2025-076T00:00:00) -
    best if start_time and end_times are at a 1-minute resolution
    :type end_time: pandas.Timestamp
    :param df_nicer_vis: dataframe of nicer visibility windows (AGS 3 file)
    :type: pandas.DataFrame
    :return df_nicer_vis_timefilt: time-filtered visibility table
    :rtype: pandas.DataFrame
    """
    df_nicer_vis_timefilt = df_nicer_vis[
        df_nicer_vis['vis_start'].between(start_time, end_time, inclusive='neither') |
        df_nicer_vis['vis_end'].between(start_time, end_time, inclusive='neither')
        ].reset_index()

    return df_nicer_vis_timefilt


def bright_earth_targetvis(issorbitdata, df_nicer_vis, od_windows, tragetid, srcRA, srcDEC, freq_brearth=240):
    """
    Calculates orbitday files
    :param issorbitdata: ISS OEM ephemeris in dataframe format
    :type issorbitdata: pandas.DataFrame
    :param df_nicer_vis: NICER visibility
    :type df_nicer_vis: pandas.DataFrame
    :param tragetid: unique identifier of a target
    :type tragetid: str
    :param od_windows: Start and end times that define orbit day boundaries
    :type od_windows: pandas.DataFrame
    :param srcRA: Right ascension in degrees J2000
    :type srcRA: float
    :param srcDEC: Declination in degrees J2000
    :type srcDEC: float
    :param freq_brearth: frequency of timestamps for bright earth calculation in seconds
    :type freq_brearth: int
    :return targetbright_earth_df: Target bright_earth_angle for each timestamp during orbit-day intervals only
    :rtype: pandas.DataFrame
    """
    # List of visibility window during orbit-day
    od_source_all_vis_windows_list = []
    target_od_startend_times_list = []

    # Filter for source
    df_nicer_vis_srcflt = df_nicer_vis[df_nicer_vis['target_id'] == tragetid].reset_index(drop=True)
    # Initiate a temporary visibility window empty dataframe
    df_nicer_vis_srcflt_tmp = pd.DataFrame(columns=['vis_start', 'vis_end'], index=[0])

    # Loop over each orbit_day window
    for ii in od_windows.index:
        # Loop over each visibility window
        for jj in df_nicer_vis_srcflt.index:

            # Change start and/or end time of temp visibility window to match od_windows
            if df_nicer_vis_srcflt['vis_end'].loc[jj] <= od_windows['start_time'].loc[ii]:
                # If the start and end_times of the visibility window are before start of orbit_day, simply skip
                continue

            elif ((df_nicer_vis_srcflt['vis_start'].loc[jj] < od_windows['start_time'].loc[ii]) and
                  (df_nicer_vis_srcflt['vis_end'].loc[jj] > od_windows['start_time'].loc[ii]) and
                  (df_nicer_vis_srcflt['vis_end'].loc[jj] < od_windows['end_time'].loc[ii])):
                # If the start of the visibility window is before start of orbit_day, but ends within it
                df_nicer_vis_srcflt_tmp.loc[0, "vis_start"] = od_windows['start_time'].loc[ii]
                df_nicer_vis_srcflt_tmp.loc[0, "vis_end"] = df_nicer_vis_srcflt.loc[jj, "vis_end"]

            elif ((df_nicer_vis_srcflt['vis_start'].loc[jj] < od_windows['start_time'].loc[ii]) and
                  (df_nicer_vis_srcflt['vis_end'].loc[jj] > od_windows['end_time'].loc[ii])):
                # If the start of the visibility window is before start of orbit_day, but ends after it
                df_nicer_vis_srcflt_tmp.loc[0, "vis_start"] = od_windows['start_time'].loc[ii]
                df_nicer_vis_srcflt_tmp.loc[0, "vis_end"] = od_windows['end_time'].loc[ii]

            elif (((df_nicer_vis_srcflt['vis_start'].loc[jj] >= od_windows['start_time'].loc[ii]) and
                   (df_nicer_vis_srcflt['vis_start'].loc[jj] < od_windows['end_time'].loc[ii])) and
                  (df_nicer_vis_srcflt['vis_end'].loc[jj] > od_windows['end_time'].loc[ii])):
                # If the start of the visibility window is within start and end of orbit_day, but ends after it
                df_nicer_vis_srcflt_tmp.loc[0, "vis_start"] = df_nicer_vis_srcflt.loc[jj, "vis_start"]
                df_nicer_vis_srcflt_tmp.loc[0, "vis_end"] = od_windows['end_time'].loc[ii]

            elif ((df_nicer_vis_srcflt['vis_start'].loc[jj] >= od_windows['start_time'].loc[ii]) and
                  (df_nicer_vis_srcflt['vis_end'].loc[jj] <= od_windows['end_time'].loc[ii])):
                # If the start and end of visibility window is within start and end of orbit_day
                df_nicer_vis_srcflt_tmp.loc[0, "vis_start"] = df_nicer_vis_srcflt.loc[jj, "vis_start"]
                df_nicer_vis_srcflt_tmp.loc[0, "vis_end"] = df_nicer_vis_srcflt.loc[jj, "vis_end"]

            elif df_nicer_vis_srcflt['vis_start'].loc[jj] >= od_windows['end_time'].loc[ii]:
                # If the start of the visibility window is after the end of orbit_day, skip
                continue

            # First let's retain the information of orbit_day visibility windows per traget
            # Append results in a dictionary
            target_od_startend_times_list.append({
                'target_id': df_nicer_vis_srcflt['target_id'].loc[jj],  # keep original target id
                'target_name': df_nicer_vis_srcflt['target_name'].loc[jj],  # keep original target name
                'od_vis_start': df_nicer_vis_srcflt_tmp['vis_start'].loc[0],
                'od_vis_end': df_nicer_vis_srcflt_tmp['vis_end'].loc[0]
            })

            duration_vis_window = (df_nicer_vis_srcflt_tmp['vis_end'].loc[0] -
                                   df_nicer_vis_srcflt_tmp['vis_start'].loc[0]).total_seconds()
            if duration_vis_window >= 240:
                od_source_vis_windows = pd.date_range(start=df_nicer_vis_srcflt_tmp['vis_start'].loc[0],
                                                      end=df_nicer_vis_srcflt_tmp['vis_end'].loc[0],
                                                      freq=str(freq_brearth) + 's')
            else:
                od_source_vis_windows = pd.date_range(start=df_nicer_vis_srcflt_tmp['vis_start'].loc[0],
                                                      end=df_nicer_vis_srcflt_tmp['vis_end'].loc[0],
                                                      freq=str(duration_vis_window) + 's')

            # Ensure the last timestamp matches 'end', otherwise append it
            if od_source_vis_windows[-1] != df_nicer_vis_srcflt_tmp['vis_end'].loc[0]:
                od_source_vis_windows = od_source_vis_windows.append(
                    pd.Index([df_nicer_vis_srcflt_tmp['vis_end'].loc[0]]))

            # Convert to series
            od_source_vis_windows = od_source_vis_windows.to_series().reset_index(drop=True)

            # Putting all od visibility windows into a list
            od_source_all_vis_windows_list.append(od_source_vis_windows)

    if not od_source_all_vis_windows_list:
        return None, None
    else:
        # orbit_day full visibility window sampled at frequency freq_brearth
        od_source_all_vis_windows = pd.concat(od_source_all_vis_windows_list, ignore_index=True)
        # orbit_day start-end times
        target_od_startend_times = pd.DataFrame(target_od_startend_times_list).reset_index(drop=True)

    # Let's now measure bright earth angles at each time stamp
    ##########################################################
    be_angle_list = []
    for kk in od_source_all_vis_windows.index:
        # Get ISS cartesian coordinates at exact time stamp
        iss_cartesian = observing_geometry.extrapolate_iss_position(issorbitdata, od_source_all_vis_windows.loc[kk])

        # Get bright earth angle
        be_timestamp_astropy = Time(od_source_all_vis_windows.loc[kk].strftime('%Y-%m-%dT%H:%M:%S'), format='isot',
                                    scale='utc')
        be_angle = observing_geometry.bright_earth_angle(iss_cartesian, be_timestamp_astropy, srcRA, srcDEC)

        be_angle_list.append(be_angle)

    # Put all information in a coherent dataframe
    targetbright_earth_df = pd.DataFrame({
        'target_id': [tragetid] * len(be_angle_list),  # Assign same source id to all rows
        'target_od_time': od_source_all_vis_windows,  # Store timestamps
        'brightearth': be_angle_list  # Store corresponding scalar values
    })

    return targetbright_earth_df, target_od_startend_times


def visibilityplot_plotly(nicer_vis, target_brightearth, alltargets_od_startend_times, od_windows, start_time,
                          end_time, freq_bound=60, outputFile='visibilities'):
    """
    :param nicer_vis: NICER target visibilities
    :type nicer_vis: pandas.DataFrame
    :param target_brightearth: bright-earth angle for all targets
    :type target_brightearth: pandas.DataFrame
    :param alltargets_od_startend_times: per target orbit-day start-end times visibility windows
    :type alltargets_od_startend_times: pandas.DataFrame
    :param start_time: Start time of visibility window
    :type start_time: pandas.Timestamp
    :param end_time: End time of visibility window
    :type end_time: pandas.Timestamp
    :param od_windows: orbit-day windows (essentially day-night boundaries)
    :type od_windows: pandas.DataFrame
    :param freq_bound: frequency of timestamps between start_time and end_time in seconds
    :type freq_bound: int
    :param outputFile: name of output file (default = "visibilities")
    :type outputFile: str or None
    """
    # Create combined y-axis labels in the format (target_ID, target_name, sunangle_start, sunangle_trend)
    nicer_vis = nicer_vis.copy()
    nicer_vis["label"] = nicer_vis.apply(
        lambda row: f"({row['target_id']}, {row['target_name']}, {row['sunangle_start']:.2f}, {row['sunangle_trend']})",
        axis=1
    )

    # Map these labels to evenly spaced y positions between 0 and 1
    unique_labels = nicer_vis["label"].unique()
    y_positions = np.linspace(0, 1, len(unique_labels))
    label_mapping = {label: pos for label, pos in zip(unique_labels, y_positions)}

    label_to_name = {}
    for _, row in nicer_vis.iterrows():
        label = f"({row['target_id']}, {row['target_name']}, {row['sunangle_start']:.2f}, {row['sunangle_trend']})"
        label_to_name[label] = row['target_id']

    # -----------------------------
    # Build Plotly Figure
    # -----------------------------
    fig = go.Figure()

    # 1) Timeline lines (black horizontal lines)
    for _, row in nicer_vis.iterrows():
        y_val = label_mapping[row["label"]]
        fig.add_trace(go.Scatter(
            x=[row['vis_start'], row['vis_end']],
            y=[y_val, y_val],
            mode='lines',
            line=dict(color='black', width=5),
            showlegend=False
        ))

    # 2) Day windows as shapes
    for _, block in od_windows.iterrows():
        od_start_time = block['start_time']
        od_end_time = block['end_time']
        color = "yellow"
        fig.add_shape(
            type="rect",
            x0=od_start_time,
            x1=od_end_time,
            y0=0,  # from bottom to top of the entire plot
            y1=1,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0
        )

    # 3) Red vertical lines for times of interest
    fig.add_shape(
        type="line",
        x0=start_time, x1=start_time,
        y0=0, y1=1,
        line=dict(color='red', width=2, dash='dash'),
        name="Start Time"
    )
    fig.add_shape(
        type="line",
        x0=end_time, x1=end_time,
        y0=0, y1=1,
        line=dict(color='red', width=2, dash='dash'),
        name="End Time"
    )

    # 4) Brightearth intervals
    # We'll do a simple approach: interpret each consecutive pair as an interval
    # with color mapped to the brightearth of row i.
    targets_be_angle = target_brightearth.drop_duplicates(subset=['target_id', 'target_od_time'], keep='first')
    # Group by srcname, then plot intervals from row i to row i+1 while respecting source orbit-day visibilities
    for src, group in targets_be_angle.groupby('target_id'):
        group = group.sort_values('target_od_time').reset_index(drop=True)

        # Filter per target od start-end visibility windows
        target_od_startend_times = alltargets_od_startend_times[alltargets_od_startend_times['target_id'] ==
                                                                group['target_id'].head(1).values[0]]
        target_od_startend_times = target_od_startend_times.reset_index(drop=True)

        for od_ind in target_od_startend_times.index:
            # Filter bright_earth_angle calculation per-target per orbit-day visibility window
            group_target_od = group[group['target_od_time'].between(
                target_od_startend_times.loc[od_ind, 'od_vis_start'],
                target_od_startend_times.loc[od_ind, 'od_vis_end'])].reset_index()

            # Loop until second-to-last row
            for i, (_, row) in enumerate(group_target_od.iterrows()):

                # Skip last bin; we are plotting from i to i+1
                if i == len(group_target_od) - 1:
                    break

                targetvis_start_time = group_target_od.loc[i, 'target_od_time']
                targetvis_end_time = group_target_od.loc[i + 1, 'target_od_time']

                # Map brightearth -> color
                # Let's do a continuous scale from 40..150. We'll use Matplotlib for convenience
                br_val = group_target_od.loc[i, 'brightearth']

                # cmap = colormaps['Set1']
                # norm = mcolors.Normalize(vmin=40, vmax=150)
                # rgba = cmap(norm(br_val))  # returns (r, g, b, a)
                # color_hex = mcolors.to_hex(rgba)

                cmap = colormaps['Set1']
                boundaries = [40, 60, 70, 80, 90, 100, 110, 120, 130, 180]
                # ncolors should typically match the number of intervals (here 9 intervals)
                norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
                rgba = cmap(norm(br_val))  # returns (r, g, b, a)
                color_hex = mcolors.to_hex(rgba)

                # Y-value: match the timeline's y
                # We'll assume row['target_id'] => 'Target A' => find the label (and it should)
                y_val_candidates = [lab for lab in label_mapping if label_to_name[lab] ==
                                    group_target_od.loc[i, 'target_id']]

                if not y_val_candidates:
                    continue
                y_val = label_mapping[y_val_candidates[0]]

                fig.add_trace(go.Scatter(
                    x=[targetvis_start_time, targetvis_end_time],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=color_hex, width=5),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"brightearth={br_val}; {y_val_candidates[0]}"
                ))

    # Set layout
    fig.update_layout(
        title="Target Visibility Timeline",
        xaxis_title="Time",
        yaxis_title="Target (ID, Name, SA, SA trend)",
        xaxis=dict(type='date'),
        yaxis=dict(
            tickvals=list(label_mapping.values()),
            ticktext=list(label_mapping.keys()),
            range=[-0.02, 1.02]  # a bit of padding
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Rotate x-ticks
    fig.update_xaxes(tickangle=45, range=[start_time - pd.Timedelta(seconds=freq_bound / 2),
                                          end_time + pd.Timedelta(seconds=freq_bound / 2)])

    fig.layout.autosize = True

    if isinstance(outputFile, str):
        fig.write_html(f'./{outputFile}.html')

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Provide orbit status and Sun angle information to NICER target catalog")
    parser.add_argument("catalog_name", help="A NICER target catalog", type=str)
    parser.add_argument("ags3", help="A NICER AGS3 visibility file", type=str)
    parser.add_argument("start_time", help="Start of visibilities (Y-jTH:M:S, e.g., "
                                           "2025-075T00:00:00)", type=str)
    parser.add_argument("end_time", help="End of visibilities (Y-jTH:M:S, e.g., "
                                         "2025-075T00:00:00)", type=str)
    parser.add_argument("-fb", "--freq_bound", help="Frequency to calculate orbit_day boundaries "
                                                    "(default=60 in seconds)", type=int, default=60)
    parser.add_argument("-fr", "--freq_brearth", help="Frequency to calculate bright earth angles "
                                                      "(default=240 in seconds)", type=int, default=240)
    parser.add_argument("-sl", "--sa_ll", help="Sun angle lower range for target filtering "
                                               "(default=70 in degrees)", type=int, default=70)
    parser.add_argument("-su", "--sa_ul", help="Sun angle upper range for target filtering "
                                               "(default=180 in degrees)", type=int, default=180)
    parser.add_argument("-of", "--outputFile", help="Name of output visibility plot/files "
                                                    "(default = visibilities(.html))", type=str,
                        default='visibilities')
    parser.add_argument("-sr", "--saveresults", help="Boolean to save dataframes to parquet files "
                                                     "(default = True)", type=bool, default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-sp", "--saveplot", help="Boolean to save results to a static html plot (only "
                                                  "for debugging, default = False)", type=bool, default=False,
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    execution_start_time = time.time()

    # Raise exception if start_time is after end_time
    time_data_dt = pd.to_datetime(pd.Series([args.start_time, args.end_time]), format='%Y-%jT%H:%M:%S', utc=True)
    if time_data_dt.iloc[0] >= time_data_dt.iloc[1]:
        raise ValueError("Invalid timestamp detected: 'start_time' must be earlier than 'end_time'.")
    elif (time_data_dt.iloc[1] - time_data_dt.iloc[0]).total_seconds() > 14400:
        raise ValueError("Invalid timestamp detected: 'start_time' and 'end_time' are too far apart, limit to "
                         "14400 seconds (4 hours).")

    visibilitytargetcat(args.catalog_name, args.ags3, args.start_time, args.end_time, args.freq_bound,
                        args.freq_brearth, args.sa_ll, args.sa_ul, args.outputFile, args.saveresults, args.saveplot)

    execution_end_time = time.time()
    print(' Script ran to completion in {} minutes.'.format((execution_end_time - execution_start_time) / 60))

    return None


if __name__ == '__main__':
    main()
