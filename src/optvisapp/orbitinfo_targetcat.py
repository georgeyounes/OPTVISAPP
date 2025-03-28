import argparse

import numpy as np
import sys
import pandas as pd

from optvisapp import ags_iss, observing_geometry
from optvisapp.optvisapp_logging import get_logger

sys.dont_write_bytecode = True

# Log config
############
logger = get_logger(__name__)


def orbitinfo_targetcat(catalog_name, ags3, iss_orbit_file, start_time, end_time):
    """
    Add orbit info and sun angle to target catalog
    :param catalog_name: NICER visibility file
    :type catalog_name: str
    :param ags3: NICER visibility file
    :type ags3: str
    :param iss_orbit_file: iss OEM ephem orbit file
    :type iss_orbit_file: str
    :param start_time: Start time in the same format as visibility windows in AGS3 ('%Y-%jT%H:%M:%S', e.g., 2025-075T00:00:00)
    :type start_time: str
    :param end_time: End time in the same format as visibility windows in AGS3 ('%Y-%jT%H:%M:%S', e.g., 2025-076T00:00:00)
    :type end_time: str
    """
    # Reading target catalog, visibility window, and ISS orbit data
    targetcat_df, targetcat_df_nosourceduplicates, targetcat_header = ags_iss.read_target_catalog(catalog_name)
    #
    df_nicer_vis = ags_iss.read_ags3_vis_file(ags3)
    #
    issorbitdata = ags_iss.read_iss_oem_ephem(iss_orbit_file)

    # Filtering NICER visibility according to start_time and end_time
    times_to_filter = pd.Series({'start_time': [start_time], 'end_time': [end_time]})
    times_to_filter['start_time'] = pd.to_datetime(times_to_filter['start_time'], format='%Y-%jT%H:%M:%S', utc=True)
    times_to_filter['end_time'] = pd.to_datetime(times_to_filter['end_time'], format='%Y-%jT%H:%M:%S', utc=True)
    df_nicer_vis_filtered = df_nicer_vis[df_nicer_vis['vis_start'].between(times_to_filter['start_time'][0],
                                                                           times_to_filter['end_time'][
                                                                               0])].reset_index()

    # List of catalog sources with orbit_day visibility windows
    catalog_orbit_day = []
    for target in targetcat_df_nosourceduplicates['Source']:
        # Filter for target name and get visibility windows
        nicer_vis_windows_src = df_nicer_vis_filtered[
            df_nicer_vis_filtered['target_name'].str.contains(target, regex=False)].reset_index(drop=True)

        if nicer_vis_windows_src.empty:
            logger.info(
                'Target {} does not have any visibility windows during the requested start and stop times'.format(
                    target))
            continue
        else:
            # Merge visibility windows into a single Series and sort
            merged_vis_series = pd.concat([nicer_vis_windows_src['vis_start'],
                                           nicer_vis_windows_src['vis_end']]).sort_values().reset_index(drop=True)

            # Calculate orbit status for each (i.e., start and end of each visibility window)
            # Here I assume that there is no visibility window that starts in orbit_night, crosses day, and back to
            # orbit_night uninterrupted (this would lead to the erronous detection of no orbit_day in this algorithm)
            vis_windows = observing_geometry.iss_islit(merged_vis_series, issorbitdata)
            src_orbit_day = ('o_d' if ((vis_windows == 1).any()) else 'o_n')

            # If any visibility window is in orbitday, save this info along with Sun angles
            if src_orbit_day == 'o_d':
                target_with_od = targetcat_df_nosourceduplicates[
                    targetcat_df_nosourceduplicates['Source'] == target].copy()
                target_with_od['orbit'] = src_orbit_day

                target_with_od_ra = target_with_od['RAJ_DEG']
                target_with_od_dec = target_with_od['DECJ_DEG']

                time_start_end_allvis = np.array([merged_vis_series.head(1).item().to_julian_date() - 2400000.5,
                                                  merged_vis_series.tail(1).item().to_julian_date() - 2400000.5])

                sunangles = observing_geometry.sunangle(time_start_end_allvis, target_with_od_ra.item(),
                                                        target_with_od_dec.item())

                target_with_od['sunangle_start'] = sunangles[0]
                target_with_od['sunangle_end'] = sunangles[1]
                target_with_od['sunangle_trend'] = (
                    'Decreasing' if ((sunangles[0] - sunangles[1]) > 0) else 'Increasing')

                catalog_orbit_day.append(target_with_od)  # Collect in list

    full_df = pd.concat(catalog_orbit_day, ignore_index=False)

    # Adding header to the different target catalog dataframes
    targetcat_df = pd.concat([targetcat_header, targetcat_df])
    targetcat_od_df = pd.concat([targetcat_header, full_df])

    # Creating an Excel file with the two different sheets
    excel_filename = catalog_name.split('.')[0] + '.xlsx'
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        targetcat_df.to_excel(writer, sheet_name="target_catalog", index=True)  # Save df1 in "Sheet1"
        targetcat_od_df.to_excel(writer, sheet_name="target_catalog_orbitday", index=True)  # Save df2 in "Sheet2"

    return


def main():
    parser = argparse.ArgumentParser(
        description="Provide orbit status and Sun angle information to NICER target catalog")
    parser.add_argument("catalog_name", help="A NICER source catalog name ISS", type=str)
    parser.add_argument("ags3", help="A NICER AGS3 visibility file", type=str)
    parser.add_argument("iss_orbit_file", help="A ISS orbit file (OEM format)", type=str)
    parser.add_argument("start_time", help="Filter visibilities starting from this start_time (same "
                                           "format as visibility windows in AGS3; Y-jTH:M:S, e.g., "
                                           "2025-075T00:00:00)", type=str)
    parser.add_argument("end_time", help="Filter visibilities ending at this end_time (same format as "
                                         "visibility windows in AGS3; Y-jTH:M:S, e.g., "
                                         "2025-075T00:00:00)", type=str)
    args = parser.parse_args()

    # Raise exception if start_time is after end_time
    time_data_dt = pd.to_datetime(pd.Series([args.start_time, args.end_time]), format='%Y-%jT%H:%M:%S', utc=True)
    if time_data_dt.iloc[0] >= time_data_dt.iloc[1]:
        raise ValueError("Invalid timestamp detected: 'start_time' must be earlier than 'end_time'.")

    # Running primary function
    orbitinfo_targetcat(args.catalog_name, args.ags3, args.iss_orbit_file, args.start_time, args.end_time)

    output_targetcatalog = args.catalog_name.split('.')[0] + '.xlsx'
    print('Script ran to completion. Please check {} for output.'.format(output_targetcatalog))

    return None


if __name__ == '__main__':
    main()
