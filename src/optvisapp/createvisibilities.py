"""
Module to download NICER enhanced visibilities from the online enhanced visibility tools
(https://heasarc.gsfc.nasa.gov/wsgi-scripts/nicer/visibility/nicervis.wsgi/)
"""

import os
import sys
import time
import argparse
import pandas as pd
from io import StringIO
import gzip

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from optvisapp.optvisapp_logging import get_logger

sys.dont_write_bytecode = True

# Log config
############
logger = get_logger(__name__)


def download_onetarget_visibility(driver, wait, ra, dec, download_dir):
    """Assumes driver and wait are already created; does one download cycle."""
    # ensure download dir
    os.makedirs(download_dir, exist_ok=True)

    # 1) Load form page and fill it out
    driver.get("https://heasarc.gsfc.nasa.gov/wsgi-scripts/nicer/visibility/nicervis.wsgi/")
    wait.until(EC.presence_of_element_located((By.NAME, "ra")))

    driver.find_element(By.NAME, "ra").clear()
    driver.find_element(By.NAME, "ra").send_keys(str(ra))
    driver.find_element(By.NAME, "dec").clear()
    driver.find_element(By.NAME, "dec").send_keys(str(dec))

    sun_checkbox = wait.until(EC.presence_of_element_located((By.NAME, "daysunlim")))
    if sun_checkbox.is_selected():
        sun_checkbox.click()

    day_radio = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//input[@type='radio' and @name='daynightsel' and @value='--day-only']")))
    day_radio.click()

    driver.find_element(By.XPATH,
                        "//input[@type='submit' and @value='Submit']").click()

    # 2) Wait for the “Download Visibilities” button, click it, and wait for the CSV
    wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//input[@type='submit' and @value='Download Visibilities']")))
    driver.find_element(By.XPATH,
                        "//input[@type='submit' and @value='Download Visibilities']").click()

    # simple wait: “.crdownload” appears, then your own wait_for_download_to_finish()
    wait.until(lambda d: any(f.endswith('.crdownload') or f.endswith('.csv')
                             for f in os.listdir(download_dir)))
    return wait_for_download_to_finish(download_dir)


def download_ntarget_visibilities(listofcoordinates, download_dir):

    # Reading list of coordinates
    targetsvisfiles_df = pd.read_csv(listofcoordinates, header=None, names=['Source', 'RAJ_DEG', 'DECJ_DEG'])
    listoffilepaths = []
    target_id = []

    # Initiating a driver
    chrome_options = Options()
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                              options=chrome_options)
    wait = WebDriverWait(driver, 30)

    try:
        for index, row in targetsvisfiles_df.iterrows():
            try:
                # attempt the download
                path = download_onetarget_visibility(
                    driver, wait,
                    row['RAJ_DEG'], row['DECJ_DEG'],
                    download_dir
                )
                logger.info(f"✅ Download succeeded for {row['Source']} → {path}")
            except (TimeoutException, TimeoutError) as e:
                # skip this target if we timed out waiting for .csv
                logger.warning(f"⚠️ Skipping {row['Source']} (RA={row['RAJ_DEG']}, DEC={row['DECJ_DEG']}): {e}")
                path = None
            except Exception as e:
                # catch any other error so one bad run doesn't kill everything
                logger.error(f"❌ Unexpected error for {row['Source']}: {e}", exc_info=True)
                path = None

            listoffilepaths.append(path)
            # still generate an ID for consistency
            target_id.append(int(f"99999{index}"))
    finally:
        driver.quit()

    targetsvisfiles_df['ID'] = target_id
    targetsvisfiles_df.insert(0, 'ID', targetsvisfiles_df.pop('ID'))  # moving column to front of dataframe

    # Create an NICER compatible target catalog
    usertargetfilename = 'nicer_userdefined_targetlist.csv'
    buffer = StringIO()
    targetsvisfiles_df.to_csv(buffer, index=False, header=True)
    buffer.seek(0)
    csv_content = buffer.read()

    # Prepend custom header lines and write the final file
    custom_header = "USER Target IDs and Names,,,\nFlight user targets,,,\n"
    with open(usertargetfilename, 'w') as f:
        f.write(custom_header)
        f.write(csv_content)

    # Finally add filepaths to AGS visibility dataframe
    targetsvisfiles_df['filepaths'] = listoffilepaths

    return targetsvisfiles_df


def read_agsfile_enhancedvisibilitytool(ags3_vis_file, target_name, target_id):
    """
    Read-in AGS NICER visibility file created from the enhanced visibility tool as a dataframe
    :param ags3_vis_file: NICER visibility file
    :type ags3_vis_file: str
    :param target_name: Target name used to create the online enhanced visibility windows
    :type target_name: str
    :param target_id: Target id which will be appended to the online enhanced visibility windows
    :type target_id: int
    :return df_nicer_vis: NICER visibility
    :rtype: pandas.DataFrame
    :return df_nicer_vis_nosrcdulpicate: NICER visibility excluding all duplicate sources
    :rtype: pandas.DataFrame

    """
    keys = [',Target,', 'Target,']
    with open(ags3_vis_file, 'rt') as fp:
        lines = fp.readlines()
        for index, row in enumerate(lines):
            if any(k in row for k in keys):
                break

    # Skip these rows and read the rest of the table as a pandas dataframe
    df_nicer_vis = pd.read_csv(ags3_vis_file, sep=",", header=None, skiprows=index + 1,
                               names=["index_row", "target_name", "target_id", "vis_start", "vis_end", "Span",
                                      "Initial", "Final", "Relative", "radom_1", "radom_2", "radom_3"])
    df_nicer_vis = df_nicer_vis.drop(["index_row", "radom_1", "radom_2", "radom_3"],
                                     axis=1)  # Dropping unnecessary columns

    df_nicer_vis['vis_start'] = pd.to_datetime(df_nicer_vis['vis_start'], format='%Y-%m-%d %H:%M:%S', utc=True)
    df_nicer_vis['vis_start'] = df_nicer_vis['vis_start'].dt.strftime('%Y-%jT%H:%M:%S')
    df_nicer_vis['vis_end'] = pd.to_datetime(df_nicer_vis['vis_end'], format='%Y-%m-%d %H:%M:%S', utc=True)
    df_nicer_vis['vis_end'] = df_nicer_vis['vis_end'].dt.strftime('%Y-%jT%H:%M:%S')

    # Remove leading/trailing whitespaces from target name, just in case
    df_nicer_vis['target_name'] = df_nicer_vis['target_name'].str.strip()
    df_nicer_vis['target_name'] = target_name

    df_nicer_vis['target_id'] = target_id

    # Drop duplicates of exact target_name and start or end of visibility windows, keep first
    # warning: these targets have different target_IDs, only first ID kept
    mask = (df_nicer_vis.duplicated(subset=['target_name', 'vis_start']) |
            df_nicer_vis.duplicated(subset=['target_name', 'vis_end']))
    df_nicer_vis_nosrcdulpicate = df_nicer_vis[~mask]

    return df_nicer_vis, df_nicer_vis_nosrcdulpicate


def readmerge_agsfiles_enhancedvisibilitytool(targetsvisdetails_df):
    """
    Read and merge target visibility files created with the online enhanced visibility tools
    :param targetsvisdetails_df: dataframe of (RA, DEC, target_name, filepaths)
    :type targetsvisdetails_df: pandas.DataFrame
    :return df_nicer_vis_all: coordinates, target names, and filepaths dataframe (RA, DEC, target_name, filepaths)
    :rtype: pandas.DataFrame
    """
    df_nicer_vis_all_list = []
    df_nicer_vis_nosrcdulpicate_list = []
    for index, row in targetsvisdetails_df.iterrows():
        # Skip visibilites where download failed
        if targetsvisdetails_df.loc[index, 'filepaths'] is None:
            continue
        else:
            df_nicer_vis, df_nicer_vis_nosrcdulpicate = read_agsfile_enhancedvisibilitytool(
                targetsvisdetails_df.loc[index, 'filepaths'], targetsvisdetails_df.loc[index, 'Source'],
                targetsvisdetails_df.loc[index, 'ID'])
            df_nicer_vis_all_list.append(df_nicer_vis)
            df_nicer_vis_nosrcdulpicate_list.append(df_nicer_vis_nosrcdulpicate)

    df_nicer_vis_all = pd.concat(df_nicer_vis_all_list)
    df_nicer_vis_nosrcdulpicate = pd.concat(df_nicer_vis_nosrcdulpicate_list)

    return df_nicer_vis_all, df_nicer_vis_nosrcdulpicate


def writedfagsvisibtocvs(df_nicer_vis, nameofcvsfile):
    """
    Read-in AGS NICER visibility file created from the enhanced visibility tool as a dataframe
    :param df_nicer_vis: NICER visibility dataframe
    :type df_nicer_vis: pandas.DataFrame
    :param nameofcvsfile: Name of output csv NICER visibility file
    :type nameofcvsfile: str
    """
    buffer = StringIO()
    df_nicer_vis.to_csv(buffer, index=False, sep='\t', header=False)
    buffer.seek(0)
    csv_content = buffer.read()

    # Prepend custom header lines and write the final file
    custom_header = ("Target\tTarget\tVisibility Start\tVisibility End\tSpan\tInitial\tFinal\tRelative\n"
                     "Name\tID\t(YYYY-DDDTHH:MM:SS)\t(YYYY-DDDTHH:MM:SS)\t(seconds)\tInterference\tInterference\tOrbits\n"
                     "----\t--\t-------------------\t-------------------\t---------\t------------\t------------\t------\n")

    gz_filename = nameofcvsfile + '.csv.gz'
    with gzip.open(gz_filename, 'wt') as f:
        f.write(custom_header)
        f.write(csv_content)

    return None


def wait_for_download_to_finish(download_dir, timeout=30):
    """
    Blocks until:
      - at least one .csv is present in download_dir
      - no .crdownload files remain
    Returns the newest .csv by creation time.
    Raises TimeoutError if download_dir never settles within `timeout` seconds.
    """
    start_time = time.time()
    end_time = start_time + timeout

    while time.time() < end_time:
        good_files = []
        # 1) scan the folder
        for name in os.listdir(download_dir):
            if name.startswith('.'):  # skip hidden
                continue
            full = os.path.join(download_dir, name)
            try:
                # only consider files touched after we started waiting
                if os.path.getmtime(full) < start_time:
                    continue
            except FileNotFoundError:
                # race: file vanished between listdir and stat
                continue
            good_files.append(full)

        # 2) split into in‑flight vs completed
        crdownloads = [f for f in good_files if f.endswith('.crdownload')]
        completed = [f for f in good_files if f.endswith('.csv')]

        # 3) if we have at least one .csv and ZERO .crdownload, we're done
        if completed and not crdownloads:
            # return the newest (by creation time)
            return max(completed, key=os.path.getctime)

        # 4) otherwise, wait a bit and retry
        time.sleep(0.5)

    # timed out
    raise TimeoutError(f"No completed download after {timeout}s in {download_dir!r}")


def main():
    parser = argparse.ArgumentParser(description="Create AGS3 visibility file from online enhanced visibility tool")
    parser.add_argument("usertargetfile", help="Text file with source name, ra, dec, (no hearder)",
                        type=str)
    parser.add_argument("download_dir", help="Directory to which files are saved", type=str)
    parser.add_argument("-of", "--outputFile", help="Name of output AGS3 visibility file"
                                                    "(default = AGS3_usertargets(.csv))", type=str,
                        default='AGS3_usertargets')
    args = parser.parse_args()

    # Download viibilities from N tragets specified in text file 'usertargetfile'
    targetsvisfiles_df = download_ntarget_visibilities(args.usertargetfile, args.download_dir)

    # Read visibilities and merge them into a single dataframe
    _, df_nicer_vis_nosrcdulpicate = readmerge_agsfiles_enhancedvisibilitytool(targetsvisfiles_df)

    # Write merged visibilities into a text file 'outputFile'.csv
    writedfagsvisibtocvs(df_nicer_vis_nosrcdulpicate, args.outputFile)

    return None


if __name__ == '__main__':
    main()
