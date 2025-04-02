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
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from optvisapp.optvisapp_logging import get_logger

sys.dont_write_bytecode = True

# Log config
############
logger = get_logger(__name__)


def download_onetarget_visibility(ra, dec, download_dir):
    """
    Download to 'download_dir' the NICER enhanced visibilities given RA and DEC
    :param ra: right ascension in degrees J2000
    :type ra: str
    :param dec: declination in degrees J2000
    :type dec: str
    :param download_dir: Download directory
    :type download_dir: str
    :return new_path: full path to downloaded file
    :rtype: str
    """
    # Ensure download path exists
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    # Set Chrome options
    chrome_options = Options()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless")  # Optional headless mode

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    wait = WebDriverWait(driver, 20)

    # Step 1: Go to NICER visibility page
    driver.get("https://heasarc.gsfc.nasa.gov/wsgi-scripts/nicer/visibility/nicervis.wsgi/")
    wait.until(EC.presence_of_element_located((By.NAME, "ra")))

    # Step 2: Fill form
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

    submit_button = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//input[@type='submit' and @value='Submit']")))
    submit_button.click()

    # Step 3: Wait for results or error
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    if "Internal Server Error" in driver.page_source:
        logger.info(f"Server error for RA={ra}, Dec={dec}")
        return None

    # Step 4: Download visibility file
    download_button = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//input[@type='submit' and @value='Download Visibilities']")))
    download_button.click()
    # Wait for the file to appear or start downloading
    wait.until(lambda d: any(f.endswith('.crdownload') or f.endswith('.csv') for f in os.listdir(download_dir)))

    downloaded_file = wait_for_download_to_finish(download_dir)

    if downloaded_file:
        new_path = downloaded_file
        logger.info("Download complete.")
        driver.quit()
        return new_path
    else:
        logger.error("Download timed out.")
        raise Exception(f"Download timed out.")


def download_ntarget_visibilities(listofcoordinates, download_dir):
    """
    Download to 'download_dir' the NICER enhanced visibilities given a .txt file of RA, DEC, target_name
    :param listofcoordinates: a text file (comma-separated) with coordinates and target names (Source, RA, DEC)
    :type listofcoordinates: str
    :param download_dir: Directory to which downloaded visibilities are saved
    :type download_dir: str
    :return targetvisdetail_df: dataframe of (Source, 'RAJ_DEG', 'DECJ_DEG', filepaths)
    :rtype: pandas.DataFrame
    """
    # Reading list of coordinates
    targetsvisfiles_df = pd.read_csv(listofcoordinates, header=None, names=['Source', 'RAJ_DEG', 'DECJ_DEG'])
    listoffilepaths = []
    target_id = []
    for index, row in targetsvisfiles_df.iterrows():
        filpath = download_onetarget_visibility(row['RAJ_DEG'], row['DECJ_DEG'], download_dir)
        listoffilepaths.append(filpath)
        target_id.append(int('99999'+str(index)))

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


def wait_for_download_to_finish(path, timeout=30):
    start_time = time.time()
    end_time = start_time + timeout
    while time.time() < end_time:
        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = [f for f in files if os.path.getmtime(f) >= start_time]

        # Exclude hidden/system files
        files = [f for f in files if not os.path.basename(f).startswith(".")]

        crdownloads = [f for f in files if f.endswith('.crdownload')]
        completed = [f for f in files if not f.endswith('.crdownload')]

        if not crdownloads and completed:
            return max(completed, key=os.path.getctime)
        time.sleep(1)
    return None


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
