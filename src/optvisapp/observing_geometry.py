"""
Module to calculate certain geometric elements related to the ISS and observing sources
Meant  for NICER
"""

import numpy as np
import pandas as pd

import astropy
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import CartesianRepresentation, get_sun, SkyCoord, AltAz, EarthLocation, GCRS
from astropy.time import Time

from joblib import Parallel, delayed


def is_in_orbit_day(sat_vec, sun_vec):
    """
    Calculating umbra, penumbra geometries, and satellite status (in earth shadow, in Sunlight)
    Accurate to within ~1 minute for LEO satellites around Umbra/Penumbra terminator lines
    From https://ntrs.nasa.gov/api/citations/19950023025/downloads/19950023025.pdf

    :param sat_vec: Satellite cartesian coordonates (km) in Geocentric coordinates system (J2000)
    :type sat_vec: numpy.ndarray
    :param sun_vec: Sun vector cartesian coordonates (km) in Geocentric coordinates system (J2000)
    :type sun_vec: numpy.ndarray
    :return in_orbit_day: Boolean of satellite in orbit_day (True) or in earth shadow (False)
    :rtype: bool
    """
    # Some constants
    R_earth = 6378.137
    R_sun = 696340.0

    # Umbra calculation
    ###################
    # Distance from geocenter to Sun
    d_sun = np.linalg.norm(sun_vec)

    # Distance from umbra to earth, and umbra angle
    d_umb_earth = (2 * R_earth * d_sun) / (2 * (R_sun - R_earth))
    theta_umb = np.arcsin((2 * R_earth) / (2 * d_umb_earth))

    # Penumbra calculation
    ######################
    d_penumb_earth = (2 * R_earth * d_sun) / (2 * (R_sun + R_earth))
    theta_penumb = np.arcsin((2 * R_earth) / (2 * d_penumb_earth))

    # Calculation of projection of earth-satellite vector on earth-Sun vector
    #########################################################################
    sun_unit_vec = sun_vec / d_sun
    R_S_vector = np.dot(sat_vec, sun_unit_vec) * sun_unit_vec
    delta_vec = sat_vec - R_S_vector

    # Shadow terminators can only be encountered if np.dot(sat_vec, sun_unit_vec) < 0; otherwise source is in orbit_day
    ###################################################################################################################
    if np.dot(sat_vec, sun_unit_vec) > 0:
        in_orbit_day = True
        return in_orbit_day

    # xi distance
    #############
    xi_dist = (d_umb_earth - np.linalg.norm(R_S_vector)) * np.tan(theta_umb)

    # kappa distance
    ################
    kappa_dist = (d_penumb_earth + np.linalg.norm(R_S_vector)) * np.tan(theta_penumb)

    # Conditions for orbit day and night
    ####################################
    if np.linalg.norm(delta_vec) > kappa_dist:
        # Away from penumbra is considered orbit-day
        in_orbit_day = True
    elif (xi_dist < np.linalg.norm(delta_vec)) and (np.linalg.norm(delta_vec) <= kappa_dist):
        # Penumbra is considered orbit-night
        in_orbit_day = False
    elif np.linalg.norm(delta_vec) <= xi_dist:
        # Umbra is of course orbit-night
        in_orbit_day = False

    return in_orbit_day


def iss_islit(times, issorbitdata):
    """
    Calculates whether ISS is in orbit day or night depending on certain times
    :param times: Pandas series of times in isot format, utc (dtype: datetime64[ns, UTC])
    :type times: pandas.DatetimeIndex
    :param issorbitdata: iss times and coordinate data from OEM ephem file
    :type issorbitdata: pandas.DataFrame
    :return iss_islit: Pandas series of orbit day (1) or night (0)
    :rtype: pandas.DataFrame
    """
    islit = np.empty(len(times), dtype=object)
    for index, time_stamp in times.items():
        # Closest ISS coordinate information to time_stamp
        closest_row = get_closest_earliest_row(issorbitdata, 'TIME_UTC', time_stamp)

        # Get ISS X, Y, Z at the exact timestamp
        t_diff = time_stamp - closest_row['TIME_UTC']
        iss_cartesian_extrapolated = np.array(
            [closest_row['ISS_X'] + closest_row['ISS_Vx'] * t_diff.total_seconds(),
             closest_row['ISS_Y'] + closest_row['ISS_Vy'] * t_diff.total_seconds(),
             closest_row['ISS_Z'] + closest_row['ISS_Vz'] * t_diff.total_seconds()])

        # Compute iss coordinates in GCRS reference frame
        sat_gcrs = GCRS(x=iss_cartesian_extrapolated[0] * u.km, y=iss_cartesian_extrapolated[1] * u.km,
                        z=iss_cartesian_extrapolated[2] * u.km, representation_type='cartesian')
        sat_vec = sat_gcrs.cartesian.xyz.to(u.km).value

        # Compute sun position
        t_iss = Time(time_stamp.strftime('%Y-%m-%dT%H:%M:%S'), format='isot', scale='utc')
        sun_vec = get_sun(t_iss).transform_to(GCRS(obstime=t_iss)).cartesian.xyz.to(u.km).value

        # Check if satellite is in Earth's shadow
        in_orbit_day = is_in_orbit_day(sat_vec, sun_vec)

        # Save day (1) or night (0) status
        islit[index] = (1 if in_orbit_day else 0)

    return islit


def viswindow_islit(vis_windows, issorbitdata, frequency=60):
    """
    Calculates whether iss is in orbit day or night according to visibility windows
    :param vis_windows: dataframe with start and end times in isot format, utc (vis_start, vis_end), indices must be integer sequential
    :type vis_windows: pandas.DataFrame
    :param issorbitdata: iss times and coordinate data from OEM ephem file
    :type issorbitdata: pandas.DataFrame
    :param frequency: frequency in seconds at which to calculate if iss in orbit day or night
    :type frequency: int
    :return vis_windows: same as input dataframe but with an extra column 'orbit' ('o_d' for day, 'o_n' for night, 'partial' for both)
    :rtype: pandas.DataFrame
    """
    islit = np.empty(len(vis_windows), dtype=object)

    vis_windows_duplicates = vis_windows.duplicated(subset=['vis_start', 'vis_end'], keep='first')
    duplicate_indices = vis_windows_duplicates.index[vis_windows_duplicates == True]

    for ii in vis_windows.index:
        # In case this row has already been seen
        if ii in duplicate_indices:
            # Cut dataframe above this row
            vis_windows_tmp = vis_windows.loc[:ii - 1]
            # Get the first occurrence
            vis_start_tmp = vis_windows['vis_start'].loc[ii]
            vis_end_tmp = vis_windows['vis_end'].loc[ii]
            index_first_occurrence = vis_windows_tmp[(vis_windows_tmp['vis_start'] == vis_start_tmp) &
                                                     (vis_windows_tmp['vis_end'] == vis_end_tmp)].index[0]

            islit[ii - vis_windows.index.min()] = islit[index_first_occurrence]
        # These are rows with start and end visibility windows that are new
        else:
            vis_range = pd.date_range(start=vis_windows['vis_start'].loc[ii], end=vis_windows['vis_end'].loc[ii],
                                      freq=str(frequency) + 's').to_frame(name='vis_times', index=False)['vis_times']

            is_orbitday = iss_islit(vis_range, issorbitdata)

            if np.all(is_orbitday):
                islit[ii - vis_windows.index.min()] = 'o_d'
            elif np.all(is_orbitday == False):
                islit[ii - vis_windows.index.min()] = 'o_n'
            else:
                islit[ii - vis_windows.index.min()] = 'partial'

    vis_windows['orbit'] = islit

    return vis_windows


def iss_horizon_line(S_iss):
    """
    # See https://physics.stackexchange.com/questions/151388/how-to-calculate-the-horizon-line-of-a-satellite
    Calculate the ISS horizon line.
    Uses vectorized numpy operations.
    """
    R_earth = 6378.137  # km
    S_iss_norm = np.linalg.norm(S_iss)
    S_iss_unitvec = S_iss / S_iss_norm

    R_S_fraction = R_earth / S_iss_norm
    t_ring = np.linspace(0, 2 * np.pi, 180, endpoint=False)

    # Compute the right-hand side matrix in one go:
    mat_x = -S_iss_unitvec[1] * np.cos(t_ring) - (S_iss_unitvec[0] * S_iss_unitvec[2]) * np.sin(t_ring)
    mat_y = S_iss_unitvec[0] * np.cos(t_ring) - (S_iss_unitvec[1] * S_iss_unitvec[2]) * np.sin(t_ring)
    mat_z = (S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2) * np.sin(t_ring)
    mat_rightside = np.column_stack((mat_x, mat_y, mat_z))

    mat_multiplier = np.sqrt((1 - R_S_fraction ** 2) / (S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2))
    full_rightside = mat_multiplier * mat_rightside
    full_leftside = R_S_fraction * S_iss_unitvec

    h_ring = R_earth * (full_leftside + full_rightside)
    return h_ring  # shape (180, 3)


def compute_hring_lit(x, y, z, time, sun_coord):
    # Create a scalar EarthLocation for this horizon point
    location = EarthLocation.from_geocentric(x, y, z, unit=u.km)
    # Transform the sun coordinate into the AltAz frame for this location
    altaz = sun_coord.transform_to(AltAz(obstime=time, location=location))
    # Return True if the altitude is > 0, meaning it's sunlit
    return altaz.alt.deg > 0


def bright_earth_angle(iss_cartesian, time, src_ra, src_dec):
    """
    Optimized calculation of bright-earth angle based on source and observatory coordinates (for NICER but should work for others)
    Based on Astropy
    :param iss_cartesian: ISS position in Cartesian coordinates (in km)
    :type iss_cartesian: numpy.ndarray
    :param time: astropy TIME object in 'isot' format and 'utc' scale
    :type time: astropy.time.Time
    :param src_ra: Source RA in J2000, degree
    :type src_ra: float
    :param src_dec: Source DEC in J2000, degree
    :type src_dec: float
    :return brightearth_angle: bright_earth angle in degrees
    :rtype: pandas.DataFrame
    """
    # 1. Compute the horizon ring (vectorized)
    h_ring = iss_horizon_line(iss_cartesian)  # (180,3)

    # 2. Compute source Cartesian coordinates (using Astropy; this is a one-time cost)
    src_coord = SkyCoord(ra=src_ra * u.degree, dec=src_dec * u.degree,
                         distance=2 * u.kpc, frame='icrs')
    src_cartesian = src_coord.cartesian.xyz.to(u.km).value  # shape (3,)

    # 3. Vectorize the angle computation:
    vec_IS = src_cartesian - iss_cartesian  # (3,)
    norm_vec_IS = np.linalg.norm(vec_IS)

    # Compute vectors from ISS to each horizon point:
    vec_IH = h_ring - iss_cartesian  # (180, 3)
    # Dot products for each horizon point:
    dot_products = np.sum(vec_IH * vec_IS, axis=1)
    # Norms for each horizon vector:
    norms_vec_IH = np.linalg.norm(vec_IH, axis=1)
    # Calculate angles (in degrees) in one vectorized step:
    angles = np.degrees(np.arccos(dot_products / (norm_vec_IS * norms_vec_IH)))

    # 4. Determine whether each horizon point is sunlit:
    # Create a vectorized CartesianRepresentation for all horizon points.
    cart_repr = CartesianRepresentation(h_ring[:, 0] * u.km,
                                        h_ring[:, 1] * u.km,
                                        h_ring[:, 2] * u.km)
    # Transform all points at once from GCRS to ITRS:
    gcrs_coords = GCRS(cart_repr, obstime=time)
    itrs_coords = gcrs_coords.transform_to(coord.ITRS(obstime=time))

    # EarthLocation.from_geocentric supports vectorized inputs.
    earth_locs = EarthLocation.from_geocentric(
        itrs_coords.cartesian.x,
        itrs_coords.cartesian.y,
        itrs_coords.cartesian.z
    )

    # Although get_sun(time) is a single coordinate, AltAz transformation
    # may not directly support a vectorized EarthLocation. For 180 points this
    # loop is acceptable:
    sun_coord = get_sun(time)
    # Parallelize the loop over horizon points:
    hring_lit = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_hring_lit)(x, y, z, time, sun_coord)
            for x, y, z in zip(itrs_coords.cartesian.x,
                               itrs_coords.cartesian.y,
                               itrs_coords.cartesian.z)
        )
    )

    # 5. Put the results into a DataFrame and select only lit horizon points:
    df = pd.DataFrame({
        'source_to_iss_horizon_angle': angles,
        'hring_lit': hring_lit
    })
    df_lit = df[df['hring_lit']]
    # If no points are lit, you might want to handle that case:
    if df_lit.empty:
        return np.nan
    bright_earth_angle_val = df_lit['source_to_iss_horizon_angle'].min()
    return bright_earth_angle_val


def bright_earth_angle_notparallel(iss_cartesian, time, src_ra, src_dec):
    """
    Optimized calculation of bright-earth angle based on source and observatory coordinates (for NICER but should work for others)
    Based on Astropy
    :param iss_cartesian: ISS position in Cartesian coordinates (in km)
    :type iss_cartesian: numpy.ndarray
    :param time: astropy TIME object in 'isot' format and 'utc' scale
    :type time: astropy.time.Time
    :param src_ra: Source RA in J2000, degree
    :type src_ra: float
    :param src_dec: Source DEC in J2000, degree
    :type src_dec: float
    :return brightearth_angle: bright_earth angle in degrees
    :rtype: pandas.DataFrame
    """
    # 1. Compute the horizon ring (vectorized)
    h_ring = iss_horizon_line(iss_cartesian)  # (180,3)

    # 2. Compute source Cartesian coordinates (using Astropy; this is a one-time cost)
    src_coord = SkyCoord(ra=src_ra * u.degree, dec=src_dec * u.degree,
                         distance=2 * u.kpc, frame='icrs')
    src_cartesian = src_coord.cartesian.xyz.to(u.km).value  # shape (3,)

    # 3. Vectorize the angle computation:
    vec_IS = src_cartesian - iss_cartesian  # (3,)
    norm_vec_IS = np.linalg.norm(vec_IS)

    # Compute vectors from ISS to each horizon point:
    vec_IH = h_ring - iss_cartesian  # (180, 3)
    # Dot products for each horizon point:
    dot_products = np.sum(vec_IH * vec_IS, axis=1)
    # Norms for each horizon vector:
    norms_vec_IH = np.linalg.norm(vec_IH, axis=1)
    # Calculate angles (in degrees) in one vectorized step:
    angles = np.degrees(np.arccos(dot_products / (norm_vec_IS * norms_vec_IH)))

    # 4. Determine whether each horizon point is sunlit:
    # Create a vectorized CartesianRepresentation for all horizon points.
    cart_repr = CartesianRepresentation(h_ring[:, 0] * u.km,
                                        h_ring[:, 1] * u.km,
                                        h_ring[:, 2] * u.km)
    # Transform all points at once from GCRS to ITRS:
    gcrs_coords = GCRS(cart_repr, obstime=time)
    itrs_coords = gcrs_coords.transform_to(coord.ITRS(obstime=time))

    # EarthLocation.from_geocentric supports vectorized inputs.
    earth_locs = EarthLocation.from_geocentric(
        itrs_coords.cartesian.x,
        itrs_coords.cartesian.y,
        itrs_coords.cartesian.z
    )

    # Although get_sun(time) is a single coordinate, AltAz transformation
    # may not directly support a vectorized EarthLocation. For 180 points this
    # loop is acceptable:
    sun_coord = get_sun(time)
    hring_lit = np.array([
        sun_coord.transform_to(AltAz(obstime=time,
                                     location=EarthLocation.from_geocentric(x, y, z, unit=u.km))
                               ).alt.deg > 0
        for x, y, z in zip(itrs_coords.cartesian.x,
                           itrs_coords.cartesian.y,
                           itrs_coords.cartesian.z)
    ])

    # 5. Put the results into a DataFrame and select only lit horizon points:
    df = pd.DataFrame({
        'source_to_iss_horizon_angle': angles,
        'hring_lit': hring_lit
    })
    df_lit = df[df['hring_lit']]
    # If no points are lit, you might want to handle that case:
    if df_lit.empty:
        return np.nan
    bright_earth_angle_val = df_lit['source_to_iss_horizon_angle'].min()
    return bright_earth_angle_val


def bright_earth_angle_notptimized(iss_cartesian, time, src_ra, src_dec):
    """
    Calculate bright-earth angle based on source and observatory coordinates (for NICER but should work for others)
    Based on Astropy
    :param iss_cartesian: ISS position in Cartesian coordinates (in km)
    :type iss_cartesian: numpy.ndarray
    :param time: astropy TIME object in 'isot' format and 'utc' scale
    :type time: astropy.time.Time
    :param src_ra: Source RA in J2000, degree
    :type src_ra: float
    :param src_dec: Source DEC in J2000, degree
    :type src_dec: float
    :return brightearth_angle: bright_earth angle in degrees
    :rtype: pandas.DataFrame
    """
    # Derive the coordinates of the ISS horizon line
    h_ring = iss_horizon_line(iss_cartesian)
    # Source cartesian coordinates - use random distance, will not matter for our calculation
    src_cartesian = np.array(SkyCoord(ra=src_ra * u.degree, dec=src_dec * u.degree, distance=2 * u.kpc,
                                      frame='icrs').cartesian.xyz.to(u.km).value)
    # Calculate angle between ISS (I) and source (S) and ISS (I) and edge of horizon line (H)
    vec_IS = np.array([src_cartesian[0] - iss_cartesian[0], src_cartesian[1] - iss_cartesian[1],
                       src_cartesian[2] - iss_cartesian[2]])
    source_to_iss_horizon_angle = np.empty(len(h_ring), dtype=float)

    hring_lit = np.empty(len(h_ring), dtype=bool)
    for nn in range(len(h_ring)):
        # First let's calculate Source to ISS horizon angles
        vec_IH = np.array(
            [h_ring[nn, 0] - iss_cartesian[0], h_ring[nn, 1] - iss_cartesian[1], h_ring[nn, 2] - iss_cartesian[2]])

        # Calculate cosine of the angle
        source_to_iss_horizon_angle[nn] = np.degrees(
            np.arccos(np.dot(vec_IS, vec_IH) / (np.linalg.norm(vec_IS) * np.linalg.norm(vec_IH))))

        # Here we calculate whether Sun is above horizon at h_ring coordinates
        xyz = (h_ring[nn, 0], h_ring[nn, 1], h_ring[nn, 2])  # Xyz coord for each prop. step
        gcrs = coord.GCRS(CartesianRepresentation(*xyz, unit=u.km), obstime=time)  # Let AstroPy know xyz is in GCRS
        itrs = gcrs.transform_to(coord.ITRS(obstime=time))  # Convert GCRS to ITRS
        earth_location = EarthLocation(*itrs.cartesian.xyz)
        # Sun location in Alt-Az
        sun_coord = get_sun(time)
        sun_altaz = sun_coord.transform_to(AltAz(obstime=time, location=earth_location))
        # We are not considering refraction, but if we wish to (pressure=101325*u.Pa, temperature=15*u.deg_C,
        # relative_humidity=0.6)
        # Is location Sun-lit
        hring_lit[nn] = (sun_altaz.alt.deg > 0)

    source_to_iss_horizon = pd.DataFrame({'source_to_iss_horizon_angle': source_to_iss_horizon_angle,
                                          'hring_lit': hring_lit})
    source_to_iss_horizon_od = source_to_iss_horizon[source_to_iss_horizon['hring_lit']]
    brightearth_angle = source_to_iss_horizon_od['source_to_iss_horizon_angle'].min()

    return brightearth_angle


def iss_horizon_line_notoptimized(S_iss):
    # See https://physics.stackexchange.com/questions/151388/how-to-calculate-the-horizon-line-of-a-satellite
    R_earth = 6378.137  # (km)
    S_iss_norm = np.linalg.norm(S_iss)
    S_iss_unitvec = S_iss / S_iss_norm

    # Earth to ISS altitude fraction
    R_S_fraction = R_earth / S_iss_norm
    t_ring = np.linspace(0, 2 * np.pi, 180, endpoint=False)

    # Matrix right-side of equation
    mat_x = (-S_iss_unitvec[1] * np.cos(t_ring) - (S_iss_unitvec[0] * S_iss_unitvec[2]) * np.sin(t_ring))
    mat_y = (S_iss_unitvec[0] * np.cos(t_ring) - (S_iss_unitvec[1] * S_iss_unitvec[2]) * np.sin(t_ring))
    mat_z = ((S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2) * np.sin(t_ring))
    mat_rightside = np.array([mat_x, mat_y, mat_z]).T
    # Matrix multiplier
    mat_multiplier = np.sqrt((1 - R_S_fraction ** 2) / (S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2))
    # Full right side of equation
    full_rightside = mat_multiplier * mat_rightside

    # Left side of equation
    full_leftside = R_S_fraction * S_iss_unitvec

    # Horizon_line
    h_ring = R_earth * (full_leftside + full_rightside)

    return h_ring


def sunangle(nicertimemjd, srcRA, srcDEC):
    """
    Calculates Sun angle for a source with RA and DEC in degrees J2000
    :param nicertimemjd: numpy array of (nicer) times in MJD
    :type nicertimemjd: numpy.ndarray
    :param srcRA: Right ascension in degrees J2000
    :type srcRA: float
    :param srcDEC: Declination in degrees J2000
    :type srcDEC: float
    :return srcsunangle: array of source sun angles at times nicertimemjd in degrees
    :rtype: numpy.ndarray
    """
    # Define astropy Time instance in mjd format
    nicerTIME = Time(nicertimemjd, format='mjd')

    # Get Sun coordinates
    sunAngGeo = get_sun(nicerTIME)
    sunAngTETE = sunAngGeo.tete

    srcsunangle = np.zeros(len(nicertimemjd))
    for jj in range(len(nicertimemjd)):
        RA_sun = sunAngTETE[jj].ra.deg
        DEC_sun = sunAngTETE[jj].dec.deg
        srcsunangle[jj] = np.rad2deg(np.arccos(np.sin(np.deg2rad(DEC_sun)) *
                                               np.sin(np.deg2rad(srcDEC)) +
                                               np.cos(np.deg2rad(DEC_sun)) *
                                               np.cos(np.deg2rad(srcDEC)) *
                                               np.cos(np.deg2rad(RA_sun) -
                                                      np.deg2rad(srcRA))))

    return srcsunangle


def get_closest_earliest_row(df, timestamp_col, target_time):
    """
    Finds the row with the closest but earliest timestamp to a given target time.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the timestamps.
    - timestamp_col (str): The name of the timestamp column.
    - target_time (pd.Timestamp): The target time to compare against.

    Returns:
    - pd.Series or None: The row with the closest but earliest timestamp, or None if no valid timestamp is found.
    """
    # Filter for timestamps that are earlier than or equal to the target time
    filtered_df = df[df[timestamp_col] <= target_time]

    # Get the row with the closest (latest among earlier) timestamp
    if not filtered_df.empty:
        closest_row = filtered_df.loc[filtered_df[timestamp_col].idxmax()]
        return closest_row
    else:
        return None  # No valid timestamps found


def extrapolate_iss_position(iss_df, timestamp):
    """
    Extrapolates the ISS position based on a given orbit time and a reference row containing ISS position and velocity.

    Parameters:
    - orbit_time (datetime): The target time for extrapolation.
    - closest_row (pd.Series): A Pandas Series containing:
        - 'TIME_UTC' (datetime): Reference time
        - 'ISS_X', 'ISS_Y', 'ISS_Z' (float): ISS position at reference time (in km)
        - 'ISS_Vx', 'ISS_Vy', 'ISS_Vz' (float): ISS velocity components (in km/s)

    Returns:
    - np.ndarray: Extrapolated ISS position [X, Y, Z] in km.
    """
    #
    closest_row = get_closest_earliest_row(iss_df, 'TIME_UTC', timestamp)

    # Compute time difference in seconds
    t_diff = (timestamp - closest_row['TIME_UTC']).total_seconds()

    # Extrapolate position using velocity
    iss_cartesian_extrapolated = np.array([
        closest_row['ISS_X'] + closest_row['ISS_Vx'] * t_diff,
        closest_row['ISS_Y'] + closest_row['ISS_Vy'] * t_diff,
        closest_row['ISS_Z'] + closest_row['ISS_Vz'] * t_diff
    ])

    return iss_cartesian_extrapolated
