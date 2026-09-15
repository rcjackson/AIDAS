import numpy as np
import paramiko
import datetime
import os
import xarray as xr
import logging

from ..util import azimuth_point

def make_scan_file(elevations, azimuths,
                   out_file_name, azi_speed=1.,
                   el_speed=0.1,
                   wait=0, acceleration=30, repeat=7,
                   rays_per_point=20, dyn_csm=False,
                   AZ_COUNTS_PER_ROT=500000, EL_COUNTS_PER_ROT=250000):
    """
    Makes a CSM scanning strategy file for a Halo Photonics Doppler Lidar.
    
    Parameters
    ----------
    no_points: int
        The number of points to collect in the ray
    elevations: float 1d array or tuple
        The elevation of each sweep in the scan. If this is a 2-tuple, then
        the script will generate an RHI spanning the smallest to largest elevation
    azimuths: float or 2-tuple
        If this is a 2-tuple, then this script will generate a PPI from min_azi to max_azi.
    out_file_name: str
        The output name of the file.
    azi_speed: float
        The speed of the azimuth motor in degrees per second.
    el_speed: float
        The speed of the elevation motor in degrees per second.
    wait: int
        The wait time in milliseconds at each point in the scan.
    acceleration: int
        The acceleration of the motor in ticks per second squared. This is a constant that depends on the lidar hardware.
    repeat: int
        The number of times to repeat the scan strategy. This is used to ensure that the lidar collects enough data points for each scan.
    rays_per_point: int
        The number of rays to collect at each point in the scan. This is used to ensure
        that the lidar collects enough data points for each scan.
    dyn_csm: bool
        Set to True to send CSM assuming Dynamic CSM mode
    AZ_COUNTS_PER_ROT: int
        The number of counts per rotation for the azimuth motor. This is a constant that depends
        on the lidar hardware and is used to convert from degrees to the encoded values that the lidar uses for its scan strategy.
    EL_COUNTS_PER_ROT: int
        The number of counts per rotation for the elevation motor. This is a constant that depends
        on the lidar hardware and is used to convert from degrees to the encoded values that the lidar uses
        for its scan strategy.
    
    Returns
    -------
    None
        This function does not return anything. It generates a CSM scan strategy file for the Halo Lidar 
        and saves it to the specified output file name.
    """    
    speed_azi_encoded = int(azi_speed * (AZ_COUNTS_PER_ROT / 360.))
    speed_el_encoded = int(el_speed * (EL_COUNTS_PER_ROT / 360.))
    clockwise = True
    no_points = len(azimuths) * len(elevations)
    with open(out_file_name, 'w') as output:
        if dyn_csm is False:
            output.write('%d\r\n' % repeat)
            output.write('%d\r\n' % no_points)
            output.write('%d\r\n' % rays_per_point)
  
        for el in elevations:
            if clockwise:
                az_array = azimuths
            else:
                az_array = azimuths.reverse()
            for az in az_array:
                azi_encoded = -int(az * (AZ_COUNTS_PER_ROT / 360.))
                el_encoded = -int(el * (EL_COUNTS_PER_ROT / 360.))
                output.write("A.1=%d,S.1=%d,P.1=%d*A.2=%d,S.2=%d,P.2=%d\r\n" %
                             (acceleration, speed_azi_encoded, azi_encoded,
                              acceleration, speed_el_encoded, el_encoded))
                output.write('W%d\r\n' % (wait))
            clockwise = ~clockwise
    return


def send_scan(file_name, lidar_ip_addr, lidar_uname, lidar_pwd, out_file_name='user.txt', dyn_csm=False,
              client=None):
    """
    Sends a scan to the lidar

    Parameters
    ---------
    file_name: str
        Path to the CSM-format scan strategy
    lidar_ip_addr:
        IP address of the lidar
    lidar_uname:
        The username of the lidar
    lidar_password:
        The lidar's password
    out_file_name:
        The output file name on the lidar
    dyn_csm: bool
        Set to True to assume Dynamic CSM mode
    client: paramiko.SSHClient, optional
        An optional SSH client to use for the connection. If not provided, a new client will be created
        and closed within this function.

    """
    if client is not None:
        ssh = client
        close_client = False
    else:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(lidar_ip_addr, username=lidar_uname, password=lidar_pwd)
        print("Connected to the Lidar!")
        close_client = True
        
    if close_client:
        with ssh.open_sftp() as sftp:
            if dyn_csm is False:
                logging.info(f"Writing {out_file_name} on lidar.")
                sftp.put(file_name, "/C:/Lidar/System/Scan parameters/%s" % out_file_name)
            else:
                sftp.put(file_name, f"/C:/Users/End User/DynScan/{out_file_name}")
    else:
        sftp = ssh.open_sftp()
        if dyn_csm is False:
            print(f"Writing {out_file_name} on lidar.")
            sftp.put(file_name, "/C:/Lidar/System/Scan parameters/%s" % out_file_name)
        else:
            sftp.put(file_name, f"/C:/Users/End User/DynScan/{out_file_name}")

    if close_client:
        ssh.close()


def trigger_lidar_ppis_from_mask(rad_scan, lidar_lat, lidar_lon, lidar_ip_addr, lidar_uname, lidar_pwd, elevations, 
                                 az_width=30., out_file_name='user.txt', dyn_csm=False,
                                 max_distance=5000, client=None):
    """
    Triggers a PPI scan on the lidar using a scan strategy generated from a lake breeze mask.

    Parameters
    ----------
    rad_scan: RadarImage
        The radar scan containing the lake breeze mask. The azimuth of the largest lake breeze region will be
        used to determine the center of the RHI scan.
    lidar_lat: float
        The latitude of the lidar
    lidar_lon: float
        The longitude of the lidar
    lidar_ip_addr:
        IP address of the lidar
    lidar_uname:
        The username of the lidar
    lidar_password:
        The lidar's password
    out_file_name:
        The output file name on the lidar
    dyn_csm: bool
        Set to True to assume Dynamic CSM mode
    az_width: float
        The width of the azimuth scan in degrees. The scan will be centered around the azimuth of the
        largest lake breeze region as determined by the model's radar image and the location of the lidar.
    max_distance: float
        The maximum distance from the lidar to the lake breeze region for the scan to be triggered.
    client: paramiko.SSHClient, optional
        An optional SSH client to use for the connection. If not provided, a new client will be created
        and closed within the send_scan function.
    
    Returns
    -------
    bool
        Returns True if the scan was triggered, and False if the scan was not triggered due to the distance from the lidar
        to the lake breeze region being greater than max_distance.
    """
    middle_azimuth, lat, lon, dist = azimuth_point(lidar_lon, lidar_lat, rad_scan)
    if dist > max_distance:  # If the distance is greater than max_distance, don't trigger the scan
        logging.info(f"Distance from lidar to lake breeze region is {dist} meters. Not triggering scan.")
        return False
    azimuths = np.array([middle_azimuth - az_width/2, middle_azimuth + az_width/2])

    make_scan_file(elevations, azimuths, out_file_name, dyn_csm=dyn_csm)
    send_scan(out_file_name, lidar_ip_addr, lidar_uname, lidar_pwd, out_file_name=out_file_name, dyn_csm=dyn_csm,
              client=client)
    return True


def trigger_lidar_rhi_from_mask(rad_scan, lidar_lat, lidar_lon, lidar_ip_addr, lidar_uname, lidar_pwd, elevations, 
                                out_file_name='user.txt', dyn_csm=False, max_distance=5000, client=None):
    """
    Triggers a PPI scan on the lidar using a scan strategy generated from a lake breeze mask.

    Parameters
    ----------
    rad_scan: RadarImage
        The radar scan containing the lake breeze mask. The azimuth of the largest lake breeze region will be
        used to determine the center of the RHI scan.
    lidar_lat: float
        The latitude of the lidar
    lidar_lon: float
        The longitude of the lidar
    lidar_ip_addr:
        IP address of the lidar
    lidar_uname:
        The username of the lidar
    lidar_password:
        The lidar's password
    out_file_name:
        The output file name on the lidar
    dyn_csm: bool
        Set to True to assume Dynamic CSM mode
    az_width: float
        The width of the azimuth scan in degrees. The scan will be centered around the azimuth of the
        largest lake breeze region as determined by the model's radar image and the location of the lidar.
    az_res: float
        The resolution of the azimuth scan in degrees. This determines how many points will be in the scan.
        For example, if az_width is 30 and az_res is 2, then
        there will be 15 points in the azimuth scan (from -15 to +15 degrees around the center azimuth).
    max_distance: float
        The maximum distance from the lidar to the lake breeze region for the scan to be triggered.
    client: paramiko.SSHClient, optional
        An optional SSH client to use for the connection. If not provided, a new client will be created
        and closed within the send_scan function.
    
    Returns
    -------
    bool
        Returns True if the scan was triggered, and False if the scan was not triggered due to
        the distance from the lidar to the lake breeze region being greater than max_distance.
    """
    middle_azimuth, lat, lon, dist = azimuth_point(lidar_lon, lidar_lat, rad_scan)
    if dist > max_distance:  # If the distance is greater than max_distance, don't trigger the scan
        logging.info(f"Distance from lidar to lake breeze region is {dist} meters. Not triggering scan.")
        return False
    azimuths = [middle_azimuth]

    make_scan_file(elevations, azimuths, out_file_name, dyn_csm=dyn_csm)
    send_scan(out_file_name, lidar_ip_addr, lidar_uname, lidar_pwd, out_file_name=out_file_name, dyn_csm=dyn_csm,
              client=client)
    return True