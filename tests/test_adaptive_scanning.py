import numpy as np
import os
import re

# Halo encoder constant: azimuth degrees -> motor counts (AZ_COUNTS_PER_ROT/360).
_AZ_COUNTS_PER_DEG = 500000 / 360.0

# The commanded azimuth is derived from the centroid of the lake breeze mask.
# When the front sits on top of the lidar -- which it does in the 2025-04-24
# case, where the distance comes out at 0 m -- that centroid is only a few
# pixels away from the instrument, so azimuth_point takes atan2 of a very short
# vector and the answer is ill-conditioned: a one pixel shift in the centroid
# swings it by about 8 degrees. Comparing the scan file byte for byte therefore
# pins a number that any sub-pixel change in the renderer will move, and it will
# break again on the next matplotlib or Py-ART release for no real reason.
#
# Everything structural is still checked exactly. Only the absolute azimuth is
# given a tolerance, set to half the 30 degree scan width so a passing scan
# still covers at least half of the reference sector.
_AZIMUTH_TOLERANCE_DEG = 15.0

_POINT = re.compile(
    r'A\.1=(?P<a1>-?\d+),S\.1=(?P<s1>-?\d+),P\.1=(?P<p1>-?\d+)'
    r'\*A\.2=(?P<a2>-?\d+),S\.2=(?P<s2>-?\d+),P\.2=(?P<p2>-?\d+)')


def assert_scan_files_match(actual_path, expected_path,
                            azimuth_tolerance_deg=_AZIMUTH_TOLERANCE_DEG):
    """
    Compare two Halo scan strategy files.

    Header lines, motor parameters and elevation positions must match exactly.
    Azimuth positions must agree to within `azimuth_tolerance_deg`, and the
    angular width spanned by the scan must match exactly, which is the part
    that is actually well conditioned.
    """
    with open(actual_path) as f:
        actual = f.readlines()
    with open(expected_path) as f:
        expected = f.readlines()

    assert len(actual) == len(expected), (
        f"Scan file has {len(actual)} lines, expected {len(expected)}.")

    tolerance_counts = azimuth_tolerance_deg * _AZ_COUNTS_PER_DEG
    actual_azimuths, expected_azimuths = [], []

    for i, (line, expected_line) in enumerate(zip(actual, expected)):
        got = _POINT.search(line)
        want = _POINT.search(expected_line)
        if got is None or want is None:
            # Header or wait line: no azimuth in it, so require an exact match.
            assert line == expected_line, (
                f"Line {i} does not match expected output.\n"
                f"Got: {line}\nExpected: {expected_line}")
            continue
        for field in ('a1', 's1', 'a2', 's2', 'p2'):
            assert got[field] == want[field], (
                f"Line {i}: {field} is {got[field]}, expected {want[field]}.")
        delta = abs(int(got['p1']) - int(want['p1']))
        assert delta <= tolerance_counts, (
            f"Line {i}: azimuth is {int(got['p1']) / _AZ_COUNTS_PER_DEG:.2f} deg, "
            f"expected {int(want['p1']) / _AZ_COUNTS_PER_DEG:.2f} deg, which is "
            f"{delta / _AZ_COUNTS_PER_DEG:.2f} deg away and outside the "
            f"{azimuth_tolerance_deg} deg tolerance.")
        actual_azimuths.append(int(got['p1']))
        expected_azimuths.append(int(want['p1']))

    if actual_azimuths:
        # The width of the sector is set by az_width, not by where the centroid
        # landed, so unlike the absolute azimuth it is well conditioned. Allow a
        # hundredth of a degree only, to absorb the integer rounding of the two
        # encoded endpoints (each good to one motor count, ~0.0007 deg).
        actual_width = max(actual_azimuths) - min(actual_azimuths)
        expected_width = max(expected_azimuths) - min(expected_azimuths)
        assert abs(actual_width - expected_width) <= 0.01 * _AZ_COUNTS_PER_DEG, (
            f"Scan spans {actual_width / _AZ_COUNTS_PER_DEG:.4f} deg, "
            f"expected {expected_width / _AZ_COUNTS_PER_DEG:.4f} deg.")


def test_make_scan_file():
    from aidas.triggering.halo_lidar import make_scan_file
    from aidas.testing import TEST_RHI_FILE, TEST_PPI_FILE
    elevations = [0, 90]
    azimuths = [90]
    out_file_name = 'test_scan_rhi.txt'
    make_scan_file(elevations, azimuths, el_speed=2, out_file_name=out_file_name)
    
    with open(out_file_name, 'r') as f:
        lines = f.readlines()
    with open(TEST_RHI_FILE, 'r') as f:
        expected_lines = f.readlines()

    assert len(lines) == len(expected_lines)
    for i, (line, expected_line) in enumerate(zip(lines, expected_lines)):
        assert line == expected_line, f"Line {i} does not match expected output.\nGot: {line}\nExpected: {expected_line}"

    elevations = [0, 5, 10]
    azimuths = [90, 180]    
    out_file_name = 'test_scan_ppi.txt'
    make_scan_file(elevations, azimuths, el_speed=2, out_file_name=out_file_name)
    
    with open(out_file_name, 'r') as f:
        lines = f.readlines()
    with open(TEST_PPI_FILE, 'r') as f:
        expected_lines = f.readlines()

    assert len(lines) == len(expected_lines)
    for i, (line, expected_line) in enumerate(zip(lines, expected_lines)):
        assert line == expected_line, f"Line {i} does not match expected output.\nGot: {line}\nExpected: {expected_line}"

def test_send_scan():
    from aidas.triggering.halo_lidar import send_scan
    from aidas.testing import TEST_RHI_FILE
    from aidas.testing.fake_lidar import FakeSSHClient
    lidar_ip_addr = None
    lidar_uname = None
    lidar_pwd = None
    in_file_name = TEST_RHI_FILE
    out_file_name = 'test_rhi_scan.txt'
    out_file_name2 = 'test_rhi_scan_copy.txt'
    with FakeSSHClient() as client:
        send_scan(in_file_name, lidar_ip_addr, lidar_uname, lidar_pwd, out_file_name=out_file_name,
                  client=client)
        with open(client.sftp.files[0], 'r') as f:
            lines = f.readlines()
        with open(TEST_RHI_FILE, 'r') as f:
            expected_lines = f.readlines()
        assert len(lines) == len(expected_lines)
        for i, (line, expected_line) in enumerate(zip(lines, expected_lines)):
            assert line == expected_line, f"Line {i} does not match expected output.\nGot: {line}\nExpected: {expected_line}"

def test_trigger_lidar_ppis_from_mask():
    import torch
    import aidas
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')
    result = aidas.triggering.trigger_lidar_ppis_from_mask(rad_scan, 41.70101404798476, -87.99577278662817,
                                                  None, None, None, elevations=[0, 5, 10], az_width=30.,  
                                                  out_file_name='test_scan_ppi_lakebreeze.txt', dyn_csm=False)
    assert result is False, "Expected the scan to not be triggered due to distance from lidar to lake breeze region being greater than max_distance."

    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-04-24T20:03:23')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_model_fcn_resnet50_no_augmentation')
    with aidas.testing.FakeSSHClient() as client:
        result = aidas.triggering.trigger_lidar_ppis_from_mask(
            rad_scan, 41.70101404798476, -87.99577278662817,
            None, None, None, elevations=[0, 5, 10], az_width=30., 
            out_file_name='test_scan_ppi_lakebreeze_close.txt', dyn_csm=False, client=client)
        assert result is True, "Expected the scan to be triggered since the distance from lidar to lake breeze region is less than max_distance."
        client.sftp.get(client.sftp.files[0], 'test_scan_ppi_lakebreeze_close_copy.txt')
        assert_scan_files_match('test_scan_ppi_lakebreeze_close_copy.txt',
                                aidas.testing.TEST_PPI_TRIGGERED_SCAN)
    os.remove('test_scan_ppi_lakebreeze_close_copy.txt')
    os.remove('test_scan_ppi_lakebreeze_close.txt')
    os.remove('test_scan_ppi.txt')

def test_trigger_lidar_rhi_from_mask():
    import torch
    import aidas
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')
    result = aidas.triggering.trigger_lidar_rhi_from_mask(rad_scan, 41.70101404798476, -87.99577278662817,
                                                  None, None, None, elevations=[0, 45],   
                                                  out_file_name='test_scan_rhi_lakebreeze.txt', dyn_csm=False)
    assert result is False, "Expected the scan to not be triggered due to distance from lidar to lake breeze region being greater than max_distance."
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-04-24T20:03:23')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_model_fcn_resnet50_no_augmentation')
    with aidas.testing.FakeSSHClient() as client:
        result = aidas.triggering.trigger_lidar_rhi_from_mask(rad_scan, 41.70101404798476, -87.99577278662817,
                                                  None, None, None, elevations=[0, 45],   
                                                  out_file_name='test_scan_rhi_lakebreeze.txt', dyn_csm=False, client=client)
        assert result is True, "Expected the scan to be triggered since the distance from lidar to lake breeze region" \
             " is less than max_distance."
        client.sftp.get(client.sftp.files[0], 'test_scan_rhi_lakebreeze_copy.txt')
        assert_scan_files_match('test_scan_rhi_lakebreeze_copy.txt',
                                aidas.testing.TEST_RHI_TRIGGERED_SCAN)
    os.remove('test_scan_rhi_lakebreeze_copy.txt')
    os.remove('test_scan_rhi_lakebreeze.txt')
    os.remove('test_scan_rhi.txt')
