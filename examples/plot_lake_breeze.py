"""
Example on how to plot lake breeze and determine optimal radar pointing direction
=================================================================================

Example workflow on how to run AIDAS and plot the resulting lake breeze. In
addition, we will set up an instrument at ATMOS and show the angle at which
the radar should be pointed to optimally capture the lake breeze.
"""

import matplotlib.pyplot as plt
import numpy as np

import aidas

# %%
# The location of the Argonne Testbed for Multiscale Observational Studies (ATMOS)

atmos_location = (41.70101404798476, -87.99577278662817)  # (lat, lon)

# %%
# Load a radar scan and run the lake breeze model.

rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
rad_scan = aidas.model.infer_lake_breeze(
    rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')

# %%
# Calculate the azimuth angle from the radar to the lake breeze front center.

angle, lat, lon, dist = aidas.util.azimuth_point(atmos_location[1], atmos_location[0], rad_scan)

print(f'Azimuth angle to point radar: {angle:.2f} degrees'
      f'\nLatitude: {lat:.4f}, Longitude: {lon:.4f}')

# %%
# Plot the lake breeze with the radar pointing direction.

fig, ax = aidas.vis.visualize_lake_breeze(rad_scan, bg_field='reflectivity')
plt.scatter(atmos_location[1], atmos_location[0], c='red', s=100, label='ATMOS')
plt.scatter(lon, lat, c='blue', s=100, label='Lake Breeze Center')

# Note the angle is clockwise from North
plt.quiver(atmos_location[1], atmos_location[0], np.sin(np.deg2rad(angle)),
           np.cos(np.deg2rad(angle)), scale=5, color='k')
plt.legend()
plt.show()
