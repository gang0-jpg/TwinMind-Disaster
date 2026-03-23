import numpy as np
import matplotlib.pyplot as plt

dem = np.load('/home/team-010/data/twinmind_disaster/processed_dem/mosaic/dem_mosaic.npy')

plt.figure(figsize=(8, 8))
plt.imshow(dem, origin='upper')
plt.colorbar(label='Elevation (m)')
plt.title('DEM Mosaic')
plt.tight_layout()
plt.savefig('/home/team-010/data/twinmind_disaster/processed_dem/mosaic/dem_mosaic.png', dpi=150)
print("[OK] saved: /home/team-010/data/twinmind_disaster/processed_dem/mosaic/dem_mosaic.png")
