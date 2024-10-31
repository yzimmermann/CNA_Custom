import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Generate random data within the specified interval
data = np.random.uniform(low=4.5, high=9.5, size=(10, 10))

# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='coolwarm', extent=[4.5, 9.5, 0, 1])  # extent specifies the data limits for the image
cbar = plt.colorbar(orientation='horizontal', format=mticker.FixedFormatter(['<', '>']), extend='both')


# Show plot
plt.show()
