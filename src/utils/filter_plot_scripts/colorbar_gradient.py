import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use(['science','ieee','grid'])
#plt.style.use(['science','grid'])
matplotlib.rcParams.update(
        {
            "font.family": "serif",  # Font family to be used
            "font.serif": "Times New Roman", # Times New Roman
            "text.usetex": True,  # Render texts/mathematics using pdflatex compiler
            "legend.fontsize": 12
        }
)
# Generate random data within the specified interval
data = np.random.uniform(low=4.5, high=7.0, size=(10, 10))

# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='coolwarm', extent=[4.5, 7.0, 0, 1])  # extent specifies the data limits for the image
cbar = plt.colorbar(orientation='horizontal', location='bottom', format=mticker.FixedFormatter(['Compact', 'Colossal']), extend='both')

# Save the plot for the later edditing
plt.savefig("colorbar_gradient.pdf", dpi=600)
# Show plot
#plt.show()
