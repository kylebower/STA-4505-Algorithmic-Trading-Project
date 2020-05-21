import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get a data set of optimal actions for pairs (inventory,price)
# At a fixed time
# Generate sample data
data = np.random.rand(10, 10)
print(data)

# Columns are x axis (inventory) and rows are y axis (price)

cmap = sns.diverging_palette(20, 240, as_cmap=True)  # husl color system
heat_map = sns.heatmap(data, cmap=cmap)
plt.xlabel("Inventory")
plt.ylabel("Price")
plt.show()
