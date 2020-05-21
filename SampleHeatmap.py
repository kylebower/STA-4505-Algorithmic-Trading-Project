import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Get a data set of optimal actions for pairs (inventory,price)
# At a fixed time
# Generate sample data
data = np.random.rand(10, 10)
print(data)

# Columns are x axis (inventory) and rows are y axis (price)

cmap = sb.diverging_palette(20, 220, as_cmap=True)
heat_map = sb.heatmap(data, cmap=cmap)
plt.xlabel("Inventory")
plt.ylabel("Price")
plt.show()
