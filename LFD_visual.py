import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


x_1 = np.array([2,2])
x_2 = np.array([-2,-2])

z = np.array([1,5])

Lambda = 1
Reg = 4


X_grid = np.arange(-4, 4, 0.25)
Y_grid = np.arange(-4, 4, 0.25)
X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)

N, _ = X_grid.shape


Z_grid_1 = np.zeros([N,N])
Z_grid_2 = np.zeros([N,N])

for i in range(N):
    for j in range(N):
        grid_ij = np.array([X_grid[i,j], Y_grid[i,j]])
        Dist_ij = np.sum((grid_ij - x_1)**2)
        Residual_ij = (np.sum(x_1*z) - Lambda * Dist_ij)/(Lambda * Reg)
        Z_grid_1[i,j] = np.exp(Residual_ij)
Z_grid_1 = Z_grid_1/np.sum(Z_grid_1)

for i in range(N):
    for j in range(N):
        grid_ij = np.array([X_grid[i,j], Y_grid[i,j]])
        Dist_ij = np.sum((grid_ij - x_2)**2)
        Residual_ij = (np.sum(x_2*z) - Lambda * Dist_ij)/(Lambda * Reg)
        Z_grid_2[i,j] = np.exp(Residual_ij)
Z_grid_2 = Z_grid_2/np.sum(Z_grid_2)

Z_grid = Z_grid_1 + Z_grid_2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.2)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.axis('off')
plt.savefig("LFD.pdf", bbox_inches='tight')