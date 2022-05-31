from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.animation as animation
import numpy as np
import time
from vect_tools import *
import yourdfpy



robot = yourdfpy.URDF.load("ABH_URDF/ability_hand.urdf")



figure = pyplot.figure()
ax=mplot3d.Axes3D(figure)


# Load the STL files and add the vectors to the plot
base = mesh.Mesh.from_file('ABH_URDF/models/FB_palm_ref.STL')
base.transform(robot.get_transform("base", "base"))
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(base.vectors))


Il1 = mesh.Mesh.from_file('ABH_URDF/models/idx-F1.STL')
MIDX = robot.get_transform("index_L2", "base")
Il1.transform(MIDX)
xyz,rpy = get_xyz_rpy(MIDX)
print('INDEX: ', 'xyz = ', xyz*1e3, 'rpy = ', rpy*180/np.pi)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(Il1.vectors))
base


Il2 = mesh.Mesh.from_file('ABH_URDF/models/idx-F2.STL')
m = robot.get_transform("index_L2", "base")
m = np.dot(m,ht_rotz(53.1*np.pi/180))
Il2.transform(m)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(Il2.vectors))


# fl2 = mesh.Mesh.from_file('ABH_URDF/models/idx-F2.STL')
# fl2.transform(robot.get_transform("base", "index_L2"))
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(fl2.vectors))


# Auto scale to the mesh size
ax.axes.set_xlim3d(-.1,0.1)
ax.axes.set_ylim3d(-.1,0.1)
ax.axes.set_zlim3d(-.1,0.1)



combinedMesh = mesh.Mesh(np.concatenate([base.data, Il1.data, Il2.data]))
combinedMesh.save('combined.stl')

# Show the plot to the screensoli
pyplot.show()


