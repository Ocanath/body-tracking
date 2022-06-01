from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.animation as animation
import numpy as np
import time
from vect_tools import *
import yourdfpy



robot = yourdfpy.URDF.load("ABH_URDF/ability_hand.urdf")
q1 = 10*np.pi/180
q2 = 51*np.pi/180
cfg = dict(index_q1=q1,
	middle_q1=q1,
	ring_q1=q1, 
	pinky_q1=q1, 
	index_q2=q2, 
	middle_q2=q2, 
	pinky_q2=q2, 
	ring_q2=q2, 
	thumb_q1=-30*np.pi/180, 
	thumb_q2=20*np.pi/180)
robot.update_cfg(cfg)


figure = pyplot.figure()
ax=mplot3d.Axes3D(figure)


# Load the STL files and add the vectors to the plot
base = mesh.Mesh.from_file('ABH_URDF/models/FB_palm_ref.STL')
base.transform(robot.get_transform("base", "base"))
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(base.vectors))
combinedMesh = base


mesh_name_str = ["index_mesh_", 
	"middle_mesh_",
	"ring_mesh_",
	"pinky_mesh_"]

for mesh_name in mesh_name_str:
	"""
	Get Index finger geometry and frame reference from urdf
	"""
	pMesh = mesh.Mesh.from_file('ABH_URDF/models/idx-F1.STL')
	Hlink = robot.get_transform(mesh_name+"1", "base")
	pMesh.transform(Hlink)
	ax.add_collection3d(mplot3d.art3d.Poly3DCollection(pMesh.vectors))
	combinedMesh = mesh.Mesh(np.concatenate([combinedMesh.data, pMesh.data]))

	"""
	Get Index finger geometry and frame reference from urdf
	"""
	pMesh = mesh.Mesh.from_file('ABH_URDF/models/idx-F2.STL')
	Hlink = robot.get_transform(mesh_name+"2", "base")
	pMesh.transform(Hlink)
	ax.add_collection3d(mplot3d.art3d.Poly3DCollection(pMesh.vectors))
	combinedMesh = mesh.Mesh(np.concatenate([combinedMesh.data, pMesh.data]))



pMesh = mesh.Mesh.from_file('ABH_URDF/models/thumb-F1.STL')
Hlink = robot.get_transform("thumb_mesh_1", "base")
pMesh.transform(Hlink)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(pMesh.vectors))
combinedMesh = mesh.Mesh(np.concatenate([combinedMesh.data, pMesh.data]))

pMesh = mesh.Mesh.from_file('ABH_URDF/models/thumb-F2.STL')
Hlink = robot.get_transform("thumb_mesh_2", "base")
pMesh.transform(Hlink)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(pMesh.vectors))
combinedMesh = mesh.Mesh(np.concatenate([combinedMesh.data, pMesh.data]))


pMesh = mesh.Mesh.from_file('ABH_URDF/models/WRISTADAPTER.STL')
pMesh.transform(robot.get_transform("base", "base"))
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(pMesh.vectors))
combinedMesh = mesh.Mesh(np.concatenate([combinedMesh.data, pMesh.data]))


# Auto scale to the mesh size
ax.axes.set_xlim3d(-.1,0.1)
ax.axes.set_ylim3d(-.1,0.1)
ax.axes.set_zlim3d(-.1,0.1)

combinedMesh.save('combined.stl')

# Show the plot to the screensoli
# pyplot.show()

print("Success!")