import yourdfpy

robot = yourdfpy.URDF.load("ABH_URDF/ability_hand.urdf")

m1 = robot.get_transform("index_L1", "base")
print(m1)

# robot.show()
