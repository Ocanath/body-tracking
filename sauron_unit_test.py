import unittest
import math

from serialhelper import fletchers_checksum16, create_sauron_position_payload
from PPP_stuffing import PPP_stuff
from sauron_ik import get_ik_angles_double

class TestSerialHelper(unittest.TestCase):
	def test_create_position_payload(self):
		th1 = -2255
		th2 = 944
		payload = create_sauron_position_payload(th1, th2)
		a = []
		for p in payload:
			a.append(int(p))
		print(a)
		b = bytearray([126, 250, 0, 49, 247, 255, 255, 176, 3, 0, 0, 3, 233, 126])
		a = []
		for p in b:
			a.append(int(p))
		print(a)

		self.assertEqual(b, payload)

	def test_ppp_stuff(self):
		b_in = bytearray([1,2,3])
		btrue = bytearray([126,1,2,3,126])
		btest = PPP_stuff(b_in)
		self.assertEqual(btrue, btest)

	def test_sauron_ik(self):
		t1, t2 = get_ik_angles_double(-3., 3., 10.)
		th1_deg = t1*180.0/math.pi
		th2_deg = t2*180.0/math.pi
		self.assertAlmostEqual(-16.70, th1_deg, places=2)
		self.assertAlmostEqual(15.39, th2_deg, places=2)

if __name__ == '__main__':
	unittest.main()