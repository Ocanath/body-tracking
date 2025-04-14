import unittest
from serialhelper import fletchers_checksum16, create_sauron_position_payload
from PPP_stuffing import PPP_stuff

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


if __name__ == '__main__':
	unittest.main()