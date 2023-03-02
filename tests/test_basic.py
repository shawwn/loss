import unittest

from loss import m

class TestCase(unittest.TestCase):
  def test_basic(self):
    self.assertEqual(1, 1)

  def test_add(self):
    vec = m.MVec3(2, 2, 2)
    self.assertEqual(vec, vec)
    self.assertEqual(vec + vec - vec, vec)
    self.assertEqual(12, vec.dot(vec))
    self.assertEqual((0.5, 0.5, 0.5), vec.inverted())


if __name__ == '__main__':
  unittest.main()
