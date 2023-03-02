import unittest

import loss
import loss.engine.common.math as m
import loss.engine.graphics as g

class TestCase(unittest.TestCase):
  def test_basic(self):
    self.assertEqual(1, 1)

  def test_add(self):
    vec = m.MVec3(1, 1, 1)
    self.assertEqual(vec, vec)
    self.assertEqual(vec + vec - vec, vec)
    self.assertGreater(vec + vec, vec)


if __name__ == '__main__':
  unittest.main()
