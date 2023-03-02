from loss.engine import window
from loss.engine.common import math as m
from loss.engine import graphics as g
import time

class App:
  def __init__(self, width=800, height=600):
    self.win = window.Window(width, height, "App")
    self.start_time = time.time()
    self.last_time = self.start_time
    self.fps_count = 0
    self.fps_time = self.last_time
    self.cam = g.GrCamera()
    self.win.loop(self)

  def __call__(self):
    cur_time = time.time()
    dtime = cur_time - self.last_time

    theta = 0.01 * (cur_time - self.start_time)
    rot = m.MMat3x3.from_euler_yxz(theta, 0.0, 0.0)
    self.cam.set_rot(rot)


    # self.clear([0.1, 0.1, 0.1])
    # self.draw(self.indices,
    #           vertices=self.vertices,
    #           normals=self.normals,
    #           uvs=self.uvs,
    #           texture=self.texture,
    #           wvp=wvp)
    # image = self.rend.execute()

    # Display frames per second
    self.last_time = cur_time
    self.fps_count += 1
    if cur_time - self.fps_time > 1.0:
      print(f"fps: {self.fps_count}")
      print([m.RadToDeg(v) for v in rot.to_euler_yxz()])
      print(repr(self.cam.view_matrix))
      self.fps_count = 0
      self.fps_time = cur_time

    # return image

App()