from loss.engine import window
import time

class App:
  def __init__(self, width=800, height=600):
    self.win = window.Window(width, height, "App")
    self.start_time = time.time()
    self.last_time = self.start_time
    self.fps_count = 0
    self.fps_time = self.last_time
    self.win.loop(self)

  def __call__(self):
    theta = 0.01 * (time.time() - self.start_time)
    # wvp = utils.rotation(0., theta, 0.0)

    # self.clear([0.1, 0.1, 0.1])
    # self.draw(self.indices,
    #           vertices=self.vertices,
    #           normals=self.normals,
    #           uvs=self.uvs,
    #           texture=self.texture,
    #           wvp=wvp)
    # image = self.rend.execute()

    # Display frames per second
    cur_time = time.time()
    dtime = cur_time - self.last_time
    self.last_time = cur_time
    self.fps_count += 1
    if cur_time - self.fps_time > 1.0:
      # fps = 1.0 / float(dtime + 1e-9)
      fps = self.fps_count
      print("fps: %.1f" % fps)
      self.fps_count = 0
      self.fps_time = cur_time

    # return image

App()