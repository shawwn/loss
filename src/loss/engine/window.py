"""Create a window for rendering."""
import glfw
import numpy as np
import OpenGL.GL as gl

class Window:
  """Create a window for rendering."""
  width: int
  height: int

  def __init__(self, width: int, height: int, title: str):
    self.width = width
    self.height = height

    if not glfw.init():
      return

    self.win = glfw.create_window(width, height, title, None, None)
    if not self.win:
      glfw.terminate()
      return

    zeros = np.zeros([height, width, 3], dtype=np.uint8)

    glfw.make_context_current(self.win)

    self.tid = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.tid)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, zeros)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                       gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                       gl.GL_LINEAR)

  def loop(self, update_fn):
    """Loop."""
    while not glfw.window_should_close(self.win):
      image = update_fn()

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

      if image is not None:
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tid)
        gl.glTexSubImage2D(
          gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, gl.GL_RGB,
          gl.GL_UNSIGNED_BYTE, image)

      gl.glBegin(gl.GL_QUADS)
      gl.glTexCoord2f(0.0, 1.0)
      gl.glVertex3f(-1.0, 1.0, 0.0)
      gl.glTexCoord2f(1.0, 1.0)
      gl.glVertex3f(1.0, 1.0, 0.0)
      gl.glTexCoord2f(1.0, 0.0)
      gl.glVertex3f(1.0, -1.0, 0.0)
      gl.glTexCoord2f(0.0, 0.0)
      gl.glVertex3f(-1.0, -1.0, 0.0)
      gl.glEnd()

      glfw.swap_buffers(self.win)
      glfw.poll_events()

    glfw.terminate()
