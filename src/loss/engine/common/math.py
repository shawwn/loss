from __future__ import annotations
import numpy as onp
import jax.numpy as np
import math
import functools
import operator


def V(*args, dtype=np.float32):
  if len(args) == 1:
    return np.asarray(args[0], dtype=dtype)
  assert len(args) > 1
  return np.array(args, dtype=dtype)

def Flat(x): return np.reshape(x, [-1])

def Eye(size, dtype=np.float32): return np.eye(size, dtype=dtype)

def Pi(): return np.pi

def Abs(x): return np.abs(x)
def Neg(x): return np.negative(x)
def Pos(x): return np.positive(x)
def Sqrt(x): return np.sqrt(x)
def Inv(x): return np.reciprocal(x)
def InvSqrt(x): return Inv(Sqrt(x))

def Sin(x): return np.sin(x)
def Cos(x): return np.cos(x)
def Tan(x): return np.tan(x)

def ASin(x): return np.arcsin(x)
def ACos(x): return np.arccos(x)
def ATan(x): return np.arctan(x)
def ATan2(x, y): return np.arctan2(x, y)

def Less(x, y): return np.less(x, y).all()
def Equal(x, y): return np.equal(x, y).all()

def Dot(x, y): return np.dot(x, y)
def MatMul(x, y): return np.matmul(x, y)
def Add(x, y): return np.add(x, y)
def Sub(x, y): return np.subtract(x, y)
def Mul(x, y): return np.multiply(x, y)
def Div(x, y): return np.divide(x, y)

def MatrixInverse(x): return np.linalg.inv(x)

def DegToRad(deg): return deg * (Pi() / 180.0)
def RadToDeg(rad): return rad * (180.0 / Pi())

def ShapeOf(v): return np.shape(v)
def NumEl(v): return onp.prod(ShapeOf(v))

def IsVector(v): return hasattr(v, 'x') and hasattr(v, 'y')

def GetDType(v):
  if isinstance(v, np.dtype):
    return v
  if hasattr(v, 'dtype'):
    return v.dtype
  if hasattr(v, '_data'):
    return GetDType(v._data)
  raise ValueError("Can't determine dtype", v)

def GetData(v, dtype):
  # #if isinstance(v, np.ndarray):
  # #  return v
  # #if isinstance(v, MData):
  # #  return v._data
  # if hasattr(v, '_data'):
  #   return v.astype(dtype)
  # if isinstance(v, (tuple, list)):
  #   return np.asarray(v, dtype=dtype)
  # return v
  return np.asarray(v, dtype=GetDType(dtype))

def SetAt(v, i, value):
  value = GetData(value, GetDType(v))
  return v.at[i].set(value)

class MData:
  def __init__(self, data):
    self._data = data

  def __jax_array__(self):
    return self._data

  def data_from(self, v):
    return GetData(v, self.dtype)

  @classmethod
  def new(cls, x):
    return cls(x)

  def binary_op(self, op, v):
    return self.new(op(self._data, self.data_from(v)))

  def unary_op(self, op):
    return self.new(op(self._data))


  def __matmul__(self, v): return self.binary_op(MatMul, v)

  def __add__(self, v): return self.binary_op(Add, v)
  def __sub__(self, v): return self.binary_op(Sub, v)
  def __mul__(self, v): return self.binary_op(Mul, v)
  def __radd__(self, v): return self.binary_op(Add, v)
  def __rsub__(self, v): return self.binary_op(Sub, v)
  def __rmul__(self, v): return self.binary_op(Mul, v)
  def __neg__(self, v): return self.unary_op(Neg)
  def __pos__(self, v): return self.unary_op(Pos)
  def __abs__(self): return self.unary_op(Abs)

  def __eq__(self, other):
    return self.binary_op(Equal, other)

  def __getitem__(self, i):
    return self._data[i]

  def __setitem__(self, i, value):
    self._data = SetAt(self._data, i, value)

  def assign(self, rhs):
    self[:] = self.data_from(rhs)[:]

  @property
  def dtype(self):
    return GetDType(self._data)


class MVec3(MData):
  def __init__(self, *args):
    super().__init__(V(0.0, 0.0, 0.0))
    if len(args) == 1:
      self[:] = args[0]
    elif len(args) == 3:
      self[0] = args[0]
      self[1] = args[1]
      self[2] = args[2]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  @staticmethod
  def from_xyz(x, y, z):
    return MVec3(x, y, z)

  @staticmethod
  def from_vec3(v):
    return MVec3(v.x, v.y, v.z)

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return '<MVec3 {}>'.format(repr(self._data))

  @property
  def x(self):
    return self._data[0]

  @property
  def y(self):
    return self._data[1]

  @property
  def z(self):
    return self._data[2]

  @x.setter
  def x(self, value):
    self[0] = value

  @y.setter
  def y(self, value):
    self[1] = value

  @z.setter
  def z(self, value):
    self[2] = value

  def dot(self, v):
    return Dot(self._data, self.data_from(v))

  @property
  def mag_sqr(self):
    x = self._data[0]
    y = self._data[1]
    z = self._data[2]
    return x*x + y*y + z*z

  @property
  def mag(self):
    return Sqrt(self.mag_sqr)

  def normalize(self):
    invMag = Inv(self.mag)
    self.x *= invMag
    self.y *= invMag
    self.z *= invMag

  def normalized(self):
    r = self.new(self)
    r.normalize()
    return r

  def cross(self, v):
    _data = self._data
    v_data = self.data_from(v)
    return MVec3( _data[ 1 ]*v_data[ 2 ] - _data[ 2 ]*v_data[ 1 ],
                  _data[ 2 ]*v_data[ 0 ] - _data[ 0 ]*v_data[ 2 ],
                  _data[ 0 ]*v_data[ 1 ] - _data[ 1 ]*v_data[ 0 ] )

  def invert(self):
    self.x = Inv(self.x)
    self.y = Inv(self.y)
    self.z = Inv(self.z)

  def inverted(self):
    r = self.new(self)
    r.invert()
    return r


class MVec4(MData):
  def __init__(self, *args):
    super().__init__(V(0.0, 0.0, 0.0, 0.0))
    if len(args) == 1:
      self[:] = args[0]
    elif len(args) == 4:
      self[0] = args[0]
      self[1] = args[1]
      self[2] = args[2]
      self[3] = args[3]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  @staticmethod
  def from_xyzw(x, y, z, w):
    return MVec3(x, y, z, w)

  @staticmethod
  def from_vec4(v):
    return MVec4(v.x, v.y, v.z, v.w)

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return '<MVec4 {}>'.format(repr(self._data))

  @property
  def x(self):
    return self._data[0]

  @property
  def y(self):
    return self._data[1]

  @property
  def z(self):
    return self._data[2]

  @property
  def w(self):
    return self._data[3]

  @x.setter
  def x(self, value):
    self[0] = value

  @y.setter
  def y(self, value):
    self[1] = value

  @z.setter
  def z(self, value):
    self[2] = value

  @w.setter
  def w(self, value):
    self[3] = value


class MMat3x3(MData):
  def __init__(self, *args):
    super().__init__(Eye(3))
    if len(args) == 1:
      self.assign(args[0])
    elif len(args) == 9:
      self[ 0, 0 ] = args[ 0 ]
      self[ 0, 1 ] = args[ 1 ]
      self[ 0, 2 ] = args[ 2 ]
      self[ 1, 0 ] = args[ 3 ]
      self[ 1, 1 ] = args[ 4 ]
      self[ 1, 2 ] = args[ 5 ]
      self[ 2, 0 ] = args[ 6 ]
      self[ 2, 1 ] = args[ 7 ]
      self[ 2, 2 ] = args[ 8 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat3x3\n{}>".format(self._data)

  def __getitem__(self, idx):
    r = super().__getitem__(idx)
    if isinstance(idx, int):
      r = MVec3(r)
    return r

  def get_at(self, i):
    # i = operator.index(i)
    # return self._data[i // 3, i % 3]
    return Flat(self._data)[i]

  def set_at(self, i, value):
    i = operator.index(i)
    self[i // 3, i % 3] = value

  def get_col(self, i):
    return MVec3(
        self[ 0,  i ],
        self[ 1,  i ],
        self[ 2,  i ])

  def set_col(self, i, col):
    col = self.data_from(col)
    self[ 0, i ] = col[0]
    self[ 1, i ] = col[1]
    self[ 2, i ] = col[2]

  def transpose(self):
    self[0, 1], self[1, 0] = self[1, 0], self[0, 1]
    self[0, 2], self[2, 0] = self[2, 0], self[0, 2]
    self[0, 3], self[3, 0] = self[3, 0], self[0, 3]

  def transposed(self):
    return self.new(
        self[0, 0], self[1, 0], self[2, 0],
        self[0, 1], self[1, 1], self[2, 1],
        self[0, 2], self[1, 2], self[2, 2])

  # def inverse_transposed(self, output: MMat3x3 = None):
  #   if output is None:
  #     r = self.new()
  #     assert self.inverse_transposed(r)
  #     return r
  #   if not output.inverse(output):
  #     return False
  #   output.transpose()
  #   return True

  def inverse_transposed(self):
    return self.inverse().transposed()

  def __mul__(self, m):
    return self.binary_op(MatMul, m)

  def inverse(self):
    return MatrixInverse(self._data)

  def rotate_point(self, point):
    point = self.data_from(point)
    _data = Flat(self._data)
    return MVec3(
      _data[ 0 ] * point[0] + _data[ 1 ] * point[1] + _data[ 2 ] * point[2],
      _data[ 3 ] * point[0] + _data[ 4 ] * point[1] + _data[ 5 ] * point[2],
      _data[ 6 ] * point[0] + _data[ 7 ] * point[1] + _data[ 8 ] * point[2] )

  def rotate_point_fast(self, point):
    _data = Flat(self._data)
    x = _data[ 0 ] * point.x + _data[ 1 ] * point.y + _data[ 2 ] * point.z
    y = _data[ 3 ] * point.x + _data[ 4 ] * point.y + _data[ 5 ] * point.z
    z = _data[ 6 ] * point.x + _data[ 7 ] * point.y + _data[ 8 ] * point.z
    point.x = x
    point.y = y
    point.z = z


  # //----------------------------------------------------------
  # void
  # MMat3x3::ToEulerYXZ( float& y, float& x, float& z )
  # {
  #     // singularity at north pole
  #     if ( _data[ 3 ] > 0.998 ) {
  #         y = ATan2( _data[ 2 ], _data[ 8 ] );
  #         z = HALF_PI;
  #         x = 0.0f;
  #         return;
  #     }
  #
  #     // singularity at south pole
  #     if ( _data[ 3 ] < -0.998 ) {
  #         y = ATan2( _data[ 2 ], _data[ 8 ] );
  #         z = -HALF_PI;
  #         x = 0.0f;
  #         return;
  #     }
  #
  #     // matrix to euler, normal case.
  #     y = ATan2( -_data[ 6 ], _data[ 0 ] );
  #     x = ATan2( -_data[ 5 ], _data[ 4 ] );
  #     z = ASin( _data[ 3 ] );
  # }
  def to_euler_yxz(self):
    _data = Flat(self._data)
    #
    # TODO: handle singularity at north pole?
    # if ( _data[ 3 ] > 0.998 ) {
    #     y = ATan2( _data[ 2 ], _data[ 8 ] );
    #     z = HALF_PI;
    #     x = 0.0f;
    #     return;
    # }
    #
    # TODO: handle singularity at south pole?
    # if ( _data[ 3 ] < -0.998 ) {
    #     y = ATan2( _data[ 2 ], _data[ 8 ] );
    #     z = -HALF_PI;
    #     x = 0.0f;
    #     return;
    # }
    #
    # matrix to euler, normal case.
    y = ATan2( -_data[6], _data[0] )
    x = ATan2( -_data[5], _data[4] )
    z = ASin( _data[3] )
    return y, x, z


  # //----------------------------------------------------------
  # void
  # MMat3x3::FromEulerYXZ( float y, float x, float z )
  # {
  #     // Assuming the angles are in radians.
  #     float ch = Cos( y );
  #     float sh = Sin( y );
  #     float cb = Cos( x );
  #     float sb = Sin( x );
  #     float ca = Cos( z );
  #     float sa = Sin( z );
  #
  #     _data[ 0 ] = ch * ca;
  #     _data[ 1 ] = sh*sb - ch*sa*cb;
  #     _data[ 2 ] = ch*sa*sb + sh*cb;
  #     _data[ 3 ] = sa;
  #     _data[ 4 ] = ca*cb;
  #     _data[ 5 ] = -ca*sb;
  #     _data[ 6 ] = -sh*ca;
  #     _data[ 7 ] = sh*sa*cb + ch*sb;
  #     _data[ 8 ] = -sh*sa*sb + ch*cb;
  # }
  @classmethod
  def from_euler_yxz(cls, y, x, z):
    # Assuming the angles are in radians.
    ch = Cos( y )
    sh = Sin( y )
    cb = Cos( x )
    sb = Sin( x )
    ca = Cos( z )
    sa = Sin( z )

    _0 = ch * ca
    _1 = sh*sb - ch*sa*cb
    _2 = ch*sa*sb + sh*cb
    _3 = sa
    _4 = ca*cb
    _5 = -ca*sb
    _6 = -sh*ca
    _7 = sh*sa*cb + ch*sb
    _8 = -sh*sa*sb + ch*cb
    return cls(_0, _1, _2,
               _3, _4, _5,
               _6, _7, _8)

  # //----------------------------------------------------------
  # void
  # MMat3x3::MakeXRotation( float angleInRad )
  # {
  #     _data[ 0 ] = 1.0f;
  #     _data[ 1 ] = 0.0f;
  #     _data[ 2 ] = 0.0f;
  #     _data[ 3 ] = 0.0f;
  #     _data[ 4 ] = Cos( angleInRad );
  #     _data[ 5 ] = -Sin( angleInRad );
  #     _data[ 6 ] = 0.0f;
  #     _data[ 7 ] = Sin( angleInRad );
  #     _data[ 8 ] = Cos( angleInRad );
  # }
  @classmethod
  def make_x_rotation(cls, angleInRad):
    _0 = 1.0
    _1 = 0.0
    _2 = 0.0
    _3 = 0.0
    _4 = Cos( angleInRad )
    _5 = -Sin( angleInRad )
    _6 = 0.0
    _7 = Sin( angleInRad )
    _8 = Cos( angleInRad )
    return cls(_0, _1, _2,
               _3, _4, _5,
               _6, _7, _8)

  # //----------------------------------------------------------
  # void
  # MMat3x3::MakeYRotation( float angleInRad )
  # {
  #     _data[ 0 ] = Cos( angleInRad );
  #     _data[ 1 ] = 0.0f;
  #     _data[ 2 ] = Sin( angleInRad );
  #     _data[ 3 ] = 0.0f;
  #     _data[ 4 ] = 1.0f;
  #     _data[ 5 ] = 0.0f;
  #     _data[ 6 ] = -Sin( angleInRad );
  #     _data[ 7 ] = 0.0f;
  #     _data[ 8 ] = Cos( angleInRad );
  # }
  @classmethod
  def make_y_rotation(cls, angleInRad):
    _0 = Cos( angleInRad )
    _1 = 0.0
    _2 = Sin( angleInRad )
    _3 = 0.0
    _4 = 1.0
    _5 = 0.0
    _6 = -Sin( angleInRad )
    _7 = 0.0
    _8 = Cos( angleInRad )
    return cls(_0, _1, _2,
               _3, _4, _5,
               _6, _7, _8)

  # //----------------------------------------------------------
  # void
  # MMat3x3::MakeZRotation( float angleInRad )
  # {
  #     _data[ 0 ] = Cos( angleInRad );
  #     _data[ 1 ] = -Sin( angleInRad );
  #     _data[ 2 ] = 0.0f;
  #     _data[ 3 ] = Sin( angleInRad );
  #     _data[ 4 ] = Cos( angleInRad );
  #     _data[ 5 ] = 0.0f;
  #     _data[ 6 ] = 0.0f;
  #     _data[ 7 ] = 0.0f;
  #     _data[ 8 ] = 1.0f;
  # }
  @classmethod
  def make_z_rotation(cls, angleInRad):
    _0  = Cos( angleInRad )
    _1  = -Sin( angleInRad )
    _2  = 0.0
    _3  = Sin( angleInRad )
    _4  = Cos( angleInRad )
    _5  = 0.0
    _6  = 0.0
    _7  = 0.0
    _8  = 1.0
    return cls(_0, _1, _2,
               _3, _4, _5,
               _6, _7, _8)

class MMat4x4(MData):
  def __init__(self, *args):
    super().__init__(Eye(4))
    if len(args) == 1:
      v = self.data_from(args[0])
      if Rank(v) == 2 and NumEl(v) == 9:
        rot = v
        self[ 0, 0 ] = rot[ 0, 0 ]
        self[ 0, 1 ] = rot[ 0, 1 ]
        self[ 0, 2 ] = rot[ 0, 2 ]

        self[ 1, 0 ] = rot[ 1, 0 ]
        self[ 1, 1 ] = rot[ 1, 1 ]
        self[ 1, 2 ] = rot[ 1, 2 ]

        self[ 2, 0 ] = rot[ 2, 0 ]
        self[ 2, 1 ] = rot[ 2, 1 ]
        self[ 2, 2 ] = rot[ 2, 2 ]
      else:
        self.assign(v)
    elif len(args) == 2:
      rot = self.data_from(args[0])
      pos = self.data_from(args[1])
      self[ 0, 0 ] = rot[ 0, 0 ]
      self[ 0, 1 ] = rot[ 0, 1 ]
      self[ 0, 2 ] = rot[ 0, 2 ]
      self[ 0, 3 ] = pos[ 0 ]

      self[ 1, 0 ] = rot[ 1, 0 ]
      self[ 1, 1 ] = rot[ 1, 1 ]
      self[ 1, 2 ] = rot[ 1, 2 ]
      self[ 1, 3 ] = pos[ 1 ]

      self[ 2, 0 ] = rot[ 2, 0 ]
      self[ 2, 1 ] = rot[ 2, 1 ]
      self[ 2, 2 ] = rot[ 2, 2 ]
      self[ 2, 3 ] = pos[ 2 ]

    elif len(args) == 16:
      self[ 0, 0 ] = args[ 0 ]
      self[ 0, 1 ] = args[ 1 ]
      self[ 0, 2 ] = args[ 2 ]
      self[ 0, 3 ] = args[ 3 ]

      self[ 1, 0 ] = args[ 4 ]
      self[ 1, 1 ] = args[ 5 ]
      self[ 1, 2 ] = args[ 6 ]
      self[ 1, 3 ] = args[ 7 ]

      self[ 2, 0 ] = args[ 8 ]
      self[ 2, 1 ] = args[ 9 ]
      self[ 2, 2 ] = args[ 10 ]
      self[ 2, 3 ] = args[ 11 ]

      self[ 3, 0 ] = args[ 12 ]
      self[ 3, 1 ] = args[ 13 ]
      self[ 3, 2 ] = args[ 14 ]
      self[ 3, 3 ] = args[ 15 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat4x4\n{}>".format(self._data)

  def __getitem__(self, idx):
    r = super().__getitem__(idx)
    if isinstance(idx, int):
      r = MVec4(r)
    return r

  def get_at(self, i):
    # return self[i // 4, i % 4]
    return Flat(self._data)[i]

  def set_at(self, i, value):
    i = operator.index(i)
    self[i // 4, i % 4] = value

  def __mul__(self, m):
    return self.binary_op(MatMul, m)

  def inverse(self):
    return self.unary_op(MatrixInverse)

  def get_rotate(self, mat: MMat3x3):
    _data = Flat(self._data)

    xInvScale = InvSqrt( _data[ 0 ]*_data[ 0 ] + _data[ 1 ]*_data[ 1 ] + _data[ 2 ]*_data[ 2 ] )
    yInvScale = InvSqrt( _data[ 4 ]*_data[ 4 ] + _data[ 5 ]*_data[ 5 ] + _data[ 6 ]*_data[ 6 ] )
    zInvScale = InvSqrt( _data[ 8 ]*_data[ 8 ] + _data[ 9 ]*_data[ 9 ] + _data[ 10 ]*_data[ 10 ] )

    mat[ 0, 0 ] = xInvScale*_data[ 0 ]
    mat[ 0, 1 ] = xInvScale*_data[ 1 ]
    mat[ 0, 2 ] = xInvScale*_data[ 2 ]

    mat[ 1, 0 ] = yInvScale*_data[ 4 ]
    mat[ 1, 1 ] = yInvScale*_data[ 5 ]
    mat[ 1, 2 ] = yInvScale*_data[ 6 ]

    mat[ 2, 0 ] = zInvScale*_data[ 8 ]
    mat[ 2, 1 ] = zInvScale*_data[ 9 ]
    mat[ 2, 2 ] = zInvScale*_data[ 10 ]

  @property
  def rotate(self):
    _data = Flat(self._data)
    xInvScale = InvSqrt( _data[ 0 ]*_data[ 0 ] + _data[ 1 ]*_data[ 1 ] + _data[ 2 ]*_data[ 2 ] )
    yInvScale = InvSqrt( _data[ 4 ]*_data[ 4 ] + _data[ 5 ]*_data[ 5 ] + _data[ 6 ]*_data[ 6 ] )
    zInvScale = InvSqrt( _data[ 8 ]*_data[ 8 ] + _data[ 9 ]*_data[ 9 ] + _data[ 10 ]*_data[ 10 ] )
    return MMat3x3( xInvScale*_data[ 0 ], xInvScale*_data[ 1 ], xInvScale*_data[ 2 ],
                    yInvScale*_data[ 4 ], yInvScale*_data[ 5 ], yInvScale*_data[ 6 ],
                    zInvScale*_data[ 8 ], zInvScale*_data[ 9 ], zInvScale*_data[ 10 ] )

  def set_rotate(self, rot):
    scale = self.scale
    self[ 0, 0 ] = scale.x * rot[ 0, 0 ]
    self[ 0, 1 ] = scale.x * rot[ 0, 1 ]
    self[ 0, 2 ] = scale.x * rot[ 0, 2 ]
    self[ 1, 0 ] = scale.y * rot[ 1, 0 ]
    self[ 1, 1 ] = scale.y * rot[ 1, 1 ]
    self[ 1, 2 ] = scale.y * rot[ 1, 2 ]
    self[ 2, 0 ] = scale.z * rot[ 2, 0 ]
    self[ 2, 1 ] = scale.z * rot[ 2, 1 ]
    self[ 2, 2 ] = scale.z * rot[ 2, 2 ]

  def get_translate(self, pos):
    _data = Flat(self._data)
    pos.x = _data[ 3 ]
    pos.y = _data[ 7 ]
    pos.z = _data[ 11 ]

  @property
  def translate(self):
    _data = Flat(self._data)
    return MVec3( _data[ 3 ], _data[ 7 ], _data[ 11 ] )

  def set_translate(self, pos):
    self[ 0, 3 ] = pos.x
    self[ 1, 3 ] = pos.y
    self[ 2, 3 ] = pos.z

  def get_scale_sqr(self, scale):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    _data = Flat(self._data)
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    scale.x = x.mag_sqr
    scale.y = y.mag_sqr
    scale.z = z.mag_sqr

  @property
  def scale_sqr(self):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    _data = Flat(self._data)
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    return MVec3( x.mag_sqr, y.mag_sqr, z.mag_sqr )


  def get_scale(self, scale):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    _data = Flat(self._data)
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    scale.x = x.mag
    scale.y = y.mag
    scale.z = z.mag

  @property
  def scale(self):
    _data = Flat(self._data)
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    return MVec3( x.mag, y.mag, z.mag )

  def set_scale(self, scale):
    scale = MVec3(scale)

    _data = Flat(self._data)
    # orthonormalize the matrix axes.
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )
    x.normalize()
    y.normalize()
    z.normalize()
    x = x * scale.x
    y = y * scale.y
    z = z * scale.z
    self.set_axes(x, y, z)

  def set_axes(self, x, y, z):
    self[ 0, 0 ] = x.x
    self[ 0, 1 ] = x.y
    self[ 0, 2 ] = x.z
    self[ 1, 0 ] = y.x
    self[ 1, 1 ] = y.y
    self[ 1, 2 ] = y.z
    self[ 2, 0 ] = z.x
    self[ 2, 1 ] = z.y
    self[ 2, 2 ] = z.z

  def set_orientation(self, side, forward):
    side = MVec3(side)
    forward = MVec3(forward)

    # compute the new matrix axes.
    x = MVec3( side.normalized() )
    z = MVec3( forward.normalized() )
    y = MVec3( z.cross( x ).normalized() )
    x = y.cross( z ).normalized()

    # scale them.
    scale = self.scale
    x = x * scale.x
    y = y * scale.y
    z = z * scale.z

    # set them.
    self.set_axes( x, y, z )

  def transform_coord(self, coord):
    coord = self.data_from(coord)
    _data = Flat(self._data)
    x = _data[ 0 ] * coord[0] + _data[ 1 ] * coord[1] + _data[  2 ] * coord[2] + _data[ 3 ]
    y = _data[ 4 ] * coord[0] + _data[ 5 ] * coord[1] + _data[  6 ] * coord[2] + _data[ 7 ]
    z = _data[ 8 ] * coord[0] + _data[ 9 ] * coord[1] + _data[ 10 ] * coord[2] + _data[ 11 ]
    invW = Inv( _data[ 12 ]*coord[0] + _data[ 13 ]*coord[1] + _data[ 14 ]*coord[2] + _data[ 15 ] )
    return MVec3( x * invW, y * invW, z * invW )

  def transform_coord_no_persp(self, coord):
    coord = self.data_from(coord)
    _data = Flat(self._data)
    x = _data[ 0 ] * coord[0] + _data[ 1 ] * coord[1] + _data[  2 ] * coord[2] + _data[ 3 ]
    y = _data[ 4 ] * coord[0] + _data[ 5 ] * coord[1] + _data[  6 ] * coord[2] + _data[ 7 ]
    z = _data[ 8 ] * coord[0] + _data[ 9 ] * coord[1] + _data[ 10 ] * coord[2] + _data[ 11 ]
    return MVec3( x, y, z )

  def transform_coord_no_persp_fast(self, coord):
    _data = Flat(self._data)
    x = _data[ 0 ] * coord[0] + _data[ 1 ] * coord[1] + _data[  2 ] * coord[2] + _data[ 3 ]
    y = _data[ 4 ] * coord[0] + _data[ 5 ] * coord[1] + _data[  6 ] * coord[2] + _data[ 7 ]
    z = _data[ 8 ] * coord[0] + _data[ 9 ] * coord[1] + _data[ 10 ] * coord[2] + _data[ 11 ]
    coord.x = x
    coord.y = y
    coord.z = z


class MPlane:
  def __init__(self, *args):
    self._normal = MVec3(0.0, 1.0, 0.0)
    self._d = V(0.0)
    if len(args) == 1:
      self.assign(args[0])
    elif len(args) == 2:
      self._normal.assign(args[0])
      self._normal.normalize()
      if IsVector(args[1]):
        self.set_d(-self._normal.dot(args[1]))
      else:
        self.set_d(args[1])
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MPlane normal=({}, {}, {}) d={}>".format(self._normal.x, self._normal.y, self._normal.z, self._d)

  @property
  def normal(self):
    return self._normal

  @property
  def d(self):
    return self._d

  def set_normal(self, value):
    self._normal.assign(value)

  def set_d(self, value):
    self._d = V(value)

  def assign(self, rhs):
    self.set_normal(rhs.normal)
    self.set_d(rhs.d)

  def dist(self, point):
    return self._normal.dot(point) + self._d


