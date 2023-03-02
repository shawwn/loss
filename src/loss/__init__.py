# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from loss.engine.common import math as m
from loss.engine import graphics as g