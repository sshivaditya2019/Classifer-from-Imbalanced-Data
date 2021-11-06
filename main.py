import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

requires = open('requirements.txt', 'r')
dependencies = [line.split(',') for line in requires.readlines()]


pkg_resources.require(dependencies)
