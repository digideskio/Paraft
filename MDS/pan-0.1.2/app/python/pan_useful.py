#!/usr/bin/python

###############################################################################
#
# Copyright (C) 2011 Paulo Joia Filho
# University of Sao Paulo - Sao Carlos/SP, Brazil.
# All Rights Reserved.
#
# This file is part of Projection Analyzer (PAn).
#
# PAn is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# PAn is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
# License for more details.
#
# This code was developed by Paulo Joia Filho <pjoia@icmc.usp.br>
# at Institute of Mathematics and Computational Sciences - ICMC
# University of Sao Paulo, Sao Carlos/SP, Brazil (http://www.icmc.usp.br)
#
# Contributor(s):  Luis Gustavo Nonato <gnonato@icmc.usp.br>
#
# You should have received a copy of the GNU Lesser General Public License
# along with PAn. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


def rgb2hex(rgb):
    """Given a len 3 rgb tuple of 0-1 floats, return the hex string"""
    return '#%02x%02x%02x' % tuple(rgb)


def isnumber(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False

    return True
