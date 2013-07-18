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

import os
from ctypes import *

#
# Libs
#
def pan_loadlibs():
    libpath = "../../lib/"    
    libext = ".dylib" if os.name == "posix" else ".dll"        
    libpanuseful = cdll.LoadLibrary(libpath + "libpanuseful" + libext)    
    libpandconv = cdll.LoadLibrary(libpath + "libpandconv" + libext) 
    return libpanuseful, libpandconv

#
# data types
#         
def decimal_type(libpanuseful):     
    precision = (c_float, c_double)
    c_decimal = precision[libpanuseful.precision_get()]     
    return c_decimal 
    
#
# global enumerations
# 
GET_ID, DO_NOT_GET_ID, NO_ID_IN_FILE, CREATE_ID = 1, 2, 3, 4     
GET_CLASS, DO_NOT_GET_CLASS, NO_CLASS_IN_FILE = 1, 2, 3 

#
# global structures
# 
class id_struct(Structure):
    _fields_ = [("values", POINTER(c_char_p)), 
                ("size", c_int), 
                ("retrieve", c_ubyte)] 
                 
class class_struct(Structure):
    _fields_ = [("values", POINTER(c_char_p)), 
                ("size", c_int), 
                ("retrieve", c_ubyte),                 
                ("enumeration", POINTER(c_char_p)),
                ("enum_size", c_int),
                ("pex_consist", c_ubyte)] 
        



