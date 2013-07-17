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

from ctypes import *
from pan_objects import *
 
class PAnImportData:

    def __init__(self, libpandconv):
        self.__libpandconv = libpandconv        
        self.__maxclass = libpandconv.maxclass_get() 
        self.__c_decimal = c_float 
        
    def decimalctype_set(self, c_decimal): 
        self.__c_decimal = c_decimal 
    
    def pex_importdata(self, fullfilename):
        c_decimal = self.__c_decimal         
        fullfilename = c_char_p(fullfilename)
        
        # header
        self.__libpandconv.pex_importheader.argtypes = [c_char_p, c_char_p, c_char_p, POINTER(c_int), 
                                                        POINTER(c_int), POINTER(c_ubyte)]                                                        
        numrows = c_int(0)
        numcols = c_int(0)                
        isconsist = c_ubyte(0)
        self.__libpandconv.pex_importheader(fullfilename, None, None, pointer(numrows), 
                                     pointer(numcols), pointer(isconsist))                
        if not isconsist.value:        
            print "\nPEx file '%s' is inconsistent.\n" \
                   "Id (first column) and/or Class (last column) were not found.\n" \
                   "Aborted operation!\n\n" % fullfilename.value
            exit()
            
        # data
        self.__libpandconv.pex_importdata.argtypes = [c_char_p, c_int, c_int, POINTER(c_decimal), 
                                                      POINTER(id_struct), POINTER(class_struct)]

        dataset = (c_decimal * (numrows.value * numcols.value))()
        ids = id_struct((c_char_p * numrows.value)(), numrows, c_ubyte(GET_ID))
        classes = class_struct((c_char_p * numrows.value)(), numrows, c_ubyte(GET_CLASS), 
                               (c_char_p * self.__maxclass)(), c_int(0), c_ubyte(0))
           
        self.__libpandconv.pex_importdata(fullfilename, numrows, numcols, dataset, 
                                          pointer(ids), pointer(classes))                               
                                  
        return dataset, ids, classes
        
      
    def weka_importdata(self, fullfilename):
        c_decimal = self.__c_decimal        
        fullfilename = c_char_p(fullfilename)
        
        # header     
        self.__libpandconv.weka_importheader.argtypes = [c_char_p, c_char_p, POINTER(c_char_p), 
                                                         POINTER(c_int), POINTER(c_int), POINTER(c_int), 
                                                         POINTER(c_ubyte), POINTER(c_ubyte)]                                                          
        numrows = c_int(0)
        numcols = c_int(0)
        startrow = c_int(0)                                                       
        hasid = c_ubyte(0)
        hascl = c_ubyte(0)
        self.__libpandconv.weka_importheader(fullfilename, None, None, pointer(numrows), pointer(numcols), 
                                             pointer(startrow), pointer(hasid), pointer(hascl)) 
                
        # data
        self.__libpandconv.weka_importdata.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_decimal), 
                                                       POINTER(id_struct), POINTER(class_struct)]
                
        dataset = (c_decimal * (numrows.value * numcols.value))()
        
        retrieve = c_ubyte(GET_ID) if hasid.value else c_ubyte(CREATE_ID)        
        ids = id_struct((c_char_p * numrows.value)(), numrows, retrieve) 
        
        retrieve = c_ubyte(GET_CLASS) if hascl.value else c_ubyte(NO_CLASS_IN_FILE)                
        classes = class_struct((c_char_p * numrows.value)(), numrows, retrieve,   
                               (c_char_p * self.__maxclass)(), c_int(0), c_ubyte(0))
                     
        self.__libpandconv.weka_importdata(fullfilename, numrows, numcols, startrow, dataset, 
                                           pointer(ids), pointer(classes)) 

        return dataset, ids, classes
