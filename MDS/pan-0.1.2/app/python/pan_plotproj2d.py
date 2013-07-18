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
from sys import argv, exit
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

from pan_annotefinder import AnnoteFinder
from pan_useful import *
from pan_objects import *
from pan_importdata import PAnImportData


def default_legend(lecount, class_enum):
    letext = [''] * lecount
    for i in range(lecount):
        if isnumber(class_enum[i]):
            letext[i] = 'class %d' % float(class_enum[i])
        else:
            letext[i] = '%s' % class_enum[i]
    return letext


def usage_syntax():
    print ''
    print 'Usage syntax:'
    print '============='
    print 'pan_plotproj2d -infile <projection_file> [-saveim <yes|no>]'
    print '               [-legend <yes|no>] [-letext <legend_text>]'
    print '\nFor more details see "plot_projection" documentation.\n'


def plot_projection2d():
    argc = len(argv)

    # default arguments
    infile = ''
    wintit = ''
    saveim = 0
    legend = 0
    letext = []

    # process arguments
    for i in range(1, argc):
        if argv[i] == '-infile':
            infile = argv[i + 1]
            wintit = os.path.basename(infile)
        elif argv[i] == '-saveim':
            saveim = 1 if argv[i + 1] == 'yes' else 0
        elif argv[i] == '-legend':
            legend = 1 if argv[i + 1] == 'yes' else 0
        elif argv[i] == '-letext':
            letext = argv[i + 1].split(',')

            # verify input data file
    if not infile:
        print '\n*** Fail: input data file is required!'
        usage_syntax()
        exit()
    if not os.path.exists(infile):
        print '\n*** Fail: the provided file not exists!'
        exit()

    # load PAn Libs
    libpanuseful, libpandconv = pan_loadlibs()
    c_decimal = decimal_type(libpanuseful)

    # import data
    importdata = PAnImportData(libpandconv)
    importdata.decimalctype_set(c_decimal)

    extfile = os.path.splitext(infile)[1]
    if (extfile == '.data') or (extfile == '.prj'):
        dataset, ids, classes = importdata.pex_importdata(infile)
    elif (extfile == '.arff'):
        dataset, ids, classes = importdata.weka_importdata(infile)
    else:
        print '\n*** Fail: invalid file type!'
        exit()

    # prepare data
    numrows = ids.size
    X = np.zeros(numrows)
    Y = np.zeros(numrows)
    ID = [''] * numrows

    for i in range(numrows):
        X[i] = dataset[i * 2]
        Y[i] = -dataset[i * 2 + 1]
        ID[i] = ids.values[i]

    class_enum = [''] * classes.enum_size
    for i in range(classes.enum_size):
        class_enum[i] = classes.enumeration[i]
    class_enum = sorted(class_enum)

    # load color palette     
    RGB = np.genfromtxt(os.getcwd() + '/colormap/projection_analyzer_scale.txt', dtype='<f8,<f8,<f8', delimiter=";")
    colormap_step = 1.0
    colormap_size = RGB.size

    if classes.enum_size == 0 or classes.enum_size > colormap_size:
        RGB = np.genfromtxt(os.getcwd() + '/colormap/pseudo_rainbow_scale.txt', dtype='<f8,<f8,<f8', delimiter=";")
        colormap_size = RGB.size

        if classes.enum_size and classes.enum_size < RGB.size:
            colormap_step = RGB.size / float(classes.enum_size - 1)
            colormap_size = classes.enum_size

    RGBsz = RGB.size
    colours = []
    for i in range(colormap_size):
        j = round(i * colormap_step)
        if j >= RGBsz: j = RGBsz - 1
        colours.append(rgb2hex(RGB[j]))

        # plot
    fig = plt.figure(figsize=(9, 6))
    fig.canvas.set_window_title(wintit)
    rmarg = 0.805 if legend else 0.935
    plt.subplots_adjust(left=0.06, right=rmarg, top=0.95, bottom=0.06)
    plt.grid(True)
    ax = fig.add_subplot(111)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if classes.enum_size:
        IDcl = [[] for n in range(classes.enum_size)]
        Xcl = [[] for n in range(classes.enum_size)]
        Ycl = [[] for n in range(classes.enum_size)]

        for i in range(numrows):
            val = class_enum.index(classes.values[i]) % RGBsz
            IDcl[val].append(ID[i])
            Xcl[val].append(X[i])
            Ycl[val].append(Y[i])

        for i in range(classes.enum_size):
            plt.scatter(Xcl[i], Ycl[i], s=30, c=colours[i], marker="o")
    else:
        plt.scatter(X, Y, s=30, c=colours, marker="o")

        # legend
    if legend and classes.enum_size:

        # amount
        lecount = len(letext) if letext else classes.enum_size
        if lecount > len(colours):
            print '\n*** Fail: legend with many items. Disable the legend.'
            exit()
        if not letext:
            letext = default_legend(lecount, class_enum)

            # items
        leitems = []
        for i in range(lecount):
            leitems.append(plt.Rectangle((0, 0), 1, 1, fc=colours[i]))

        font = FontProperties(size='medium')
        plt.legend((leitems), (letext), loc='upper left', bbox_to_anchor=(1.0, 1.02), prop=font)

        # save plot as png
    if saveim:
        outfile = infile.replace('.prj', '.png')
        outfile = outfile.replace('.data', '.png')
        outfile = outfile.replace('.arff', '.png')
        plt.savefig(outfile)

    # label of points 
    af = AnnoteFinder(X, Y, ID)
    connect('button_press_event', af)

    plt.show()
    print 'Finished!'


#--- Debug test
#argv = ['pan_plotproj2d.py',       
#        '-infile', '../../data/iris_lamp.arff',  
#        '-saveim', 'no', 
#        '-legend', 'yes']    
#       #'-letext', 'helicopter,revolver,sunflower']

if __name__ == '__main__':
    plot_projection2d()

