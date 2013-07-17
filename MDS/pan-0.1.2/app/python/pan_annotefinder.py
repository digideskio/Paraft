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

 # Notice: This class was modified by Paulo Joia Filho
 # from: Cookbook / Matplotlib / Interactive Plotting,
 # available on:
 # http://www.scipy.org/Cookbook/Matplotlib/Interactive_Plotting


import pylab 
from pylab import *

class AnnoteFinder:
  def __init__(self, xdata, ydata, annotes, axis=None, xtol=None, ytol=None):
    self.data = zip(xdata, ydata, annotes)
    if xtol is None:
      xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
    if ytol is None:
      ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
    self.xtol = xtol
    self.ytol = ytol
    if axis is None:
      self.axis = pylab.gca()
    else:
      self.axis= axis
    self.drawnAnnotations = {}
    self.links = []

  def distance(self, x1, x2, y1, y2):
    """
    return the distance between two points
    """
    return(math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 ))

  def __call__(self, event):
    if event.inaxes:
      clickX = event.xdata
      clickY = event.ydata
      r=2 #raio      
      if (self.axis is None) or (self.axis==event.inaxes):
        annotes = []
        for x,y,a in self.data:
          if  (clickX-self.xtol-r < x < clickX+self.xtol+r) and  (clickY-self.ytol-r < y < clickY+self.ytol+r) :
            annotes.append((self.distance(x,clickX,y,clickY),x,y,a) )
        if annotes:
          annotes.sort()
          distance, x, y, annote = annotes[0]
          self.drawAnnote(event.inaxes, x, y, annote)
          for l in self.links:
            l.drawSpecificAnnote(annote)

  def drawAnnote(self, axis, x, y, annote):
    """
    Draw the annotation on the plot
    """
    if self.drawnAnnotations.has_key((x,y)):
      markers = self.drawnAnnotations[(x,y)]
      for m in markers:
        m.set_visible(not m.get_visible())
      self.axis.figure.canvas.draw()
    else:
      #t = axis.text(x,y, " %i"%(annote), ) # label only numbers
      t = axis.text(x,y, annote, )
      m = axis.scatter([x],[y], marker='o', c='none')
      self.drawnAnnotations[(x,y)] =(t,m)
      self.axis.figure.canvas.draw()

  def drawSpecificAnnote(self, annote):
    annotesToDraw = [(x,y,a) for x,y,a in self.data if a==annote]
    for x,y,a in annotesToDraw:
      self.drawAnnote(self.axis, x, y, a)
