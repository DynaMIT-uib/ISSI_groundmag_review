#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:19:20 2023

@author: zettergm
"""

import gemini3d.read
import gemini3d.grid.gridmodeldata
import matplotlib.pyplot as plt
import numpy as np

direc="/Users/zettergm/simulations/ssd/aurora_EISCAT3D_precip/"

cfg=gemini3d.read.config(direc)
xg=gemini3d.read.grid(direc)
dat=gemini3d.read.frame(direc,time=cfg["time"][30])

interpsize=(512,96,512)
alti,mloni,mlati,nei=gemini3d.grid.gridmodeldata.model2magcoords(xg,dat["ne"],interpsize[0]
                                                                 ,interpsize[1],interpsize[2])
nei=nei.reshape(interpsize)
alti,mloni,mlati,J1i=gemini3d.grid.gridmodeldata.model2magcoords(xg,dat["J1"],interpsize[0]
                                                                 ,interpsize[1],interpsize[2])
J1i=J1i.reshape(interpsize)


plt.subplots(2,1,dpi=150)

plt.subplot(2,1,1)
ialt=np.argmin(abs(alti-120e3))
plt.pcolormesh(mloni,mlati,nei[ialt,:,:].transpose())
plt.colorbar()
plt.xlabel("mlon")
plt.ylabel("mlat")
plt.title("$n_e$ @ 120 km altitude")

plt.subplot(2,1,2)
ialt=np.argmin(abs(alti-180e3))
plt.pcolormesh(mloni,mlati,J1i[ialt,:,:].transpose())
plt.colorbar()
plt.xlabel("mlon")
plt.ylabel("mlat")
plt.title("$J_1$ @ 180 km altitude")

