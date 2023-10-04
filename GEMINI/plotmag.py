#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:19:46 2023

@author: zettergm
"""

import gemini3d.magtools
import matplotlib.pyplot as plt

magdat=gemini3d.magtools.magframe("/Users/zettergm/simulations/ssd/aurora_EISCAT3D/magfields/20160303_15310.000000.h5")

plt.subplots(3,1,dpi=150)
plt.subplot(3,1,1)
plt.pcolormesh(magdat["mlon"],magdat["mlat"],magdat["Br"][0,:,:]*1e9)
plt.colorbar()
plt.xlabel("mlon")
plt.ylabel("mlat")
plt.title("$B_r$ (nT)")

plt.subplot(3,1,2)
plt.pcolormesh(magdat["mlon"],magdat["mlat"],magdat["Btheta"][0,:,:]*1e9)
plt.colorbar()
plt.xlabel("mlon")
plt.ylabel("mlat")
plt.title("$B_{\\theta}$ (nT)")

plt.subplot(3,1,3)
plt.pcolormesh(magdat["mlon"],magdat["mlat"],magdat["Bphi"][0,:,:]*1e9)
plt.colorbar()
plt.xlabel("mlon")
plt.ylabel("mlat")
plt.title("$B_\\phi$ (nT)")
