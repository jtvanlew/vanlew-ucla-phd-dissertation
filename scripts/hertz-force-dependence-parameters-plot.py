# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:39:21 2014

@author: jon
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1,1.e6,1001)

y1 = x**(1./6)
y2 = x**(1./3)
y3 = x**(1./2)
y4 = x**(5./4)
plt.loglog(x,y1, x,y2, x,y3, x,y4)
plt.ylim([10**0,10**1])
plt.legend([r'$F_{max}$',r'$E^*$',r'$\rho$',r'$R_0$'],loc='lower right')
plt.grid()
plt.grid(b=True, which='minor')
plt.show()
