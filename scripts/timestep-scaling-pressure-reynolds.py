import numpy as np
import matplotlib.pyplot as plt 

E     = 900.e9      # dyne/cm2
nu    = 0.24
rho   = 2.260    # g/cm3 note: it waas 12 for strain_edit 1-6
Rp    = 0.05   # cm

pressVels  = np.array([1.e-2, 1.e-3, 1.e-4, 5.e-5])  # cm / s
deltaT = 1.e-6
straindot = pressVels/deltaT

Re = np.sqrt(straindot**2 * rho * Rp**2/E)

P* = np.array([
	])/E

for i in np.arange(len(Re)):
	print Re[i]