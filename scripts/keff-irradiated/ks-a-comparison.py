import numpy as np
import matplotlib.pyplot as plt

nu = 0.24
E = 60.e9
Estar = 1/(2*((1-nu**2)/E))
Rstar = 0.00025
Fn = 38.79
Fn1third = Fn**(1/3.)

ks = 2.4*0.01
a = (0.75*Rstar/Estar)**(1/3.)*Fn**(1./3)
Fn = (ks/(2*(0.75*Rstar/Estar)**(1/3.)))**(3.)