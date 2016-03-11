d_p = 0.001
u0lb = 0.02
dx = 20/240.
dt = dx*u0lb

pi = 1.87
po = 1.714

rho0 = 1
mu = 4.17e-6

dp =( (pi-po)/3) * (dx**2/dt**2)*3
print(dp)

L = dx * 263 * d_p #m
phi = 0.64

vs = 0.05

dpk = 180.*mu/d_p**2 * (phi**2)/(1-phi)**3 * vs

print( dpk)	