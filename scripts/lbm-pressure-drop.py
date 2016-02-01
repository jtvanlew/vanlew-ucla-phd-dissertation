d_p = 0.001
u0lb = 0.02
dx = 20/240.
dt = dx*u0lb

pi = 605.888
po = 507.294

rho0 = 1
mu = 4.17e-5

dp =( 1e-4*(pi-po)/3) * (dx*2/dt*2)
print(dp)

L = dx * 263 * d_p #m
phi = 0.64

vs = 0.05

dpk = L*180*mu/d_p**2 * (phi**2)/(1-phi)**3 * vs

print( dpk)