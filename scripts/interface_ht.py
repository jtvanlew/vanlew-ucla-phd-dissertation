E = 120e9
nu = .24
Ewall = 220e9
nuwall = 0.24
R = 0.0005
ks = 2.4
kw = 22

kstar = 2/(1/ks + 1/kw)
Rstar = R
Estar = 1/((1-nu**2)/E + (1-nuwall**2)/Ewall)


F = .0433
Hc1 = 2*kstar*(0.75*Rstar/Estar)**(1/3) * F**(1/3)

F = 2.6
Hc2 = 2*kstar*(0.75*Rstar/Estar)**(1/3) * F**(1/3)

deltaT = 10

q1 = Hc1*2.31*deltaT*1000**2
q2 = Hc2*2.31*deltaT*1000**2

print(q1, q2)