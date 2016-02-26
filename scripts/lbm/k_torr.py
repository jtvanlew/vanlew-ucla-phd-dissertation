ks = 2.4
kf = 0.34
kappa = ks/kf
phi = 0.64
k_torr = kf * (kappa -1)*( phi / (phi**0.5 + kappa*(1-phi**0.5)) )
print(k_torr)