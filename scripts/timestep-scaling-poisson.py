import numpy as np

nu = np.linspace(0,1,100)

y = np.sqrt(2*(1+nu))/(-.1631 * nu + 0.876605)

import matplotlib.pyplot as plt 

plt.plot(nu,y)
plt.show()