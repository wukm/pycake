# coding: utf-8
from hfft import discrete_gaussian_kernel
help(discrete_gaussian_kernel)
discrete_gaussian_kernel(100,1)
import matplotlib.pyplot as plt
np.arange(-25,25)
import numpy as np
np.linspace(-1,1,n=50)
np.linspace(-1,1,num=50)
dom = _
plt.plot(dom, discrete_gaussian_kernel(len(dom), 1))
plt.plot(dom, discrete_gaussian_kernel(len(dom), 1))
len(dom)
dgk = discrete_gaussian_kernel
dgk(50)
dgk(50,1)
_.shape
dgk(49,1)
_.shape
dgk(25,1)
dgk(25,1)
dom = np.linspace(-1,1,num=25)
plt.plot(dom,dgk(25,1))
plt.show()
plt.plot(dom,dgk(25,.5))
plt.show()
plt.plot(dom,dgk(25,.1))
plt.show()
plt.plot(dom,dgk(25,.05))
plt.show()
plt.plot(dom,dgk(25,.01))
plt.show()
plt.plot(dom,dgk(25,.001))
plt.show()
plt.plot(dom,dgk(25,.0001))
plt.show()
plt.plot(dom,dgk(25,0))
plt.show()
from scipy.signal import gaussian
get_ipython().run_line_magic('pinfo', 'gaussian')
gaussian(25,.1)
gaussian(25,.1)
plt.plot(dom, gaussian(25,.1))
plt.show()
plt.plot(dom, gaussian(25,.1), dom, dgk(25,.1))
plt.show()
plt.plot(dom, gaussian(25,10), dom, dgk(25,10))
plt.show()
plt.plot(dom, gaussian(25,10), dom, dgk(25,10))
plt.plot(dom, (1/np.sqrt(2*np.pi*10**2))*gaussian(25,10), dom, dgk(25,10))
plt.show()
plt.plot(dom, (1/np.sqrt(2*np.pi*10**2))*gaussian(25,10), dom, dgk(25,10**2))
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(25,10), dom, dgk(25,10**2))
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(100,10), dom, dgk(100,10**2))
dom = np.linspace(-1,1,num=100)
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*100**2)))*gaussian(100,10), dom, dgk(100,10**2))
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10**2))
dom = np.linspace(-1,1,num=101)
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10**2))
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10**2))
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10))
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10**2))
pt.show()
plt.show()
plt.plot(dom, np.sqrt((1/np.sqrt(2*np.pi*10**2)))*gaussian(101,10), dom, dgk(101,10))
plt.show()
dgk(5,1)
dgk(15,1)
dom = np.arange(15)
plt.imshow(dom, dgk(len(dom),1))
plt.imshow(dom, dgk(dom.size,1))
plt.imshow(dom, dgk(dom.size,1))
plt.imshow(dom, dgk(dom.size,1))
plt.imshow(dom, dgk(dom.size,1))
plt.imshow(dom, dgk(dom.size,1.))
dom = np.linspace(15)
dom = np.linspace(0,1,15)
dom
plt.imshow(dom, dgk(dom.size,1.))
plt.imshow(dom, dgk(15,1.))
get_ipython().run_line_magic('clear', '')
plt.imshow(dom, dgk(15,1.))
plt.plot(dom, dgk(15,1.))
plt.show()
plt.plot(dom, dgk(15,1.))
plt.show()
plt.plot(dom, dgk(15,.05))
plt.show()
plt.plot(dom, dgk(15,0))
plt.show()
plt.plot(dgk(100,0))
plt.show()
plt.scatter(dgk(100,0))
get_ipython().run_line_magic('pinfo', 'plt.scatter')
plt.plot(dgk(100,0), 'o')
plt.show()
plt.plot(dgk(100,0), 'o', markersize=1)
pt.show()
plt.show()
plt.plot(dgk(100,1), 'o', markersize=1)
plt.show()
plt.plot(dgk(100,15), 'o', markersize=1)
plt.show()
gaussian(100,15)
gaussian(100,15) / (15*np.sqrt(2*np.pi))
plot(gaussian(100,15) / (15*np.sqrt(2*np.pi)), 'or', markersize=1)
plt.plot(gaussian(100,15) / (15*np.sqrt(2*np.pi)), 'or', markersize=1)
plt.plot(dgk(100,15), 'o', markersize=1)
plt.show()
plt.plot(gaussian(105,15) / (15*np.sqrt(2*np.pi)), 'or', markersize=1)
plt.plot(dgk(105,15), 'o', markersize=1)
plt.show()
plt.plot(dgk(105,5), 'o-', markersize=1)
plt.plot(gaussian(105,5) / (5*np.sqrt(2*np.pi)), 'or', markersize=1)
plt.show()
plt.plot(gaussian(105,2) / (2*np.sqrt(2*np.pi)), 'or', markersize=1)
plt.plot(dgk(105,2), 'o-', markersize=1)
plt.show()
plt.plot(dgk(105,1.2), 'o-', markersize=1)
plt.show()
sigma = 2; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=1); plt.plot(dgk(105,2), 'o-', markersize=1)
plt.show()
sigma = 2; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,2), 'o-', markersize=1)
plt.show()
sigma = 1.2; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,2), 'o-', markersize=1)
plt.show()
sigma = 1.2; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
sigma = .8; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
sigma = .3; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
plt.show()
sigma = .01; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
sigma = .01; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
sigma = .10; plt.plot(gaussian(105,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(105,sigma), 'o-', markersize=1)
plt.show()
sigma = .10; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'ob-', markersize=1)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
np.arange(-(n_samples//2), (n_samples//2) + 1)
N=100; np.arange(-(N//2), (N//2) + 1)
N=100; np.arange(-N//2, (N//2) + 1)
N=100; np.arange((-N)//2, (N//2) + 1)
N=100; np.arange(-(N+1)//2, (N//2) + 1)
N=100; np.arange(-(N-1)//2, (N//2) + 1)
_.shape
N=100; np.arange(-(N-1)//2, (N//2))
N=100; np.arange(-(N+1)//2, (N//2))
N=100; np.arange(-(N)//2-1, (N//2))
N=100; np.arange(-((N)//2-1), (N//2))
N=100; np.arange(-((N)//2-1), (N//2)+1)
_.shape
N=101; np.arange(-((N)//2-1), (N//2)+1)
_.shape
from importlib import reload
reload(hfft)
import hfft
reload(hfft)
from hfft import discrete_gaussian_kernel as dgk
N=101; np.arange(-((N)//2-1), (N//2)+1)
dgk(50,.1)
_.shape
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
dgk(100,sigma).shape
gaussian(100,sigma).shape
dgk(100,sigma)
_.max()
gaussian(100,sigma).max()
gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi))
_.max()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=1)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=2)
plt.show()
sigma = .20; plt.plot(gaussian(100,sigma) / (sigma*np.sqrt(2*np.pi)), 'or', markersize=2); plt.plot(dgk(100,sigma), 'o-', markersize=2)
plt.show()
