import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
import numdifftools as ndt
from scipy.optimize import minimize
from scipy import optimize
import pylab as py

#Define variables
f = open('1H0323+342_results(flare3,1,free).txt', 'r')
lines =f.readlines()[30:55]
f.close()
x1 = []
y1 = []
z1 = []
for line in lines:
    p = line.split()
    x1.append(float(p[0]))
    y1.append(float(p[2]))
    z1.append(float(p[3]))
xdata = np.array(x1)
xdata[:] = [(x-393172518.469)/86400 for x in xdata]
ydata = np.array(y1)
ydata[:] = [y*10**7 for y in ydata]
zdata = np.array(z1)
zdata[:] = [z*10**7 for z in zdata]

#Define the parameters and function
def func(params):
    A0 = params[0]
    A1 = params[1]   
    T1 = params[2]
    trise1 = params[3]
    tdecay1 = params[4]   

    yPred = A0+A1*(np.exp((xdata-T1)/trise1)+np.exp((T1-xdata)/tdecay1))**(-1)

# Calculate negative log likelihood
#scale=zdata
    LL = -np.sum( stats.norm.logpdf(ydata, loc=yPred) )

    return(LL)

#Fit and print results
initParams = (5,4,70,2,2)
bnds = [(1,20),(1,20),(60,80),(0.001,30),(0.001,30)]
results =  (func,initParams, method='SLSQP', bounds=bnds, tol=1e-8, options={'disp': True})
print results

#Define yOut function for fitted parameters
estParms = results.x
xarray = np.linspace(0,150,800)
yOut = yPred = estParms[0]+estParms[1]*(np.exp((xarray-estParms[2])/estParms[3])+np.exp((estParms[2]-xarray)/estParms[4]))**(-1)

#Plot output
py.clf()
py.plot(xdata,ydata, 'go')
py.plot(xarray, yOut)
py.show()


#Calculate Hessian using numdifftools and pandas packages
Hfun = ndt.Hessian(func, full_output=True)
hessian_ndt, info = Hfun(results['x'])
se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
results = pd.DataFrame({'parameters':results['x'],'std err':se})
results.index=['A0','A1','T1','trise','tdecay1']   
results.head()
print results.head()





