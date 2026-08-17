#Import Cheby.py, it must be in the same directory with this file
from Chebyv2 import Chebyshev_domain
from Chebyv2 import Chebyshev_d2
from Chebyv2 import Chebyshev_d1
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import scipy.linalg as la
import csv


#Number of collocation points
N=200

#Define Chebyshev matrix, first and second derivative
x=Chebyshev_domain(N-1)
D=Chebyshev_d1(N-1,x)
D2=Chebyshev_d2(N-1,x)

#parameter of Lorentz force
Q=2290.0 

#Tau_inv
L=7.68	

#Tau
t=1.0/L

#Magnefic field profile
A=1.85 

sqt=np.sqrt(t)
#u- from analytical solution
def um(x):
	w=(Q*(t/2))*(np.exp(-x/sqt)*sp.erf(A*x) - np.exp(1.0/(4.0*A*A*t))*sp.erf(1.0/(2.0*A*sqt) + A*x))
	return w

#u+ from analytical solution	
def up(x):
	w=(Q*(t/2))*(np.exp(x/sqt)*sp.erf(A*x) + np.exp(1.0/(4.0*A*A*t))*sp.erf(1.0/(2.0*A*sqt) - A*x))
	return w

def ump(x):
	w=(-0.5)*np.exp(-x/sqt)*Q*sqt*sp.erf(A*x)
	return w
	
def upp(x):
	w=(0.5)*np.exp(x/sqt)*Q*sqt*sp.erf(A*x)
	return w
	

#Dummy constants
Ap=ump(1)
AA=um(1)
Bp=upp(1)
BB=up(1)

Cp=ump(-1)
CC=um(-1)
Dp=upp(-1)
DD=up(-1)

	
#constants from analytical solution	
c1=( CC -np.exp(4/sqt)*(AA+Ap*sqt) + np.exp(2/sqt)*( BB-DD + (-Bp+Dp)*sqt) + Cp*sqt )/(-1+np.exp(4/sqt))
c2=( BB +np.exp(4/sqt)*(-DD+Dp*sqt) + np.exp(2/sqt)*(-AA+CC + (-Ap+Cp)*sqt) -Bp*sqt )/(-1+np.exp(4/sqt))



#Analytical solution of unperturbed main velocity
def u(x):
	w=(c1+um(x))*np.exp(x/sqt) + (c2+up(x))*np.exp(-x/sqt)
	return w

#second derivative of unperturbed main velocity	
def d2udx2(x):
		return u(x)/t-Q*sp.erf(A*x)

def build_system(k):

	#Build the generalized eigenvalue problem A*phi=c*B*phi
	
	I=np.eye(N)
	Lp=D2-k*k*I
	Lp_inv=la.inv(Lp)
	upp=np.vectorize(d2udx2)(x)
	
	#Define M matrix, M=U - Upp*Linv - (1/i*k)*I*L + (1/i*k*Tau)*I
	A=np.diag(u(x)) - np.diag(upp)@Lp_inv - (1.0/(1j*k))*I@Lp + (1.0/(1j*k*t))*I
	B=I

#	A=np.diag(u(x))@Lp - np.diag(upp)
#	B=Lp
	
	# Boundary conditions, the D2 matrix is considered for free-slip boundary conditions
	row=0
	A[row, :] = 0; B[row, :] = 0
	A[row, 0] = 1.0

	row = 1
	A[row, :] = 0; B[row, :] = 0
	A[row, 0:N] = D2[0,:]

	row = N-1
	A[row, :] = 0; B[row, :] = 0
	A[row, N-1] = 1.0
	
	row = N-2
	A[row, :] = 0; B[row, :] = 0
	A[row, 0:N] = D2[N-1,:]
     
	return A,B

#Calculate w=kci, ci=Img(c) for each case
def Wval(k):	
	A,B=build_system(k)
	vals,vecs=la.eig(A,B)
	gamma=k*np.imag(vals)
	gamma_max=np.max(gamma)
	idx_inestable = np.argmax(gamma)
	c_inestable = vals[idx_inestable]
	return k*np.imag(c_inestable)

#Number of test "k" values for dispersion relation			
Np=60
kvals=np.linspace(0.4,0.8,Np)
#kvals=np.linspace(1.0e-3,1.5,Np)
Wvals=np.zeros(Np)

for i in range(0,Np):
	Wvals[i]=Wval(kvals[i])


# Found the maximum
idx = np.argmax(Wvals)
kmax = kvals[idx]
Wmax = Wvals[idx]

# Plot results
plt.figure(figsize=(8,5))
plt.title('Dispersion relation')
plt.ylabel('ω=kci')
plt.xlabel('k ')
plt.plot(kvals,Wvals,'-o',label='Spectral Cheabyshev',color='orange')
plt.axvline(kmax, color='g', linestyle='--', label=f'k_max={kmax:.8f}')
plt.axhline(Wmax, color='r', linestyle=':', label=f'ω_max={Wmax:.6f}')
plt.legend()
plt.grid(True)
plt.show()
