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
N=170

#Define Chebyshev matrix, first and second derivative
x=Chebyshev_domain(N-1)
D=Chebyshev_d1(N-1,x)
D2=Chebyshev_d2(N-1,x)

#parameter of Lorentz force
Q=950.0 

#Tau_inv
L=7.68	

#Tau
t=1.0/L

#Magnefic field profile
A=1.85 

#u1 from analytical solution
def u1(x):
	w=(-Q*t/2)*( -np.exp(-x/np.sqrt(t))*sp.erf(A*x) + np.exp(1/(4*A*A*t))*sp.erf( 1/(2*A*np.sqrt(t)) + A*x) )
	return w

#u2 from analytical solution	
def u2(x):
	w=(Q*t/2)*( np.exp(x/np.sqrt(t))*sp.erf(A*x) + np.exp(1/(4*A*A*t))*sp.erf( 1/(2*A*np.sqrt(t)) - A*x) )
	return w

#constants from analytical solution	
c2=( u1(1)-u1(-1)+u2(1)*np.exp(-2/np.sqrt(t))-u2(-1)*np.exp(2/(np.sqrt(t)) ) )/(2.0*np.sinh(2/np.sqrt(t)))
c1=-c2*np.exp(-2/np.sqrt(t))-u1(1)-u2(1)*np.exp(-2/np.sqrt(t))

#Analytical solution of unperturbed main velocity
def u(x):
	w=(c1+u1(x))*np.exp(x/np.sqrt(t)) + (c2+u2(x))*np.exp(-x/np.sqrt(t))
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

	# Boundary conditions
	row=0
	A[row, :] = 0; B[row, :] = 0
	A[row, 0] = 1.0

	row = 1
	A[row, :] = 0; B[row, :] = 0
	A[row, 0:N] = D[0,:]

	row = N-1
	A[row, :] = 0; B[row, :] = 0
	A[row, N-1] = 1.0
	
	row = N-2
	A[row, :] = 0; B[row, :] = 0
	A[row, 0:N] = D[N-1,:]
     
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
kvals=np.linspace(0.8,1.8,Np)
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
