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

#Build the generalized eigenvalue problem A*phi=c*B*phi depending on k,tau,A,Q
def build_system(k,t,A,Q):

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
	
	#Build the generalized eigenvalue problem A*phi=c*B*phi
	
	I=np.eye(N)
	Lp=D2-k*k*I
	Lp_inv=la.inv(Lp)
	upp=np.vectorize(d2udx2)(x)
	matA=np.diag(u(x)) - np.diag(upp)@Lp_inv - (1.0/(1j*k))*I@Lp + (1.0/(1j*k*t))*I
	matB=I
	
	
	# Boundary conditions
	row=0
	matA[row, :] = 0; matB[row, :] = 0
	matA[row, 0] = 1.0
	
	row = 1
	matA[row, :] = 0; matB[row, :] = 0
	matA[row, 0:N] = D[0,:]

	row = N-1
	matA[row, :] = 0; matB[row, :] = 0
	matA[row, N-1] = 1.0
	
	row = N-2
	matA[row, :] = 0; matB[row, :] = 0
	matA[row, 0:N] = D[N-1,:]
     
    
	return matA,matB

#Calculate w=kci, ci=Img(c) for each case
def Wval(k,t_,A_,Q_):	
	A,B=build_system(k,t_,A_,Q_)
	vals,vecs=la.eig(A,B)
	gamma=k*np.imag(vals)
	gamma_max=np.max(gamma)
	idx_inestable = np.argmax(gamma)
	c_inestable = vals[idx_inestable]
	return k*np.imag(c_inestable)

#Number of test "k" values for dispersion relation, returns the max(Wvals)
def Wmax(t_,A_,Q_):	
			
	Np=60
	kvals=np.linspace(0.75,1.75,Np)
	Wvals=np.zeros(Np)

	for i in range(0,Np):
		Wvals[i]=Wval(kvals[i],t_,A_,Q_)

	# Found maximum
	idx = np.argmax(Wvals)
	kmax = kvals[idx]
	Wmax = Wvals[idx]
	return Wmax


#Testing points of tau_inv and A
Lvals=np.linspace(0.01,20.0,10)
Avals=np.linspace(1.0,2.0,10)

#Build temporal data array that saves [wmax,tau_inv,A,Q] for each case 
data=[]

for A in Avals:
	for L in Lvals:
	
		#Initialize Q2 and Q1 to find roots 
		# by a modified Newton's method [Hoffman,Numerical Methods for Engineers and Scientists, 2a Ed.,2001,pag.146] 
		
		#Large Q2 as initial point
		Q2=10000.0
		wmax2=Wmax(1.0/L , A, Q2 )
		
		Q1=Q2-0.01
		wmax1=Wmax(1.0/L , A, Q1 )
		while(True):
			
			gp=(wmax2-wmax1)/(Q2-Q1) 
			Qnew = Q1 - wmax1/gp
			wmaxnew=Wmax(1.0/L , A, Qnew )
			
			while(wmaxnew==0.0):
				Qnew=0.5*(Q2+Qnew)
				wmaxnew=Wmax(1.0/L , A, Qnew )
						
			if( (0.0< wmaxnew < 0.1) ):
				print("	Found:",wmaxnew,1.0/L,A,Qnew)
				data.extend([[wmaxnew,1.0/L,A,Qnew]])
				break
			
			print(" Searching... ",wmaxnew,1.0/L,A,Qnew)
			
			
			
			wmax1=wmax2
			wmax2=wmaxnew
			
			Q1=Q2
			Q2=Qnew
			
# Save temporal data array in file "Qcrit_data.csv"				
with open("Qcrit_data.csv","w",newline="") as archivo:
	writer=csv.writer(archivo)
	writer.writerow(["# wmax","tau","A","Qcrit"])
	for i in range(0,int(len(data))):
		writer.writerow([ data[i][0],data[i][1],data[i][2],data[i][3] ])
