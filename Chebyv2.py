import numpy as np

def Chebyshev_domain(N):
	x = np.cos(np.pi * np.arange(N + 1) / N)
	return x
	
def Chebyshev_d1(N,x):
    c = np.ones(N + 1); c[0] = 2.; c[N] = 2.
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                if i == 0: D[i, j] = (2 * N**2 + 1) / 6.
                elif i == N: D[i, j] = -(2 * N**2 + 1) / 6.
                else: D[i, j] = -x[j] / (2 * (1 - x[j]**2))
            else:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i + j)) / (x[i] - x[j])
    return D

def Chebyshev_d2(N,x):
	C = np.ones(N + 1); C[0] = 2.; C[N] = 2.    
	D2 = np.zeros((N + 1, N + 1))
	for i in range(N+1):
		for j in range(N+1):
			if ( (i==0 and j==0) or (i==N and j==N) ):
				D2[i,j]=(N**4 - 1)/15.0
			elif ( (i==N) and (0 <= j <= N-1) ):
				D2[i,j]=((-1)**(N+j))*2*((2*N**2 + 1)*(1+x[j])- 6)/(3*C[j]*(1+x[j])**2)
			elif ( (i==0) and (1 <= j <= N) ):
				D2[i,j]=( ((-1)**j)*2*((2*N**2 + 1)*(1-x[j])- 6))/( 3*C[j]*(1-x[j])**2 )
			elif ( (i==j) and (1 <= j <= N-1 ) ):
				D2[i,j]=-((N**2-1)*(1-x[i]**2) + 3)/(3*((1-x[i]**2)**2))
			elif ( (i != j) and (1 <= i <= N-1) and (0 <= j <= N) ):
				D2[i,j]=(((-1)**(i+j))*(x[i]**2 +x[i]*x[j]-2))/(C[j]*(1-x[i]**2)*((x[i]-x[j])**2))
	
	return D2
		
