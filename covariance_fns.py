# this file contains all necessary functions to generate covariance matrices for all relevant GP

import numpy

# Brownian Bridge with start and end point different from each other 
def gen_BB2_cov(nlen, fac=0.1):
	
	sigma_p = numpy.zeros((nlen, nlen))
	for i in range(nlen):
		for j in range(i):
			sigma_p[i, j] = (1 - i/nlen)*(1 - j/nlen) + (i*j)/(nlen*nlen) + (fac*fac)*(j - (i*j)/nlen)
			sigma_p[j, i] = sigma_p[i, j]
		sigma_p[i, i] = (1 - i/nlen)*(1 - i/nlen) + (i*i)/(nlen*nlen) + (fac*fac)*i*(1 - i/nlen)

	return sigma_p, numpy.linalg.det(sigma_p)

# Brownian Bridge with same start and end point
def gen_BB_cov(nlen, fac=0.11):
	
	sigma_p = numpy.zeros((nlen, nlen))
	for i in range(nlen):
		for j in range(i):
			sigma_p[i, j] = 1 + (fac*fac)*(j - (i*j)/nlen)
			sigma_p[j, i] = sigma_p[i, j]
		sigma_p[i, i] = 1 + (fac*fac)*i*(1 - i/nlen)

	return sigma_p, numpy.linalg.det(sigma_p)

# Ornstein–Uhlenbeck process
def gen_OU_cov(nlen, fac=0.1, alpha=1):
	
	b = (fac * fac) / (2 * alpha)

	sigma_p = numpy.zeros((nlen, nlen))
	for i in range(nlen):
		for j in range(i):
			sigma_p[i, j] = (1 + b * (numpy.exp(2 * alpha * j) - 1)) * numpy.exp(-alpha * (i + j))
			sigma_p[j, i] = sigma_p[i, j]
		sigma_p[i, i] = b + (1 - b) * numpy.exp(-2 * alpha * i)

	return sigma_p, numpy.linalg.det(sigma_p)

# Fractional Brownian Motion
def gen_frac_cov(H, nlen, fac=0.3):
	
	sigma_p = numpy.zeros((nlen, nlen))
	for i in range(nlen):
		for j in range(i):
			sigma_p[i, j] = 1 + (fac * fac) * 0.5 * (i**(2*H) + j**(2*H) - (i - j)**(2*H))
			sigma_p[j, i] = sigma_p[i, j]
		sigma_p[i, i] = 1 + (fac * fac) * i**(2*H)

	return sigma_p, numpy.linalg.det(sigma_p)

# Slow features (taken from paper: Disentangling Space and Time in Video with Hierarchical Variational Auto-encoders)
def gen_slowfeature(nlen, fac=0.1):
	
	sigma_p = numpy.ones((nlen, nlen))
	for i in range(1, nlen):
		sigma_p[i, i] += fac * fac
	return sigma_p, numpy.linalg.det(sigma_p)

def covariance_function(p_type, nlen, H = None):

	if (p_type == 'frac'):
		if (H == None):	
			raise Exception("Invalid Argument H")
		return gen_frac_cov(H, nlen)

	elif (p_type == 'bb'):
		return gen_BB_cov(nlen)

	elif(p_type == 'bb2'):
		return gen_BB2_cov(nlen)

	elif(p_type == 'ou'):
		return gen_OU_cov(nlen)

	elif(p_type == 'slow'):
		return gen_slowfeature(nlen)
		
	else:
		raise Exception('Gaussian Process {}, H = {} not implemented'.format(p_type, H))

if __name__ == "__main__":
	s, d = gen_frac_cov(0.1, 5)
	#s, d = gen_slowfeature(8)
	#s, d = gen_OU_cov(5)
	#s, d = gen_BB_cov(5)
	#s, d = gen_BB2_cov(5)

	print(s)
	print(d)