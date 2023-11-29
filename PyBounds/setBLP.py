#########################
### Imported packages ###
#########################
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .vertex import *
# Class: Vertex
# Functions: subVertex, addVertex, lambdaVertex, negVertex, dot, norm, xangle, fetchY, fetchX, distVertex

from .segment import * 
# Class: Segment
##-methods: checkInput, length, dim
# Functions: dotDist

from .polygons import * 
# Class: Polygon
##-method: sort, angles, plot, scatter
# Functions: minkowskiSum, lambdaPolygon, dHausdorff, hausdorff

###############################
###   Defined Structures:   ###
###############################  

class Options:
	"""A class used to represnt options for Monte Carlo iterations
	
	Attributes
	----------
	MC_iterations: int
		number of iterations
	seed : int
		the random seed
	rng : numpy.random._generator.Generator
		the random number generator 
	conf_level: float
		the confidence level
	tol : float, optional
		the tolerance (default is 1e-6)
	"""

	def __init__(self, MC_iterations, seed, rng, conf_level, tol = 1e-6):
		"""
		Parameters
		----------
		MC_iterations: int
			number of iterations
		seed : int
			the random seed
		rng : numpy.random._generator.Generator
			the random number generator 
		conf_level: float
			the confidence level
		tol : float, optional
			the tolerance (default is 1e-6)
		"""

		self.MC_iterations = MC_iterations
		self.seed = seed
		self.rng = rng
		self.conf_level = conf_level
		self.tol = tol
		
class TestResults:
	"""A class used to represnt the results of a statistical test, including confidence intervals, critical values, and test statistics.

	
	Attributes
	----------
	testStat: float
		test statistics
	criticalVal : float
		the critical value
	ConfidenceInterval: list
		a list of the lower and upper bounds of the condidence interval
	"""

	def __init__(self, testStat = None, criticalVal= None, ConfidenceInterval= None):
		"""A class used to represnt test results 
		
		Attributes
		----------
		testStat: float, optional
			test statistics (default is None)
		criticalVal : float, optional
			the critical value (default is None)
		ConfidenceInterval: list, optional
			a list of the lower and upper bounds of the condidence interval (default is None)
		"""

		self.testStat = testStat 
		self.criticalVal = criticalVal
		self.ConfidenceInterval = ConfidenceInterval
		
class Results:
	"""A class used to represnt the results of a statistical analysis, including bounds, null values, hypothesis test results, and derivative hypothesis test results.
	
	Attributes
	----------
	bound: list
		a list of real numbers representing the bounds
	null: list, optional
		a list of real numbers representing null values (default is None)
	Htest : TestResults, optional
		an instance of `TestResults` representing the results of a hypothesis test (default is None)
	dHtest: TestResults, optional
		An instance of `TestResults` representing the results of a derivative hypothesis test (default is None)
	"""

	def __init__(self, bound, null = None, Htest = None, dHtest = None):
		"""
		Parameters
		----------
		bound: list
			a list of real numbers representing the bounds
		null: list, optional
			a list of real numbers representing null values (default is None)
		Htest : TestResults, optional
			an instance of `TestResults` representing the results of a hypothesis test (default is None)
		dHtest: TestResults, optional
			An instance of `TestResults` representing the results of a derivative hypothesis test (default is None)
		"""

		self.bound = bound
		self.null = null
		self.Htest = Htest
		self.dHtest = dHtest  

#####################
###   Constants   ###
#####################

default_options = Options(2000, 15217, np.random.MT19937, 0.95)

#####################
###  Functions:   ###
#####################

def plus(x):
	return max(0.0, x)

def minus(x):
	return max(0.0, -x)

def HdistInterval(v1:list, v2:list):
	v = np.array(v1) - np.array(v2)
	return max(abs(v))

def dHdistInterval(v1:list, v2:list):
	v = np.array(v1) - np.array(v2)
	return max(plus(v[0]), minus(v[1]))

## Plan: add DataFrame capabilities
def EY(yl:list,yu:list,H0:list,options:Options=default_options,method="Asymptotic"):
	"""THis is the shell function that calls either the asymptotic distribution version or the bootstrap version of EY."""

	if method =="Asymptotic":
		return EYasy(yl,yu,H0,options)
	else:
		return EYboot(yl,yu,H0,options)

def EYboot(yl:list, yu:list, H0:list, options:Options=default_options):
	"""This function uses a bootstrap test. This option is not in BM(2008) for EY but it is proved for BLP in section 4"""

	LB = np.mean(yl)
	UB = np.mean(yu)
	bound = [LB, UB]
	
	# test Statistic
	n = len(yl)
	sqrt_n = np.sqrt(n)
	TestStat_H = sqrt_n*HdistInterval(bound, H0)
	TestStat_dH = sqrt_n*dHdistInterval(bound, H0)
	
	B = options.MC_iterations #number of MC iterations to compute the critical value
	alpha = options.conf_level #confidence level for the critical value1
	
	r_H = []
	r_dH = []
	rng = np.random.Generator(options.rng(options.seed))

	for i in range(B):
		indx = rng.integers(low=0, high=n, size=n)
		yl_b = yl[indx]
		yu_b = yu[indx]
		bound_b = [np.mean(yl_b), np.mean(yu_b)]
		r_H.append(sqrt_n*HdistInterval(bound_b, bound))
		r_dH.append(sqrt_n*dHdistInterval(bound_b, bound))
	
	r_H.sort()
	c_H = r_H[np.floor(alpha*B).astype(int)]
	CI_H = [LB - c_H/sqrt_n, UB+c_H/sqrt_n]
	Htest = TestResults(TestStat_H,c_H,CI_H) 
	
	r_dH.sort()
	c_dH = r_dH[np.floor(alpha*B).astype(int)]
	CI_dH = [LB - c_dH/sqrt_n, UB + c_dH/sqrt_n]
	dHtest = TestResults(TestStat_dH,c_dH,CI_dH)
	   
	results = Results(bound, H0, Htest, dHtest)
	return results            

def EYasy(yl:list, yu:list, H0:list, options:Options=default_options):
	"""This function uses the test based on the asymptotic distributin as developed in BM(2008) pp. 778-779"""

	LB = np.mean(yl)
	UB = np.mean(yu)
	bound = [LB, UB]
	
	# test Statistic
	n = len(yl)
	sqrt_n = np.sqrt(n)
	TestStat_H = sqrt_n*HdistInterval(bound, H0)
	TestStat_dH = sqrt_n*dHdistInterval(bound, H0)
	
	#Simulating the asy. distribution using a MC method to establish a critical value (quantile):
	
	# critical value based on Hausdorff distance
	Pi = np.cov(yl, yu) #covariance matrix for yl yu
	
	B = options.MC_iterations #number of MC iterations to compute the critical value
	alpha = options.conf_level #confidence level for the critical value
	
	## Following Algorithm on page 780 in BM2008:
	rng = np.random.Generator(default_options.rng(default_options.seed))
	rr = rng.multivariate_normal([0,0], Pi, B) #drawing B pairs from a bivariate-normal distribution.
	
	## test based on Hausdorff distance:
	r_H = np.amax(abs(rr),axis=1) #row max
	r_H.sort()
	c_H = r_H[np.floor(alpha*B).astype(int)]
	CI_H = [LB - c_H/sqrt_n, UB+c_H/sqrt_n]
	Htest = TestResults(TestStat_H,c_H,CI_H) 
	
	## test based on directed Hausdorff distance:
	r_dH = np.amax(np.array([list(map(plus,rr[:,0])), list(map(minus,rr[:,1]))]), axis = 0)
	r_dH.sort()
	c_dH = r_dH[np.floor(alpha*B).astype(int)]
	CI_dH = [LB - c_dH/sqrt_n, UB + c_dH/sqrt_n]
	dHtest = TestResults(TestStat_dH,c_dH,CI_dH)
	
	results = Results(bound, H0, Htest, dHtest)
	return results

def oneDproj_single(yl:list, yu:list, x:list):
	# yl, yu, and x are lists 
	x = np.array(x)-np.mean(x)
	M1 = np.multiply(x, yl)
	M2 = np.multiply(x, yu)
	s = np.dot(x,x)
	bound = [sum(np.minimum(M1, M2))/s, sum(np.maximum(M1, M2))/s]
	return bound

def oneDproj_multi(yl:list, yu:list, x, cord = None):
	# yl(yu) is a list 
	# x is a list (single variable), or matrix (multiple variable)
	
	if type(x) == list:
		bound = oneDproj_single(yl, yu, x)
	elif len(x.shape) == 1:
		bound = oneDproj_single(yl, yu, x)
	else: 
		# add constant to x, constant is in column 0
		x = sm.add_constant(x)
		if cord is not None: 
			# update cord after adding constant
			cord += 1

			# linear projection of x_cord on x_(-cord)
			r = sm.OLS(x.iloc[:, cord], x.drop(x.columns[cord], axis = 1)).fit().resid
			bound = oneDproj_single(yl, yu, r)
		else:
			bound = []
			for j in range(x.shape[1]-1):
				j+=1
				r = sm.OLS(x.iloc[:, j], x.drop(x.columns[j], axis = 1)).fit().resid
				bound.append(oneDproj_single(yl, yu, r))
	return bound

def oneDproj(yl, yu, x, cord = None, data = None, CI = True, H0 = None, options=default_options):
	"""Computes the 1D projection of the identification set on a specific dimension of the explanatory variable
	
	Parameters:
	-----------
	yl: list or str
		the yl variable, or its variable name in data if the dataframe data is provided
	yu: list or str
		the yu variable, or its variable name in data if the dataframe data is provided	
	x: list, pandas.DataFrame, or str
		the x variable (a single variable), 
		or its variable name in data if the dataframe data is provided, 
		or the list of names of multiple x variables in data if the dataframe data is provided,
		or the dataframe containing multiple x variables
	cord: int or list, optional
		the (list of) coordinates of interest (default is None)
	data: pandas.DataFrame, optional
		the dataframe (default is None)
	CI: bool, optional
		report the Confidence Interval if True (default is True)
	H0: float or list, optional
		the (list of) null hypothesis values (default is None)
	options: Options, optional
		an instance of `Options` (default is default_options)
	"""
	
	length_cord = 0 # if x is a vector and cord is None
	if cord is not None:
		if type(cord)==int:
			length_cord = 1
		else:
			length_cord = len(cord)

	if type(yl) == str:
		if data is not None:
			yl = data[yl]
		else:
			raise Exception(f"Please provide data that contains {yl}")
	if type(yu) == str:
		if data is not None:
			yu = data[yu]
		else:
			raise Exception(f"Please provide data that contains {yu}")
	
	if type(x) == str: 
	# if x is a variable name
		if data is not None:
			x = data[x]
			bound = oneDproj_single(yl, yu, x)
		else:
			raise Exception(f"Please provide data that contains {x}")
	elif type(x) == list and type(x[0]) == str : 
	# if x is a list of variable names
		if data is None:
			raise Exception(f"Please provide data that contains {x}")
		else: 
			x = data[x]
			bound = oneDproj_multi(yl, yu, x, cord)  
			if cord is None:
				length_cord = x.shape[1]
				cord = range(length_cord)           
	else: 
	# if x is a vector or matrix
		if len(x.shape)==1: 
		#vector
			bound = oneDproj_single(yl, yu, x)
		else: 
		#matrix
			bound = oneDproj_multi(yl, yu, x, cord)
			if cord is None:
				length_cord = x.shape[1] 
				cord = range(length_cord)           

	if CI == False:
		return bound
	else:
		if length_cord == 0 or 1: # if bound contains only one interval
			LB = bound[0]
			UB = bound[1]
		else:
			LB = [bound[i][0] for i in range(length_cord)]
			UB = [bound[i][1] for i in range(length_cord)]
	
		n = len(yl)
		sqrt_n = np.sqrt(n)

		if H0 is None:
			testStat_H = None
			testStat_dH = None
		else:
			testStat_H = sqrt_n*HdistInterval(bound,H0)
			testStat_dH = sqrt_n*dHdistInterval(bound,H0)

		B = options.MC_iterations #number of MC iterations to compute the critical value
		alpha = options.conf_level  #confidence level for the critical value1

		rng = np.random.Generator(options.rng(options.seed))

		if length_cord>1:
			results =dict()
			r_H = [ [ None for i in range(B) ] for j in range(length_cord) ]
			r_dH = [ [ None for i in range(B) ] for j in range(length_cord) ]
			for i in range(B):
				indx = rng.integers(low=0, high=n, size=n)
				yl_b = yl[indx]
				yu_b = yu[indx]
				x_b = x.iloc[indx,cord]

				bound_b = oneDproj_multi(yl_b,yu_b,x_b)
				for j in range(length_cord):
					r_H[j][i]=sqrt_n*HdistInterval(bound_b[j], bound[j])
					r_dH[j][i]=sqrt_n*dHdistInterval(bound_b[j], bound[j])

			for j in range(length_cord):
				r_H[j].sort()
				c_H = r_H[j][np.floor(alpha*B).astype(int)]
				CI_H = [LB[j] - c_H/sqrt_n, UB[j]+c_H/sqrt_n]
				Htest = TestResults(testStat_H,c_H,CI_H) 

				r_dH[j].sort()
				c_dH = r_dH[j][np.floor(alpha*B).astype(int)]
				CI_dH = [LB[j] - c_dH/sqrt_n, UB[j]+c_dH/sqrt_n]
				dHtest = TestResults(testStat_dH,c_dH,CI_dH)

				if H0 is None:
					results[j]=Results(bound[j],H0, Htest, dHtest) 
				elif len(H0)!=length_cord: 
					raise Exception(f"Please provide {length_cord} null values")
				else:
					results[j]=Results(bound[j],H0[j], Htest, dHtest) 
		else:
			if length_cord ==1:
				x = x.iloc[:,cord] 

			r_H = []
			r_dH = []

			for i in range(B):
				indx = rng.integers(low=0, high=n, size=n)
				yl_b = yl[indx]
				yu_b = yu[indx]
				x_b = x[indx]

				bound_b = oneDproj_single(yl_b,yu_b,x_b)
				r_H.append(sqrt_n*HdistInterval(bound_b, bound))
				r_dH.append(sqrt_n*dHdistInterval(bound_b, bound))

			r_H.sort()
			c_H = r_H[np.floor(alpha*B).astype(int)]
			CI_H = [LB - c_H/sqrt_n, UB+c_H/sqrt_n]
			Htest = TestResults(testStat_H,c_H,CI_H) 

			r_dH.sort()
			c_dH = r_dH[np.floor(alpha*B).astype(int)]
			CI_dH = [LB - c_dH/sqrt_n, UB + c_dH/sqrt_n]
			dHtest = TestResults(testStat_dH,c_dH,CI_dH)

			results = Results(bound, H0, Htest, dHtest)
		
		return results

def CI1d(yl:list, yu:list, x:list, H0:list, options:Options=default_options):
	"""Computes the 1D projection of the identification set on a specific dimesion of the explanatory variable."""
	
	#step 1: Compute the formula on page 787 in BM2008
	bound = oneDproj(yl,yu,x)
	LB = bound[0]
	UB = bound[1]
	
	#step2: compute the test statistics
	n = len(yl)
	sqrt_n = np.sqrt(n)
	TestStat_H = sqrt_n*HdistInterval(bound,H0)
	TestStat_dH = sqrt_n*dHdistInterval(bound,H0)


	B = options.MC_iterations #number of MC iterations to compute the critical value
	alpha = options.conf_level  #confidence level for the critical value1

	r_H = []
	r_dH = []
	rng = np.random.Generator(options.rng(options.seed))
	
	for i in range(B):
		indx = rng.integers(low=0, high=n, size=n)
		yl_b = yl[indx]
		yu_b = yu[indx]
		x_b = x[indx]
		bound_b = oneDproj(yl_b,yu_b,x_b)
		r_H.append(sqrt_n*HdistInterval(bound_b, bound))
		r_dH.append(sqrt_n*dHdistInterval(bound_b, bound))
	
	r_H.sort()
	c_H = r_H[np.floor(alpha*B).astype(int)]
	CI_H = [LB - c_H/sqrt_n, UB+c_H/sqrt_n]
	Htest = TestResults(TestStat_H,c_H,CI_H) 
	
	r_dH.sort()
	c_dH = r_dH[np.floor(alpha*B).astype(int)]
	CI_dH = [LB - c_dH/sqrt_n, UB + c_dH/sqrt_n]
	dHtest = TestResults(TestStat_dH,c_dH,CI_dH)
	   
	results = Results(bound, H0, Htest, dHtest)
	return results