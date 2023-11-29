import numpy as np

class Vertex:
	"""A class used to represent a vertex.
	
	Attributes
	----------
	v : numpy.array
		an array of the coordinates of the vertex
	"""

	def __init__(self, lst:list):
		"""
		Parameters
		----------
		lst: list
			a list of the coordinates of the vertex
		"""

		self.v = np.array(lst)

def subVertex(v1:Vertex, v2:Vertex):
	"""Takes in two vertices, subtracts the second vertex from the first one. """

	return Vertex(v1.v-v2.v) 

def addVertex(v1:Vertex, v2:Vertex):
	"""Takes in two vertices, returns the sum (a vertex)."""

	return Vertex(v1.v+v2.v)

def lambdaVertex(c, ver:Vertex):
	"""Takes in a constant and a vertex, returns the product (a vertex)."""

	return Vertex(c*ver.v)

def negVertex(ver:Vertex):
	"""Takes in a vertex, returns its reflection about the origin (a vertex). """

	return Vertex(-ver.v)

def dot(v1:Vertex, v2:Vertex):
	"""Takes in two vertices and retunrs the dot product. """

	return v1.v.dot(v2.v)

def norm(ver:Vertex):
	"""Takes in a vertex and returns its norm. """

	return np.sqrt(ver.v.dot(ver.v))

def xangle(p1:Vertex, p2:Vertex):
	"""Takes in two vertices, p1 and p2. Returns the polar angle of the vector from p1 to p2.""" 
	
	d = norm(Vertex(p1.v-p2.v))
	
	if d>0:
		theta = (p2.v[0]-p1.v[0])/d
		theta = np.arccos(theta)
		theta = theta+2*(np.pi-theta)*(p2.v[1]<p1.v[1])
	else:
		theta = 0
	return theta

def fetchY(ver:Vertex):
	"""Returns the 2nd coordinate of a vertex. """

	return ver.v[1]

def fetchX(ver:Vertex):
	"""Returns the 1st coordinate of a vertex."""

	return ver.v[0]

def distVertex(v1:Vertex, v2:Vertex):
	"""Returns the distance between two vertices. """

	v = subVertex(v1, v2)
	return(max(list(map(abs,v.v))))