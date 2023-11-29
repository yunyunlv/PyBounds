from .vertex import *

class Segment:
	""" A class used to represent a segment.

	Attributes
	----------
	p1 : Vertex
		one endpoint of the segment
	p2 : Vertex
		another endpoint of the segment

	Methods
	-------
	checkInput()
		checks whether the two endpoints of the segment are of the same dimension
	length()
		returns the length of the segment if checkInput() is True
		returns None otherwise
	dim()
		returns the dimension of the space that the segment lives in if checkInput() is True
		returns None otherwise
	"""

	def __init__(self, p1:Vertex, p2:Vertex):
		self.p1 = p1
		self.p2 = p2

	def checkInput(self):
		""" Checks whether the two endpoints of the segment are of the same dimension. """

		return len(self.p1.v) == len(self.p2.v)

	def length(self):
		""" Returns the length of the segment if checkInput() is True; Returns None otherwise. """
		
		if self.checkInput():
			return norm(subVertex(self.p1,self.p2))
		else:
			return None 

	def dim(self):
		""" Returns the dimension of the space that the segment lives in if checkInput() is True; Returns None otherwise."""
		
		if self.checkInput():
			return len(self.p1.v)
		else:
			return None

def dotDist(p:Vertex, segment:Segment):
	""" Returns the minimal distant from the point to the segment. 

	Parameters
	----------
	p : Vertex
	segment : Segment

	Raises
	------
	ValueError
		If the dimension of p doesn't match the dimension of segment
		If the dimensions of endpoints of the segment do not match
	"""

	if segment.checkInput():
		if len(p.v) == segment.dim():
			p31 = subVertex(p, segment.p1)
			p21 = subvertex(segment.p2, segment.p1)

			t = dot(p31, p21)/dot(p21, p21)
			t = min(1,max(t,0))

			p0 = addVertex(segment.p1, lambdaVertex(t, p21))

			return norm(subVertex(p, p0))
		else:
			raise ValueError("the dimension of p doesn't match the dimension of segment")
	else:
		raise ValueError("the dimensions of endpoints of the segment do not match")