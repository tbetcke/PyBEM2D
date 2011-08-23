import pybem2d as pbm
import numpy as np

nelems=100

circ=pbm.Arc(3,0,0,2*np.pi,1.5)
sqpoints=[[0,0],[1,0],[1,1],[0,1]]
square_dom=pbm.polygon(sqpoints)
circ_dom=pbm.Domain([circ])
mesh=pbm.Mesh([square_dom,circ_dom])
mesh.discretize(nelems)

quadrule=pbm.GaussQuadrature(n=10) # A standard Gauss Quadrature with default parameters
mToB=pbm.Legendre.legendreBasis(mesh,2) # A basis of Legendre polynomials of degree 2

kernel=pbm.LogDoubleLayer()
assembly=pbm.Assembly(mToB,quadrule)

rhsfun=lambda t,x,n: np.cos(x[0])
rhs=assembly.projFun([rhsfun])

mKernel=assembly.getKernel(kernel)
mIdentity=assembly.getIdentity()
op=.5*mIdentity+mKernel

coeffs=np.linalg.solve(op,rhs)
ev=pbm.Evaluator(mToB,kernel,quadrule)
v=pbm.Visualizer(ev,[-3,5,-3,3],200,200)
v.scattField(coeffs[:,0])


