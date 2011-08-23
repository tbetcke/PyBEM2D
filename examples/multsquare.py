import pybem2d.core.bases as pcb
import pybem2d.core.segments as pcs
import pybem2d.core.quadrules as pcq
import pybem2d.core.kernels as pck
import pybem2d.core.mesh as pcm
import pybem2d.core.assembly as pca
import pybem2d.core.evaluation as pce
import pybem2d.core.visualization as pcv

import numpy as np


k=10
nelems=50
dirs=1/np.sqrt(2)*np.array([1.0,1.0])


# Define the mesh
sq1=pcs.polygon([[0,0],[1,0],[1,1],[0,1]])
sq2=pcs.polygon([[1.5,0],[2.5,0],[2.5,1],[1.5,1]])
mesh=pcm.Mesh([sq1,sq2])
mesh.discretize(nelems)

quadrule=pcq.GaussQuadrature() # A standard Gauss Quadrature with default parameters
mToB=pcb.Legendre.legendreBasis(mesh,2) # A basis of Legendre polynomials of degree 2
kernel=pck.AcousticCombined(k,k) # The combined potential layer
singleLayer=pck.AcousticSingleLayer(k)

assembly=pca.Assembly(mToB,quadrule)

rhsfun=lambda t,x,n: 2j*k*np.exp(1j*k*(dirs[0]*x[0]+dirs[1]*x[1]))*(dirs[0]*n[0]+dirs[1]*n[1]-1)
rhs=assembly.projFun([rhsfun])

mKernel=assembly.getKernel(kernel)
mIdentity=assembly.getIdentity()
op=mIdentity+2*mKernel
print op.shape

coeffs=np.linalg.solve(op,rhs)
ev=pce.Evaluator(mToB,singleLayer,quadrule)
v=pcv.Visualizer(ev,[-1,4,-1,3],200,200,incWave=lambda x: np.exp(1j*k*(x[0]*dirs[0]+x[1]*dirs[1])))
v.fullField(-coeffs[:,0])











