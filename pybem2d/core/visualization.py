from enthought.mayavi import mlab
from evaluation import evaluate
import numpy


class Visualizer(object):

    def __init__(self,evaluator,extent,xn,yn,incWave=None):
        self.evaluator=evaluator
        self.incWave=incWave
        self.f=mlab.figure()
        self.setGrid(extent,xn,yn)
        
       
    def setGrid(self,extent,xn,yn):
        self.gx,self.gy=numpy.mgrid[extent[0]:extent[1]:xn*1j,extent[2]:extent[3]:yn*1j]
        self.points=numpy.array([self.gx.ravel(),self.gy.ravel()])
        self.xn=xn
        self.yn=yn
        self.extent=extent


    def fullField(self,coeffs,imag=False,scale=1):

        if self.f is None: self.f=mlab.figure()
        vals=self.evaluator(self.points,coeffs)
        full=vals+self.incWave(self.points)
        if imag is True:
            full=numpy.imag(full.reshape(self.xn,self.yn))
            scatt=numpy.imag(vals.reshape(self.xn,self.yn))
        else:
            full=numpy.real(full.reshape(self.xn,self.yn))
            scatt=numpy.real(vals.reshape(self.xn,self.yn))

        s1=mlab.imshow(self.gx,self.gy,scatt,vmin=-scale,vmax=scale,figure=self.f)
        offset=1.5*(self.extent[3]-self.extent[2])
        s2=mlab.imshow(self.gx,self.gy-offset,full,vmin=-scale,vmax=scale,figure=self.f)
        ranges=[self.extent[0],self.extent[1],self.extent[2],self.extent[3],0,0]
        mlab.axes(s1,ranges=ranges,zlabel='',y_axis_visibility=False)
        mlab.axes(s2,ranges=ranges,zlabel='',y_axis_visibility=False)
        mlab.view(0,0)

    def scattField(self,coeffs,imag=False,scale=1):

        if self.f is None: self.f=mlab.figure()
        vals=self.evaluator(self.points,coeffs)
        if imag is True:
            scatt=numpy.imag(vals.reshape(self.xn,self.yn))
        else:
            scatt=numpy.imag(vals.reshape(self.xn,self.yn))

        s=mlab.imshow(self.gx,self.gy,scatt,vmin=-scale,vmax=scale,figure=self.f)
        mlab.axes(s,zlabel='',y_axis_visibility=False)
        mlab.view(0,0)

    def saveFig(self,filename):
        mlab.savefig(filename,figure=self.f)

    def show(self):
        mlab.show()

    def reset(self):
        if self.f is not None: mlab.close()
        self.f=mlab.figure()

if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre
    from segments import Line,Arc
    from quadrules import GaussQuadrature
    from kernels import Identity,AcousticDoubleLayer
    from mesh import Domain,Mesh
    from assembly import Assembly
    from evaluation import Evaluator
    from scipy.io import savemat
    

    k=10
    circle=Arc(0,0,0,2*numpy.pi,.5)
    circle2=Arc(2,0,0,2*numpy.pi,.5)
    d=Domain([circle])
    d2=Domain([circle2])
    mesh=Mesh([d])
    mesh.discretize(100)
    quadrule=GaussQuadrature(5,3,0.15)
    mToB=Legendre.legendreBasis(mesh,0)
    kernel=AcousticDoubleLayer(k)

    assembly=Assembly(mToB,quadrule)
    mKernel=assembly.getKernel(kernel)
    mIdentity=assembly.getIdentity()
    op=mIdentity+2*mKernel
    savemat('circmatrix.mat',{'mat':op})
    rhs=assembly.projFun([lambda t,x,normals: -numpy.exp(1j*k*x[1])])
    print rhs.shape

    coeffs=numpy.linalg.solve(.5*mIdentity+mKernel,rhs)
   
    ev=Evaluator(mToB,kernel,quadrule)
    v=Visualizer(ev,[-2,2,-2,2],100,100,incWave=lambda x: numpy.exp(1j*k*x[1]))
    v.scattField(coeffs[:,0])
    v.show()



 
