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


def plotBndFun(meshToBasis,coeffs,n=100):
    nelems=meshToBasis.nelements
    ndoms=len(meshToBasis.mesh.domains)
    segments=meshToBasis.mesh.segments
    P=meshToBasis.P
    if P is not None:
        coeffs=numpy.dot(P.T,coeffs)
    ylist=[numpy.zeros(n*len(segments[i]),dtype=numpy.complex128) for i in range(ndoms)]
    xvec=numpy.arange(n,dtype=numpy.double)/n
    ind=numpy.zeros(ndoms)

    for elem in meshToBasis:
        xp=elem['segment'].vals(xvec)
        normals=elem['segment'].normals(xvec)
        t=numpy.array([f(xvec,xp,normals) for f in elem['basis']])
        vals=numpy.dot(numpy.array(coeffs[elem['basIds']]),numpy.array([f(xvec,xp,normals) for f in elem['basis']]))
        ylist[elem['domId']][n*ind[elem['domId']]:(ind[elem['domId']]+1)*n]=vals
        ind[elem['domId']]+=1
    import matplotlib.pyplot as plt
    fig=plt.figure()
    for i in range(ndoms):
        ax=fig.add_subplot(10*ndoms+100+i+1)
        ax.plot(numpy.abs(ylist[i]))
    plt.show()

        




if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre,NodalLin
    from segments import Line,Arc
    from quadrules import GaussQuadrature
    from kernels import Identity,AcousticDoubleLayer
    from mesh import Domain,Mesh
    from assembly import Assembly, nodalProjector
    from evaluation import Evaluator
    

    k=10
    circle=Arc(0,0,0,2*numpy.pi,.5)
    circle2=Arc(2,0,0,2*numpy.pi,.5)
    d=Domain([circle])
    d2=Domain([circle2])
    mesh=Mesh([d,d2])
    mesh.discretize(3)
    quadrule=GaussQuadrature(5,3,0.15)
    #mToB=Legendre.legendreBasis(mesh,0)
    mToB=NodalLin.nodalLinBasis(mesh)
    kernel=AcousticDoubleLayer(k)

    assembly=Assembly(mToB,quadrule)
    mKernel=assembly.getKernel(kernel)
    mIdentity=assembly.getIdentity()
    op=mIdentity+2*mKernel
    rhs=assembly.projFun([lambda t,x,normals: -numpy.exp(1j*k*x[1])])

    coeffs=numpy.linalg.solve(.5*mIdentity+mKernel,rhs)
    plotBndFun(mToB,coeffs[:,0])
    ev=Evaluator(mToB,kernel,quadrule)
    v=Visualizer(ev,[-1.5,3.5,-1,3],200,200,incWave=lambda x: numpy.exp(1j*k*x[1]))
    v.fullField(coeffs[:,0])
    v.show()



 
