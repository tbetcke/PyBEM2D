from matplotlib import pyplot as plt
from evaluation import evaluate,evalDensity
import numpy


class Visualizer(object):

    def __init__(self,evaluator,extent,xn,yn,incWave=None):
        self.evaluator=evaluator
        self.incWave=incWave
        self.f=None
        self.setGrid(extent,xn,yn)
        
       
    def setGrid(self,extent,xn,yn):
        self.gx,self.gy=numpy.mgrid[extent[0]:extent[1]:xn*1j,extent[2]:extent[3]:yn*1j]
        self.points=numpy.array([self.gx.ravel(),self.gy.ravel()])
        self.xn=xn
        self.yn=yn
        self.extent=extent


    def fullField(self,coeffs,imag=False,scale=1):

        vals=self.evaluator(self.points,coeffs)
        full=vals+self.incWave(self.points)
        if imag is True:
            full=numpy.imag(full.reshape(self.xn,self.yn))
            scatt=numpy.imag(vals.reshape(self.xn,self.yn))
        else:
            full=numpy.real(full.reshape(self.xn,self.yn))
            scatt=numpy.real(vals.reshape(self.xn,self.yn))


        if self.f is not None:
            plot.close()
            self.f=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.imshow(scatt.T,extent=self.extent,vmin=-scale,vmax=scale,origin='lower')
        plotBnd(self.evaluator.meshToBasis.mesh,ax1)
        ax2=plt.subplot(1,2,2)
        ax2.imshow(full.T,extent=self.extent,vmin=-scale,vmax=scale,origin='lower')
        plotBnd(self.evaluator.meshToBasis.mesh,ax2)
        plt.show()

    def scattField(self,coeffs,imag=False,scale=1):

        vals=self.evaluator(self.points,coeffs)
        if imag is True:
            scatt=numpy.imag(vals.reshape(self.xn,self.yn))
        else:
            scatt=numpy.real(vals.reshape(self.xn,self.yn))

        if self.f is not None:
            plot.close()
            self.f=plt.figure()
        ax=plt.axes()
        ax.imshow(scatt.T,vmin=-scale,vmax=scale,extent=self.extent,origin='lower')
        plotBnd(self.evaluator.meshToBasis.mesh,ax)
        plt.show()

    def saveFig(self,filename):
        plt.savefig(filename)

    def show(self):
        plt.show()

    def close(self):
        if self.f is not None: plt.close()


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
        ax.plot(numpy.real(ylist[i]))

    plt.show()

        
def plotBnd(mesh,ax):
    """Plot the boundary of the mesh"""

    seglist=mesh.segments

    for d in seglist:
        p=numpy.zeros((2,len(d)+1))
        for i,s in enumerate(d):
            p[:,i]=s.vals(numpy.array([0]))[:,0]
        p[:,-1]=d[-1].vals(numpy.array([1]))[:,0]
        ax.plot(p[0,:],p[1,:],'k-',lw=2)


def plotDensity(mToB,coeffs,n=100,plotVals="real",plotBnd=True):
    """Plot boundary density with Mayavi
    
       INPUT:
       mToB     - Mesh-to-Basis map
       coeffs   - Coefficients of boundary density
       n        - Discretisation points per element (default 100)
       plotVals - "real", "imag" or "abs" for real/imaginary part or absolute values
                  (default "real")
       plotBnd  - Also plot boundary of shape (default True)
    """

    from mayavi import mlab

    transforms={"real":numpy.real,"imag":numpy.imag,"abs":numpy.abs}
    
    x,f=evalDensity(mToB,coeffs,n)    
    f=transforms[plotVals](f)
                    
    xmin=min(x[0])
    xmax=max(x[0])
    ymin=min(x[1])
    ymax=max(x[1])
    zmin=max([-1,min(f)])
    zmax=min([1,max(f)])
    extent=[xmin,xmax,ymin,ymax,zmin,zmax]
    ranges=[xmin,xmax,ymin,ymax,min(f),max(f)]
    
    fig=mlab.figure(bgcolor=(1,1,1))
    mlab.plot3d(x[0],x[1],f,extent=extent,color=(0,0,0),tube_radius=None,figure=fig)
    if plotBnd:
        mlab.plot3d(x[0],x[1],numpy.zeros(x[0].shape),extent=[xmin,xmax,ymin,ymax,0,0],tube_radius=None,figure=fig)
    mlab.axes(extent=extent,ranges=ranges,line_width=3.0,figure=fig)
    mlab.show()
    

    
    
    


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
    mesh.discretize(50)
    quadrule=GaussQuadrature(5,3,0.15)
    #mToB=Legendre.legendreBasis(mesh,0)
    mToB=NodalLin.nodalLinBasis(mesh)
    kernel=AcousticDoubleLayer(k)

    assembly=Assembly(mToB,quadrule)
    mKernel=assembly.getKernel(kernel)
    mIdentity=assembly.getIdentity()
    op=mIdentity+2*mKernel
    rhs=assembly.projFun([lambda t,x,normals: -numpy.exp(1j*k*x[0])])

    coeffs=numpy.linalg.solve(.5*mIdentity+mKernel,rhs)
    plotBndFun(mToB,coeffs[:,0])
    ev=Evaluator(mToB,kernel,quadrule)
    v=Visualizer(ev,[-1.5,3.5,-1,3],100,100,incWave=lambda x: numpy.exp(1j*k*x[0]))
    v.fullField(coeffs[:,0])
    v.show()



 
