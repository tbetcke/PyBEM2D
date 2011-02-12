import numpy
from Queue import Empty
from multiprocessing import Process, Queue, cpu_count, sharedctypes
from progressbar import ProgressBar, Percentage, Bar, ETA
import time

class EvaluationWorker(Process):


    def evaluate(self,elem):

        result=numpy.zeros(self.points.shape[1],dtype=numpy.complex128)
        x,w=self.quadRule.quad1D['x'],self.quadRule.quad1D['w']
        nx=len(x)
        np=len(self.points[0])
        yp=elem['segment'].vals(x)
        dets=elem['segment'].det(x)
        normals=elem['segment'].normals(x)
        f=numpy.array([bas(x,yp,normals) for bas in elem['basis']])*dets*w
        ker=numpy.array([self.kernel(self.points,y.reshape(2,1),ny=normals[:,i]) for (i,y) in
            enumerate(yp.T)])
        fvals=numpy.dot(f.reshape(len(elem['basis']),1,nx),ker)
        return numpy.dot(self.coeffs[elem['basIds']],fvals[:,0,:])


    def  __init__(self,points,kernel,quadRule,coeffs,inputQueue,outputQueue,resVec):
        super(EvaluationWorker,self).__init__()
        self.points=points
        self.kernel=kernel
        self.quadRule=quadRule
        self.coeffs=coeffs
        self.inputQueue=inputQueue
        self.outputQueue=outputQueue
        self.resVec=resVec

    def run(self,):
        while self.inputQueue.empty() is False:
            try:
                elem=self.inputQueue.get_nowait()
                self.resVec+=self.evaluate(elem)
                self.outputQueue.put("DONE")
            except Empty:
                pass

def evaluate(points,meshToBasis,kernel,quadRule,coeffs,nprocs=None):
    """Evaluate a kernel using the given coefficients"""


    if nprocs==None: nprocs=cpu_count()

    inputQueue=Queue()
    outputQueue=Queue()

    nelements=meshToBasis.nelements

    for elem in meshToBasis: inputQueue.put(elem)
    time.sleep(1)

    buf=sharedctypes.RawArray('b',len(points[0])*numpy.dtype(numpy.complex128).itemsize)
    result=numpy.frombuffer(buf,dtype=numpy.complex128)
    result[:]=numpy.zeros(1,dtype=numpy.complex128)

    workers=[]

    for id in range(nprocs):
        worker=EvaluationWorker(points,kernel,quadRule,coeffs,inputQueue,outputQueue,result)
        worker.start()
        workers.append(worker)

    widgets=['Evaluation:', Percentage(),' ',Bar(),' ',ETA()]
    pbar=ProgressBar(widgets=widgets,maxval=nelements).start()
    for i in range(nelements): 
        outputQueue.get()
        pbar.update(i)

    for worker in workers: worker.join()

    return result.copy()

class Evaluator(object):

    def __init__(self,meshToBasis,kernel,quadRule,nprocs=None):
        self.meshToBasis=meshToBasis
        self.quadRule=quadRule
        self.nprocs=nprocs
        self.kernel=kernel

    def __call__(self,points,coeffs):
        return evaluate(points,self.meshToBasis,self.kernel,self.quadRule,coeffs,self.nprocs)


if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre
    from segments import Line,Arc
    from quadrules import GaussQuadrature
    from kernels import Identity,AcousticDoubleLayer
    from mesh import Domain,Mesh
    from assembly import assembleIdentity, assembleMatrix, projRhs, assembleElement

    k=10
    circle=Arc(0,0,0,2*numpy.pi,.5)
    circle2=Arc(2,0,0,2*numpy.pi,.5)
    d=Domain([circle])
    d2=Domain([circle2])
    mesh=Mesh([d,d2])
    mesh.discretize(100)
    quadrule=GaussQuadrature(5,3,0.15)
    mToB=Legendre.legendreBasis(mesh,2)
    kernel=AcousticDoubleLayer(k)

    matrix=assembleMatrix(mToB,kernel,quadRule=quadrule)
    print matrix[0,0],matrix[1,0],matrix[2,0]
    identity=assembleIdentity(mToB,quadrule)


    rhs=projRhs(mToB,[lambda t,x,normals: -numpy.exp(1j*k*x[1])],quadrule)
    coeffs=numpy.linalg.solve(.5*identity+matrix,rhs)
    

    gx,gy=numpy.mgrid[-2:4:200j,-2:2:200j]
    points=numpy.array([gx.ravel(),gy.ravel()])
    res=evaluate(points,mToB,kernel,quadrule,coeffs)
    res=res.reshape(200,200)
    res+=numpy.exp(1j*k*gy)
    from enthought.mayavi import mlab
    mlab.imshow(gx,gy,numpy.real(res),vmin=-1,vmax=1)
    mlab.axes()
    mlab.view(0,0)
    mlab.show()
   
    


    print "Finished" 

    

    

        
