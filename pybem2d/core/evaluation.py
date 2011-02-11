import numpy
from Queue import Empty
from multiprocessing import Process, Queue, cpu_count
from progressbar import ProgressBar, Percentage, Bar, ETA

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
        return numpy.dot(coeffs[elem['basIds']],fvals[:,0,:])


    def __init__(self,points,kernel,quadRule,coeffs,inputQueue,outputQueue):
        super(EvaluationWorker,self).__init__()
        self.points=points
        self.kernel=kernel
        self.quadRule=quadRule
        self.coeffs=coeffs
        self.inputQueue=inputQueue
        self.outputQueue=outputQueue

    def run(self,):
        while self.inputQueue.empty() is False:
            try:
                elem=self.inputQueue.get_nowait()
                result=self.evaluate(elem)
                self.outputQueue.put(result)
            except Empty:
                pass

def evaluate(points,meshToBasis,kernel,quadRule,coeffs,nprocs=None):
    """Evaluate a kernel using the given coefficients"""


    if nprocs==None: nprocs=cpu_count()

    inputQueue=Queue()
    outputQueue=Queue()

    nelements=meshToBasis.nelements

    for elem in meshToBasis: inputQueue.put(elem)

    result=numpy.zeros(len(points[0]),dtype=numpy.complex128)

    workers=[]

    for id in range(nprocs):
        worker=EvaluationWorker(points,kernel,quadRule,coeffs,inputQueue,outputQueue)
        worker.start()
        workers.append(worker)

    widgets=['Evaluation:', Percentage(),' ',Bar(),' ',ETA()]
    pbar=ProgressBar(widgets=widgets,maxval=nelements).start()
    for i in range(nelements): 
        result+=outputQueue.get()
        pbar.update(i)

    for worker in workers: worker.join()

    return result

    

if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre
    from segments import Line,Arc
    from quadrules import GaussQuadrature
    from kernels import Identity,AcousticDoubleLayer
    from mesh import Domain,Mesh

    circle=Arc(0,0,0,2*numpy.pi,1)
    d=Domain([circle])
    mesh=Mesh([d])
    mesh.discretize(900)
    quadrule=GaussQuadrature(3,2,0.15)
    mToB=Legendre.legendreBasis(mesh,0)
    kernel=AcousticDoubleLayer(10)
    gx,gy=numpy.mgrid[-2:2:100j,-2:2:100j]
    points=numpy.array([gx.ravel(),gy.ravel()])
    coeffs=numpy.ones(mToB.nbasis)
    res=evaluate(points,mToB,kernel,quadrule,coeffs)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #fig=plt.plot(numpy.real(identity[-1,:]))
    fig=plt.imshow(numpy.real(res.reshape(100,100)),cmap=cm.jet,aspect='equal')
    plt.show()

   
    


    print "Finished" 

    

    

        
