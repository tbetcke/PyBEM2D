from Queue import Empty
from multiprocessing import Process, Queue, cpu_count
from progressbar import ProgressBar, Percentage, Bar, ETA
import numpy
import time

def integrate1D(elem,funs,quadRule):
    """One dimensional integration routine""" 

    x,w=quadRule.quad1D['x'],quadRule.quad1D['w']

    xp=elem['segment'].vals(x)
    xpdet=elem['segment'].det(x)
    xpnormals=elem['segment'].normals(x)

    f1tx=numpy.array([f(x,xp,xpnormals) for f in elem['basis']]).conj()
    f2tx=numpy.array([f(x,xp,xpnormals) for f in funs]).T

    return numpy.dot(f1tx*xpdet*w,f2tx)

def assembleElement(eTest,eBas,kernel,quadRule=None,forceQuadRule=None):
    """Assemble the submatrix associated with two elements eTest, eBas using the
       quadrature quadrule.

       If forceQuadRule=(x,w) then this quadrature rule is used instead of
       quadrule.
       
    """

    #Get points depending on quad rule.

    segId1,segId2=eTest['segId'],eBas['segId']
    domId1,domId2=eTest['domId'],eBas['domId']

    x,w=[],[]

    if forceQuadRule is None:
        if (domId1!=domId2):
            x,w=quadRule.regQuad['x'],quadRule.regQuad['w']
        else:
            if (segId1==segId2):
                x,w=quadRule.elemQuad['x'],quadRule.elemQuad['w']
            elif (eTest['next']==segId2):
                x,w=quadRule.sing10['x'],quadRule.sing10['w']
            elif (eTest['prev']==segId2):
                x,w=quadRule.sing01['x'],quadRule.sing01['w']
            else:
                x,w=quadRule.regQuad['x'],quadRule.regQuad['w']
    else:
        x,w=forceQuadRule[0],forceQuadRule[1]
    #Evaluate the quadrature

    xp=eTest['segment'].vals(x[0])
    yp=eBas['segment'].vals(x[1])
    xpdet=eTest['segment'].det(x[0])
    ypdet=eBas['segment'].det(x[1])
    xpnormals=eTest['segment'].normals(x[0])
    ypnormals=eBas['segment'].normals(x[1])
    ftx=numpy.array([f(x[0],xp,xpnormals) for f in eTest['basis']]).conj()
    fty=numpy.array([f(x[1],yp,ypnormals) for f in eBas['basis']])


    kernelVals=kernel(xp,yp,nx=xpnormals,ny=ypnormals)
    t1=ftx*xpdet*kernelVals*w
    t2=fty*ypdet
    return numpy.dot(t1,t2.T)




def assembleSegment(eTest,meshToBasis,kernel,quadRule=None,forceQuadRule=None):
    """Assemble the rows of the matrix associated with the test functions on
       the element eTest.

    """

    # Assign numpy array with the right size

    result=numpy.zeros((eTest['nbas'],meshToBasis.nbasis),dtype=numpy.complex128)
    for eBas in meshToBasis:
        result[:,eBas['basIds']]=assembleElement(eTest,eBas,kernel,quadRule,forceQuadRule)
    
    return (eTest['basIds'],result)

class AssemblyWorker(Process):

    def  __init__(self,meshToBasis,inputQueue,outputQueue,kernel,quadRule=None,forceQuadRule=None):
        super(AssemblyWorker,self).__init__()
        self.meshToBasis=meshToBasis
        self.inputQueue=inputQueue
        self.outputQueue=outputQueue
        self.kernel=kernel
        self.quadRule=quadRule
        self.forceQuadRule=forceQuadRule

    def run(self):
        while self.inputQueue.empty() is False:
            try:
                eTest=self.inputQueue.get_nowait()
                result=assembleSegment(eTest,self.meshToBasis,self.kernel,self.quadRule,self.forceQuadRule)
                self.outputQueue.put(result)
            except Empty:
                pass

def assembleMatrix(meshToBasis,kernel,quadRule=None,forceQuadRule=None,nprocs=None):
    """Assemble the discrete BEM matrix using the given kernel"""

    if nprocs==None: nprocs=cpu_count()

    nbasis=meshToBasis.nbasis
    nelements=meshToBasis.nelements

    # Initialize the Queues

    inputQueue=Queue()
    outputQueue=Queue()

    #Initialize the results matrix

    kernelMatrix=numpy.zeros((nbasis,nbasis),dtype=numpy.complex128)

    # Fill the Input Queue

    for eTest in meshToBasis: 
        inputQueue.put(eTest)

    time.sleep(1)
    # Create and start the workers

    workers=[]

    for id in range(nprocs):
        worker=AssemblyWorker(meshToBasis,inputQueue,outputQueue,kernel,quadRule,forceQuadRule)
        worker.start()
        workers.append(worker)

    # Pick up the results from the outputQueue

    widgets=['Assemble matrix:', Percentage(),' ',Bar(),' ',ETA()]
    pbar=ProgressBar(widgets=widgets,maxval=nelements).start()
    for i in range(nelements):
        id,data=outputQueue.get()
        kernelMatrix[id,:]=data
        pbar.update(i)

    # Kill all processess

    for worker in workers: worker.join()
    
    return kernelMatrix


def assembleIdentity(meshToBasis,quadRule):
    """Assemble the Identity Matrix in the given basis"""

    nbasis=meshToBasis.nbasis
    nelements=meshToBasis.nelements
    result=numpy.zeros((nbasis,nbasis),dtype=numpy.complex128)

    for elem in meshToBasis:
        block=integrate1D(elem,elem['basis'],quadRule)
        result[numpy.ix_(elem['basIds'],elem['basIds'])]=block
    return result

def projFun(meshToBasis,fun,quadRule):
    """Project the functions in the list fun onto the basis defined by meshToBasis"""

    nbasis=meshToBasis.nbasis
    nelements=meshToBasis.nelements

    result=numpy.zeros((nbasis,len(fun)),dtype=numpy.complex128)
    for elem in meshToBasis:
        result[elem['basIds']]=integrate1D(elem,fun,quadRule)

    return result

class Assembly(object):

    def __init__(self,meshToBasis,quadRule,nprocs=None):
        self.meshToBasis=meshToBasis
        self.quadRule=quadRule
        self.nprocs=nprocs

    def getIdentity(self):
        return assembleIdentity(self.meshToBasis,self.quadRule)

    def getKernel(self,kernel):
        return assembleMatrix(self.meshToBasis,kernel,self.quadRule,self.nprocs)

    def projFun(self,flist):
        return projFun(self.meshToBasis,flist,self.quadRule)






if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre
    from segments import Line,Arc
    from quadrules import GaussQuadrature
    from kernels import Identity,AcousticDoubleLayer, AcousticSingleLayer
    from mesh import Domain,Mesh

    circle=Arc(0,0,0,2*numpy.pi,1)
    d=Domain([circle])
    mesh=Mesh([d])
    mesh.discretize(100)
    quadrule=GaussQuadrature(4,2,0.15)
    mToB=Legendre.legendreBasis(mesh,0)
    kernel=AcousticSingleLayer(1)
    matrix=assembleMatrix(mToB,kernel,quadRule=quadrule)
    identity=assembleIdentity(mToB,quadrule)
    res=projRhs(mToB,[lambda t,x,normals: numpy.sin(x[0])],quadrule)
    
    #fig=plt.plot(numpy.abs(numpy.diag(identity)))

    print "Finished" 

