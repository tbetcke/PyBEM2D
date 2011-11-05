from Queue import Empty
from multiprocessing import Process, Queue, cpu_count, sharedctypes, JoinableQueue
import numpy
import time
from scipy.linalg import circulant

def integrate1D(elem,funs,quadRule,fun2=None):
    """One dimensional integration routine
    
       Integrates fun over an element. If fun2 is defined the product
       fun*fun2 is integrated.
    """ 

    x,w=quadRule.quad1D['x'],quadRule.quad1D['w']

    xp=elem['segment'].vals(x)
    xpdet=elem['segment'].det(x)
    xpnormals=elem['segment'].normals(x)

    f1tx=numpy.array([f(x,xp,xpnormals) for f in elem['basis']]).conj()
    if fun2 is not None:
        f2tx=numpy.array([f(x,xp,xpnormals)*fun2(x,xp,xpnormals) for f in funs]).T
    else:
        f2tx=numpy.array([f(x,xp,xpnormals) for f in funs]).T    

    return numpy.dot(f1tx*xpdet*w,f2tx)

def assembleElement(eTest,eBas,kernel,quadRule=None,forceQuadRule=None,multOpx=None,multOpy=None):
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
    if multOpy is not None: fty=fty*multOpy(x[1],yp,ypnormals)
    if multOpx is not None: ftx=ftx*numpy.conj(multOpx(x[0],xp,xpnormals))

    kernelVals=kernel(xp,yp,nx=xpnormals,ny=ypnormals)
    t1=ftx*xpdet*kernelVals*w
    t2=fty*ypdet
    return numpy.dot(t1,t2.T)




def assembleSegment(eTest,meshToBasis,kernel,quadRule=None,forceQuadRule=None,multOpx=None,multOpy=None):
    """Assemble the rows of the matrix associated with the test functions on
       the element eTest.

    """

    # Assign numpy array with the right size

    result=numpy.zeros((eTest['nbas'],meshToBasis.nbasis),dtype=numpy.complex128)
    for eBas in meshToBasis:
        result[:,eBas['basIds']]=assembleElement(eTest,eBas,kernel,quadRule,forceQuadRule,multOpx=multOpx,multOpy=multOpy)
    
    return result

class AssemblyWorker(Process):

    def __init__(self,meshToBasis,inputQueue,resMatrix,kernel,quadRule=None,forceQuadRule=None,multOpx=None,multOpy=None):
        super(AssemblyWorker,self).__init__()
        self.meshToBasis=meshToBasis
        self.inputQueue=inputQueue
        self.kernel=kernel
        self.quadRule=quadRule
        self.forceQuadRule=forceQuadRule
        self.resMatrix=resMatrix
        self.multOpx=multOpx
        self.multOpy=multOpy

    def run(self):
        while self.inputQueue.empty() is False:
            try:
                eTest=self.inputQueue.get_nowait()
                result=assembleSegment(eTest,self.meshToBasis,self.kernel,self.quadRule,self.forceQuadRule,multOpx=self.multOpx,multOpy=self.multOpy)
                self.resMatrix[eTest['basIds'],:]=result
                self.inputQueue.task_done()
            except Empty:
                pass

def assembleMatrix(meshToBasis,kernel,quadRule=None,forceQuadRule=None,multOpx=None,multOpy=None,nprocs=None):
    """Assemble the discrete BEM matrix using the given kernel"""

    if nprocs==None: nprocs=cpu_count()

    nbasis=meshToBasis.nbasis
    nelements=meshToBasis.nelements

    # Initialize the Queues

    inputQueue=JoinableQueue()


    #Initialize shared buffer

    buf=sharedctypes.RawArray('b',nbasis*nbasis*numpy.dtype(numpy.complex128).itemsize)
    resMatrix=numpy.frombuffer(buf,dtype=numpy.complex128)
    resMatrix.shape=(nbasis,nbasis)

    # Fill the Input Queue

    for eTest in meshToBasis: 
        inputQueue.put(eTest)

    # Create and start the workers
    time.sleep(.5)
    workers=[]

    for id in range(nprocs):
        worker=AssemblyWorker(meshToBasis,inputQueue,resMatrix,kernel,quadRule,forceQuadRule,multOpx=multOpx,multOpy=multOpy)
        worker.start()
        workers.append(worker)


    # Kill all processess

    inputQueue.join()
    for worker in workers: worker.join()
    
    return resMatrix.copy() 


def assembleIdentity(meshToBasis,quadRule):
    """Assemble the Identity Matrix in the given basis"""

    return assembleMultiplicationOperator(meshToBasis,quadRule)
    
def assembleMultiplicationOperator(meshToBasis, quadRule, multOp=None):
    """Assemble a multiplication operator in the given basis
    
       If multOp is None then the discretisation of the identity operator
       is returned
    
    """    
    nbasis=meshToBasis.nbasis
    nelements=meshToBasis.nelements
    result=numpy.zeros((nbasis,nbasis),dtype=numpy.complex128)

    for elem in meshToBasis:
        block=integrate1D(elem,elem['basis'],quadRule,fun2=multOp)
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

def nodalProjector(meshToBasis):
    """Create the projector onto a nodal basis"""

    segments=meshToBasis.mesh.segments # List of segments in each domain
    nsegs=[len(s) for s in segments] # Number of segments in each domain
    ne=meshToBasis.nelements
    nb=meshToBasis.nbasis

    # Create local matrices
    P1l,P2l=[],[]
    for n in nsegs:
        P2=numpy.eye(n,n,dtype=numpy.complex128)
        P1=circulant(P2[:,1])
        P1l.append(P1)
        P2l.append(P2)
    P=numpy.zeros((ne,nb),dtype=numpy.complex128)
    ind1,ind2=0,ne
    for i,n in enumerate(nsegs):
        P[ind1:ind1+n,ind1:ind1+n]=P1l[i]
        P[ind1:ind1+n,ind2:ind2+n]=P2l[i]
        ind1+=n
        ind2+=n
    return P
    

class Assembly(object):

    def __init__(self,meshToBasis,quadRule,nprocs=None):
        self.meshToBasis=meshToBasis
        self.quadRule=quadRule
        self.nprocs=nprocs
        self.P=meshToBasis.P


    def getIdentity(self):
        return self.getMultOperator()
                
    def getMultOperator(self,multOp=None):
        result=assembleMultiplicationOperator(self.meshToBasis,self.quadRule,multOp)
        if self.P is not None:
            return numpy.dot(self.P,numpy.dot(result,self.P.T))
        else:
            return result


    def getKernel(self,kernel,multOpx=None,multOpy=None):
        kMatrix=assembleMatrix(self.meshToBasis,kernel,self.quadRule,multOpx=multOpx,multOpy=multOpy,nprocs=self.nprocs)
        if self.P is not None:
            return numpy.dot(self.P,numpy.dot(kMatrix,self.P.T))
        else:
            return kMatrix

    def projFun(self,flist):
        vec=projFun(self.meshToBasis,flist,self.quadRule)
        if self.P is not None:
            return numpy.dot(self.P,vec)
        else:
            return vec






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

