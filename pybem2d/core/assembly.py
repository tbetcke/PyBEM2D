from multiprocessing import Process, Queue
import numpy


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
            x,w=quadrule.reqQuad['x'],quadRule.regQuad['w']
        else:
            if (segId1==segId2):
                x,w=quadrule.elemQuad['x'],quadRule.elemQuad['w']
            elif (eTest['next']==segId2):
                x,w=quadrule.sing10['x'],quadRule.sing10['w']
            elif (eTest['prev']==segId2):
                x,w=quadrule.sing01['x'],quadRule.sing01['w']
            else:
                x,w=quadrule.regQuad['x'],quadRule.regQuad['w']
    else:
        x,w=forceQuadRule[0],forceQuadRule[1]
    #Evaluate the quadrature

    print x
    ftx=numpy.array([f(x[0]) for f in eTest['basis']]).conj()
    fty=numpy.array([f(x[1]) for f in eBas['basis']])

    xp=eTest['segment'].vals(x[0])
    yp=eBas['segment'].vals(x[1])
    xpdet=eTest['segment'].det(x[0])
    ypdet=eBas['segment'].det(x[1])
    xpnormals=eTest['segment'].normals(x[0])
    ypnormals=eBas['segment'].normals(x[1])

    kernelVals=kernel(xp,yp,nx=xpnormals,ny=ypnormals)
    t1=ftx*xpdet*kernelVals*w
    t2=fty*ypdet
    return numpy.dot(t1,t2.T)

def assembleSegment(eTest,meshToBasis,kernel,quadRule=None,forceQuadRule=None):
    """Assemble the rows of the matrix associated with the test functions on
       the element eTest.

    """

    # Assign numpy array with the right size

    result=numpy.zeros((eTest['nbas'],mesh.nbasis))
    for eBas in meshToBasis:
        result[:,eBas['basIds']]=assembleElement(eTest,eBas,kernel,quadRule,forceQuadRule)
    
    return (eTest['basIds'],result)

def AssemblyWorker(Process):

    def  __init__(meshToBasis,inputQueue,outputQueue,kernel,quadRule=None,forceQuadRule=None):
        super(Assembly,self).__init__()
        self.meshToBasis=meshToBasis
        self.inputQueue=inputQueue
        self.outputQueue=outputQueue
        self.kernel=kernel
        self.quadRule=quadRule
        self.forceQuadRule=forceQuadRule

    def run():
        while inputQueue.empty() is False:
            try:
                eTest=inputQueue.get_nowait()
                result=assembleSegment(eTest,meshToBasis,kernel,quadRule,forceQuadRule)
                outputQueue.put(result)
            except Queue.Empty:
                pass




    


if  __name__ == "__main__":

    from bases import ElementToBasis,Legendre
    from segments import Line
    from quadrules import GaussQuadrature
    from kernels import Identity

    l1=Line((0,0),(1,0))
    l2=Line((1,0),(1,1))
    quadrule=GaussQuadrature(1,1,0.15)

    eTest=ElementToBasis(l1,0,0,2,2)
    eTest.addBasis(Legendre(0),0)
    eBas=ElementToBasis(l2,0,1,2,2)
    eBas.addBasis(Legendre(0),1)
    print assembleElement(eTest,eBas,Identity(),quadrule=quadrule)
    

