from scipy.special import eval_sh_legendre


class ElementToBasis(object):
    """Element to basis map"""

    def __init__(self,seg,domId,segId,prevElem,nextElem):
        self.etob={'segment':seg,'domId':domId,'segId':segId,'basis':[],'basIds':[],'nbas':0,'next':nextElem,'prev':prevElem}

    def addBasis(self,basis,n):
        """Add a basis object with the global id n"""

        self.etob['basis'].append(basis)
        self.etob['basIds'].append(n)
        self.etob['nbas']+=1

    def __getitem__(self,item):
        return self.etob[item]
    

class MeshToBasis(object):
    """Mesh to basis map"""

    def __init__(self,mesh):
        self.meshToBasis=([ElementToBasis(mesh.segments[domId][segId],domId,segId,(segId-1)%len(mesh.segments[domId]),
            (segId+1)%len(mesh.segments[domId])) 
                for domId in range(len(mesh.segments)) for segId in
                range(len(mesh.segments[domId]))])
        self.nb=0
        
    def __iter__(self):
        for i in self.meshToBasis: yield i

    def __getitem__(self,index):
        return self.meshToBasis[index]

    def addBasis(self,basis):
        for e in self.meshToBasis:
            e.addBasis(basis,self.nb)
            self.nb+=1

    nelements=property(lambda self:len(self.meshToBasis))
    nbasis=property(lambda self: self.nb)


class Legendre:
    """Basis of Legendre Polynomials"""

    def __init__(self,n):
        """Degree n"""

        self.n=n

    def __call__(self,t,x=None,normal=None):
        """Callable for the Legendre Basis

        t: local variable of parameterization in [0,1]
        x: Global coordinate
        normal: normal direction of current point
        
        """

        return eval_sh_legendre(self.n,t)

    @staticmethod
    def legendreBasis(mesh,n):
        """Return a MeshToBasis object initialized with Legendre Basis functions of max degree n"""

        mToB=MeshToBasis(mesh)
        for i in range(n+1): mToB.addBasis(Legendre(i))
        return mToB



if __name__ == "__main__":
    bas=Legendre(1)
    from numpy import array
    print bas(array([1E-10,.5,1]))

    from segments import Line
    from mesh import Mesh,Domain

    a=Line((0,0),(1,0))
    b=Line((1,0),(1,1))
    c=Line((1,1),(0,1))
    d=Line((0,1),(0,0))

    d=Domain([a,b,c,d])
    mesh=Mesh([d,d])
    mesh.discretize(2)
    mToB=Legendre.legendreBasis(mesh,3)
    for e in mToB: print e['domId'],e['segId'],e['next'],e['prev']
    print mToB.nbasis
