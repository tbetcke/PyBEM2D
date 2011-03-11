from scipy.special import hankel1
from numpy import sqrt,log, ones


def absdiff(x,y):
    """Return the vectorized norm of the difference between x and y"""
    w=x-y
    return sqrt(w[0]**2+w[1]**2)

class LogSingleLayer(object):
    """Logarithmic Single Layer Potential"""

    def __call__(self,x,y,nx=None,ny=None):
        return log(0,absdiff(x,y))

class LogDoubleLayer(object):
    """Logarithmic Double Layer Potential"""

    def __call__(self,x,y,nx=None,ny=None):
        w=y-x
        a=w[0]**2+w[1]**2
        return (ny[0]*w[0]+ny[1]*w[1])/a

class LogConjDoubleLayer(object):
    """Logarithmic Conjugate Double Layer Potential"""

    def __call__(self,x,y,nx=None,ny=None):
        w=x-y
        a=w[0]**2+w[1]**2
        return (nx[0]*w[0]+nx[1]*w[1])/a

class AcousticSingleLayer(object):
    """Acoustic Single Layer Potential"""

    def __init__(self,k):
        self.k=k

    def __call__(self,x,y,nx=None,ny=None):
        return 1j/4*hankel1(0,self.k*absdiff(x,y))

class AcousticDoubleLayer(object):
    """Acoustic Double Layer Potential"""

    def __init__(self,k):
        self.k=k

    def __call__(self,x,y,nx=None,ny=None):
        w=y-x
        a=sqrt(w[0]**2+w[1]**2)
        return -self.k*1j/4*hankel1(1,self.k*a)*(ny[0]*w[0]+ny[1]*w[1])/a

class AcousticConjDoubleLayer(object):
    """Acoustic Conjugate Double Layer Potential"""

    def __init__(self,k):
        self.k=k

    def __call__(self,x,y,nx=None,ny=None):
        w=x-y
        a=sqrt(w[0]**2+w[1]**2)
        return -self.k*1j/4*hankel1(1,self.k*a)*(nx[0]*w[0]+nx[1]*w[1])/a

class AcousticCombined(object):

    def __init__(self,k,eta):
        self.k=k
        self.eta=eta
        self.Conj=AcousticConjDoubleLayer(k)
        self.Single=AcousticSingleLayer(k)

    def __call__(self,x,y,nx=None,ny=None):
        return self.Conj(x,y,nx,ny)-1j*self.eta*self.Single(x,y,nx,ny)


class Identity(object):

    def __call__(self,x,y,nx=None,ny=None):
        return ones(x.shape[1])

if  __name__ == "__main__":
    lsing=LogSingleLayer()
    ldouble=LogDoubleLayer()
    from numpy import array
    x=array([[1,2],[3,4.5]])
    y=array([[4,5.1],[-6,7]])
    normal=array([[1,2],[2,3]])
    print ldouble(x,y,ny=normal)



