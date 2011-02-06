from scipy.special import hankel1
from numpy import sqrt,log


def absdiff(x,y):
    """Return the vectorized norm of the difference between x and y"""
    w=x-y
    return sqrt(w[0]**2+w[1]**2)

class LogSingleLayer(object):
    """Logarithmic Single Layer Potential"""

    def __call__(self,x,y,normal=None):
        return log(0,absdiff(x,y))

class LogDoubleLayer(object):
    """Logarithmic Double Layer Potential"""

    def __call__(self,x,y,normal):
        w=y-x
        a=w[0]**2+w[1]**2
        return (normal[0]*w[0]+normal[1]*w[1])*w/a

class LogConjDoubleLayer(object):
    """Logarithmic Conjugate Double Layer Potential"""

    def __call__(self,x,y,normal):
        w=x-y
        a=w[0]**2+w[1]**2
        return (normal[0]*w[0]+normal[1]*w[1])*w/a

class AcousticSingleLayer(object):
    """Acoustic Single Layer Potential"""

    def __init__(k):
        self.k=k

    def __call__(self,x,y,normal=None):
        return hankel1(0,k*absdiff(x,y))
        

if  __name__ == "__main__":
    lsing=LogSingleLayer()
    ldouble=LogDoubleLayer()
    from numpy import array
    x=array([[1,2],[3,4.5]])
    y=array([[4,5.1],[-6,7]])
    normal=array([[1,2],[2,3]])
    print ldouble(x,y,normal)


