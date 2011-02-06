import scipy.special
import numpy



def gauss(n):
    x,w = scipy.special.orthogonal.p_roots(n)
    x=(x+1)/2.0
    w=.5*w
    return x,w

def gauss2d(n):
    x,w=gauss(n)
    m=numpy.mgrid[0:n,0:n].reshape(2,-1)
    w2=w[m[0]]*w[m[1]]
    x2=numpy.vstack((x[m[0]],x[m[1]]))
    return x2,w2

def shift2d(x,w,a,b):
    """Shift a 2d quadrature rule to the box defined by the tuples a and b"""

    xs=[[a[0]],[a[1]]]+x*[[b[0]-a[0]],[b[1]-a[1]]]
    ws=w*(b[0]-a[0])*(b[1]-a[1])
    return xs,ws

class GaussQuadrature:
    """Container class containing the needed 2D quadrature rules"""

    def __init__():
        pass


if __name__ == "__main__":
    print gauss(2)
    x,w=gauss2d(2)
    print shift2d(x,w,(1,1),(4,3))

