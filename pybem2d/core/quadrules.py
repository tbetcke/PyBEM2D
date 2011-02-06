import scipy.special
import numpy

from pylab import plot,show


def gauss(n):
    x,w = scipy.special.orthogonal.p_roots(n)
    x=(x+1)/2.0
    w=.5*w
    return x,w

def tensorquad(x,wx,y,wy):
    """Combine two quadrules in the x and y direction into a 2D tensor rule"""
    
    nx=len(x)
    ny=len(y)
    m=numpy.mgrid[0:nx,0:ny].reshape(2,-1)
    x2=numpy.vstack((x[m[0]],y[m[1]]))
    w2=wx[m[0]]*wy[m[1]]
    return x2,w2

def gauss2d(n):
    x,w=gauss(n)
    return tensorquad(x,w,x,w)

def shift2d(x,w,a,b):
    """Shift a 2d quadrature rule to the box defined by the tuples a and b"""

    xs=[[a[0]],[a[1]]]+x*[[b[0]-a[0]],[b[1]-a[1]]]
    ws=w*(b[0]-a[0])*(b[1]-a[1])
    return xs,ws

class GaussQuadrature:
    """Container class containing the needed 2D quadrature rules"""

    def __init__(self,n,recDepth,sigma):
        self.x,self.w=gauss(n)
        self.x2,self.w2=gauss2d(n)
        self.recDepth=recDepth
        self.sigma=sigma
        
        self.adapted1D()
        self.singElement()
        self.singPoint()

    def adapted1D(self):

        self.x_adapted1D=numpy.array([])
        self.w_adapted1D=numpy.array([])
        a,b=self.sigma,1
        for i in range(self.recDepth+1):
            self.x_adapted1D=numpy.hstack([self.x_adapted1D,a+(b-a)*self.x])
            self.w_adapted1D=numpy.hstack([self.w_adapted1D,(b-a)*self.w])
            a*=self.sigma
            b*=self.sigma

    def singElement(self):

        x,w=tensorquad(self.x_adapted1D,self.w_adapted1D,self.x_adapted1D,self.w_adapted1D)
        x1=numpy.array([x[0],1-x[1]])
        x1[1,:]*=x1[0,:]
        w1=w*x1[0,:]
        x2=numpy.array([1-x[0],x[1]])
        x2[1,:]=x2[0,:]+x2[1,:]*(1-x2[0,:])
        w2=w*(1-x2[0])
        self.x_singElement=numpy.hstack([x1,x2])
        self.w_singElement=numpy.hstack([w1,w2])

    def singPoint(self):

        x,w=tensorquad(self.x,self.w,self.x,self.w)
        a,b=self.sigma,1
        xx=numpy.array([[],[]])
        ww=numpy.array([])
        for i in range(self.recDepth+1):
            xs1,ws1=shift2d(x,w,(0,a),(a,b))
            xs2,ws2=shift2d(x,w,(a,a),(b,b))
            xs3,ws3=shift2d(x,w,(a,0),(b,a))
            xx=numpy.hstack([xx,xs1,xs2,xs3])
            ww=numpy.hstack([ww,ws1,ws2,ws3])
            a*=self.sigma
            b*=self.sigma
        self.x_sing01=numpy.array([xx[0],1-xx[1]])
        self.w_sing01=ww
        self.x_sing10=numpy.array([1-xx[0],xx[1]]) 
        self.w_sing10=ww

    regQuad=property(lambda self: {'x':self.x2, 'w':self.w2})
    elemQuad=property(lambda self: {'x':self.x_singElement,
        'w':self.w_singElement})
    sing01=property(lambda self: {'x':self.x_sing01,'w':self.w_sing01})
    sing10=property(lambda self: {'x':self.x_sing10,'w':self.w_sing10})

if  __name__ == "__main__":
    #print gauss(2)
    x,w=gauss(10)
    x2,w2=shift2d(x,w,(0,0),(1,1))
    g=GaussQuadrature(10,5,0.15)
    #Test quadrature rules
    x,w=g.elemQuad['x'],g.elemQuad['w']
    print sum(x[0]*x[1]*w)


