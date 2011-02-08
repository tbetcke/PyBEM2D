'''
Created on Dec 11, 2010

@author: tbetcke
'''
import numpy as np
from scipy.integrate import quad,odeint

def absderiv(fp,view):
    def absderivfun(t):
        vals=(view[1]-view[0])*fp(view[0]*t*(view[1]-view[0]))
        return np.sqrt(np.abs(vals[0,:])**2+np.abs(vals[1,:])**2)
    return absderivfun
        

class Segment(object):
    """Defines a segment
    
       A segment is a smooth arc given by a parametric function
       t->f(t), t in (0,1) and its tangential derivative fp(t)
       If t is an array array([t0,t1,..,tn]). Then
       f(t)=[[fx(t0),fx(t1),...,fx(tn)],
             [fy(t0),fy(t1),...,fy(tn)]]

       and
       fp(t)=[[fpx(t0),fpx(t1),...,fpx(tn)],
              [fpy(t0),fpy(t1),...,fpy(tn)]]

       Methods:
        vals(t)    - Values of the Parameterization at t
        normals(t) - Return normal vectors
        det(t)     - Return |fp(t)|

    """
    
    def __init__(self,f,fp,view=(0,1)):
        self.f=f
        self.fp=fp
        self.view=view
        
        # Compute length of segment
        
        self.L=quad(absderiv(fp,view),0,1)[0]
        
    def vals(self,t):
        return self.f(self.view[0]+t*(self.view[1]-self.view[0]))
    
    def normals(self,t):
        vals=(self.view[1]-self.view[0])*self.fp(self.view[0]+t*(self.view[1]-self.view[0]))
        n=np.vstack([vals[1,:],-vals[0,:]])
        return n
    
    def det(self,t):
        vals=(self.view[1]-self.view[0])*self.fp(self.view[0]+t*(self.view[1]-self.view[0]))
        return np.sqrt(np.abs(vals[0,:])**2+np.abs(vals[1,:])**2)

    length=property(lambda self: self.L)


def subdivide(seg,n,k=None,nmin=10):
    """Subdivide a segment into equal length subsegments
    
       INPUT:
       seg - Segment
       n - Number of subsegments
       k - If k is given then it is divided into roughly n elements per wavelength.
           However, at least nmin subsegments are created.
       nmin - See above
       
       OUTPUT:
       
       A list of new segments
    """
    
    if k is not None: n=max(1.0*k*seg.L/2/np.pi,nmin)
    lvec=seg.L*np.arange(n,dtype='double')/n
    f= absderiv(seg.fp,seg.view)
    invabsderiv=lambda t,x: 1./f(t)
    t=odeint(invabsderiv,0,lvec).ravel()
    segs=[]
    for i in range(n-1): segs.append(Segment(seg.f,seg.fp,(seg.view[0]+t[i]*(seg.view[1]-seg.view[0]),seg.view[0]+(seg.view[1]-seg.view[0])*t[i+1])))
    segs.append(Segment(seg.f,seg.fp,(seg.view[0]+t[n-1]*(seg.view[1]-seg.view[0]),seg.view[1])))
    return segs
        
    
    
class Line(Segment):
    """Define a line segment
    """
    
    def __init__(self,a,b):
        """Define a line segment
        
           line(a,b) returns a line segment between a and b           
        """
        f= lambda t: np.vstack([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])])
        fp = lambda t: np.vstack([[b[0]-a[0]],[b[1]-a[1]]])
        super(Line,self).__init__(f,fp)
        
class Arc(Segment):
    """Define an arc segment
    """
    
    def __init__(self,x0,y0,a,b):
        """Define an arc segment
        
           Input:
           x0,y0 - Coordinates of centre point
           a - start angle in radians
           b - final angle in radians
           
           A circle is created with the call arc(0,0,0,2*pi)
        """
        
        f = lambda t: np.vstack([x0+np.cos(a+t*(b-a)),y0+np.sin(a+t*(b-a))])
        fp= lambda t: (b-a)*np.vstack([-np.sin(a+t*(b-a)),np.cos(a+t*(b-a))])
        super(Arc,self).__init__(f,fp)
        
        
if __name__ == "__main__":
    
    circle=Arc(0,0,0,np.pi)
    print circle.L
    segs=subdivide(circle,10)
    print len(segs)
    print segs[0].view,segs[0].L,segs[0].vals(np.array([1]))
    segs2=subdivide(segs[0],10)
    print segs2[0].view,segs2[0].L,segs2[0].vals(np.array([1]))
