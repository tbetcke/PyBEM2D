'''
Created on Dec 11, 2010

@author: tbetcke
'''
import numpy as np
from scipy.integrate import quad,odeint
from pybem2d.core.mesh import Domain
import copy

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
    
    def __init__(self,view=(0,1),refine=(False,False)):
        self.view=view
        self.refine=refine

        # Compute length of segment
       

    def length(self):   
        return quad(absderiv(self.fp,self.view),0,1)[0]
        
    def f(self,t):
        pass

    def fp(self,t):
        pass

    def vals(self,t):
        return self.f(self.view[0]+t*(self.view[1]-self.view[0]))
    
    def normals(self,t):
        vals=(self.view[1]-self.view[0])*self.fp(self.view[0]+t*(self.view[1]-self.view[0]))
        n=np.vstack([vals[1,:],-vals[0,:]])
        nl=np.sqrt(n[0,:]**2+n[1,:]**2)
        return n/nl
    
    def det(self,t):
        vals=(self.view[1]-self.view[0])*self.fp(self.view[0]+t*(self.view[1]-self.view[0]))
        return np.sqrt(np.abs(vals[0,:])**2+np.abs(vals[1,:])**2)


def subdivide(seg,n,k=None,nmin=10,L=3,sigma=0.15):
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
    
    if k is not None: n=max(int(1.0*np.real(k)*seg.length()/2/np.pi+1),nmin)
    lvec=seg.length()*np.arange(n,dtype='double')/n
    f= absderiv(seg.fp,seg.view)
    invabsderiv=lambda t,x: 1./f(t)
    t=odeint(invabsderiv,0,lvec).ravel()
    segs=[]
    for i in range(n-1):
        dup=copy.deepcopy(seg)
        dup.view=(seg.view[0]+t[i]*(seg.view[1]-seg.view[0]),seg.view[0]+(seg.view[1]-seg.view[0])*t[i+1])
        segs.append(dup)
    dup=copy.deepcopy(seg)
    dup.view=(seg.view[0]+t[n-1]*(seg.view[1]-seg.view[0]),seg.view[1])
    segs.append(dup)
    if seg.refine[0]:
        newsegs=exponentialRefine(segs[0],sigma,L,lr='l')
        newsegs.extend(segs[1:])
        segs=newsegs
    if seg.refine[1]:
        tmp=segs[:-1]
        tmp.extend(exponentialRefine(segs[-1],sigma,L,lr='r'))
        segs=tmp
    return segs
        
def exponentialRefine(seg,sigma,L,lr='l'):
    """Exponentially refine a segmenent

       If lr='l' refine towards 0.
       If lr='r' refine towards 1.
    """

    intbounds=sigma**np.arange(L+1)
    intbounds=np.append(intbounds,0)

    if lr=='l': 
        intbounds=intbounds[::-1]
    else:
        intbounds=1-intbounds

    segs=[]
    for i in range(len(intbounds)-1):
        t1=intbounds[i]
        t2=intbounds[i+1]
        dup=copy.deepcopy(seg)
        dup.view=(seg.view[0]+t1*(seg.view[1]-seg.view[0]),seg.view[0]+t2*(seg.view[1]-seg.view[0]))
        dup.refine=(False,False)
        segs.append(dup)

    return segs




    
class Line(Segment):
    """Define a line segment
    """
    
    def __init__(self,a,b,refine=(False,False)):
        """Define a line segment
        
           line(a,b) returns a line segment between a and b           
        """

        self.a=a
        self.b=b

        super(Line,self).__init__(refine=refine)

    def f(self,t): 
        return np.vstack([self.a[0]+t*(self.b[0]-self.a[0]),self.a[1]+t*(self.b[1]-self.a[1])])

    def fp(self,t):
        if not (type(t)==np.ndarray): 
            n=1
        else:
            n=len(t)
        return np.tile(np.vstack([[self.b[0]-self.a[0]],[self.b[1]-self.a[1]]]),n)
       


class Arc(Segment):
    """Define an arc segment
    """
    
    def __init__(self,x0,y0,a,b,r,refine=(False,False)):
        """Define an arc segment
        
           Input:
           x0,y0 - Coordinates of centre point
           a - start angle in radians
           b - final angle in radians
           r - radius 
           A circle is created with the call arc(0,0,0,2*pi)
        """
    

        self.x0=x0
        self.y0=y0
        self.a=a
        self.b=b
        self.r=r

    
        super(Arc,self).__init__(refine=refine)

    def f(self,t):
        return np.vstack([self.x0+self.r*np.cos(self.a+t*(self.b-self.a)),self.y0+self.r*np.sin(self.a+t*(self.b-self.a))])
        
    def fp(self,t):
        return (self.b-self.a)*np.vstack([-self.r*np.sin(self.a+t*(self.b-self.a)),self.r*np.cos(self.a+t*(self.b-self.a))])
        
class Ellipse(Segment):
    """Define an ellipse arc

       INPUT:
       x0,y0 - Coordinates of centre point
       a     - start angle in radians
       b     - final angle in radians
       r1    - First semi-axis
       r2    - Second semi-axis

    """
    def __init__(self,x0,y0,a,b,r1,r2,refine=(False,False)):

        self.x0=x0
        self.y0=y0
        self.a=a
        self.b=b
        self.r1=r1
        self.r2=r2
           
        super(Ellipse,self).__init__(refine=refine)

    def f(self,t):
        return np.vstack([self.x0+self.r1*np.cos(self.a+t*(self.b-self.a)),self.y0+self.r2*np.sin(self.a+t*(self.b-self.a))])
        
    def fp(self,t):
        return (self.b-self.a)*np.vstack([-self.r1*np.sin(self.a+t*(self.b-self.a)),self.r2*np.cos(self.a+t*(self.b-self.a))])
            


def polygon(points,refine=True):
    """Define a polygon from a list of points and return 
       a corresponding domain object
    """


    segs=[]
    tmp=points+points[0]
    for i in range(len(points)): segs.append(Line(tmp[i],tmp[i+1],(refine,refine)))
    return Domain(segs)

        
if __name__ == "__main__":
    
   a=Line((0,0),(1,0),refine=(True,True))
   segs=subdivide(a,3)
   for s in segs:
       print s.vals(np.array([0,1]))


