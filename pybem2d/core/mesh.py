'''
Created on Dec 11, 2010

@author: tbetcke
'''
from segments import subdivide

class Domain(object):
    """A domain is a closed arc consisting of one or more segments"""

    def __init__(self,seglist):
        self.seglist=seglist
        self.L=sum([s.L for s in seglist])

    def discretize(self,N,k=None,nmin=10):
        return sum([subdivide(s,N,k,nmin) for s in self.seglist],[])
        
    segments=property(lambda self: self.seglist)
    nsegments=property(lambda self: len(self.seglist))
    length=property(lambda self: self.L)

class Mesh(object):
    """A mesh is a collection of one or more domains"""

    def __init__(self,domains):
        self.domainlist=domains
        self.seglist=None
        self.nseglist=None
        self.seglist_flattened=None

    def discretize(self,N,k=None,nmin=10):
        self.seglist=[d.discretize(N,k,nmin) for d in self.domainlist]
        self.nseglist=sum([len(d) for d in self.seglist])
        self.seglist_flattened=sum(self.seglist,[])
    
    domains=property(lambda self: self.domainlist)
    nsegments=property(lambda self:self.nseglist)
    segments=property(lambda self:self.seglist)
    segments_flattened=property(lambda self: self.seglist_flattened)

if __name__ == "__main__":

    from segments import Line
    from numpy import array
    # Define a square domain
    a=Line((0,0),(1,0))
    b=Line((1,0),(1,1))
    c=Line((1,1),(0,1))
    d=Line((0,1),(0,0))

    d=Domain([a,b,c,d])
    mesh=Mesh([d])
    mesh.discretize(2)
    print mesh.segments
    for s in mesh.segments_flattened: print s.vals(array([0]))
