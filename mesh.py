"""
Triangle mesh serialization and manipulation library.
support format id: 6ac96d18-f97d-41b3-9b0e-cc99fb00c03a

Focuses on operation on topologically and geometrically correct mesh.
"""
from __future__ import division, print_function
import numpy as np
import scipy.linalg as la
import math
import msgpack
import time
import sys
from tools3d.affine import *

class Mesh(object):
    """
    Immutable closed triangle mesh.
    """
    def __init__(self, vs, fs):
        self.vs = np.array(vs, dtype=float)
        self.fs = np.array(fs, dtype=int) # fix -> (vix0,vix1,vix2) (CCW)

        assert(self.vs.shape[1] == 3)
        assert(self.fs.shape[1] == 3)

    @staticmethod
    def load(path):
        return Mesh.unpack_headerless(msgpack.unpack(open(path,'rb')))

    @staticmethod
    def unpack_headerless(mesh):
        vs = np.array(mesh['data']['vertex'])
        fs = np.array(mesh['data']['triangle'])
        return Mesh(vs, fs)

    def save_headerless(self, path):
        msgpack.pack(self.pack_headerless(), open(path, 'wb'))

    def pack_headerless(self):
        mesh = {
            'property': {
                'vertex': ["x","y","z"],
                "triangle": []
            },
            'data': {
                'vertex': [map(float,vs) for vs in self.vs],
                'triangle': [map(int,ixs) for ixs in self.fs]
            }
        }
        return mesh

    def genus(self):
        pass

    def intersect_convex(self, other):
        """
        Test if two convex mesh intersects.
        Take O((N+M)^2) time on average for meshes with face count N and M.
        """
        for ns in [self.normals(), other.normals()]:
            dots_s = np.inner(ns, self.vs)
            dots_o = np.inner(ns, other.vs)

            sep_s_o = dots_s.max(axis=1) < dots_o.min(axis=1)
            sep_o_s = dots_o.max(axis=1) < dots_s.min(axis=1)
            if (sep_s_o | sep_o_s).any():
                return False
        
        return True

    def normals(self):
        tris = self.flatten()
        d1s = tris[:,1] - tris[:,0]
        d2s = tris[:,2] - tris[:,0]
        ns = np.cross(d1s, d2s)
        ds = np.sqrt((ns**2).sum(axis=1))
        return ns/ds[:,np.newaxis]
    
    def subdivide_half(self):
        """
        Return new Mesh where faces are subdivided at edge mid-points.

        |vert_new| = |vert| + |edge|
        |face_new| = 4|face|
        """
        mvert = {}
        for (vix0,vix1,vix2) in self.fs:
            if vix0<vix1:
                mvert[(vix0,vix1)] = (self.vs[vix0] + self.vs[vix1])/2
            if vix1<vix2:
                mvert[(vix1,vix2)] = (self.vs[vix1] + self.vs[vix2])/2
            if vix2<vix0:
                mvert[(vix2,vix0)] = (self.vs[vix2] + self.vs[vix0])/2
        assert(2*len(mvert) == 3*len(self.fs)) # each face creates 3 directed edges without overlap


        # create new vertices
        vs_ext = mvert
        vs_ext.update(dict(enumerate(self.vs)))
        fs_ext = []

        for (v0,v1,v2) in self.fs:
            v01 = tuple(sorted([v0,v1]))
            v12 = tuple(sorted([v1,v2]))
            v20 = tuple(sorted([v2,v0]))

            fs_ext += [
                [v01,v12,v20],
                [v0 ,v01,v20],
                [v1 ,v12,v01],
                [v2 ,v20,v12]]

        # flatten vertex indices
        ix_map = dict((i_ext,i) for (i,i_ext) in enumerate(vs_ext.iterkeys()))

        vs = [None]*len(ix_map)
        for (i_ext, pos) in vs_ext.iteritems():
            vs[ix_map[i_ext]] = pos

        fs = []
        for (v0_ext,v1_ext,v2_ext) in fs_ext:
            fs.append((ix_map[v0_ext], ix_map[v1_ext], ix_map[v2_ext]))

        return Mesh(vs, fs)

    def filter_loop(self):
        """
        Apply smoothing filter in Loop's subdivision method.
        """

        def alpha(n):
            return (5/8 - (3/8+math.cos(2*math.pi/n)/4)**2)/n

        # create vix -> set(vix)
        neighbors = {}
        for (v0, v1, v2) in self.fs:
            neighbors.setdefault(v0,set()).update([v1,v2])
            neighbors.setdefault(v1,set()).update([v2,v0])
            neighbors.setdefault(v2,set()).update([v0,v1])

        vs = self.vs.copy()
        for (vix,v) in enumerate(self.vs):
            n = len(neighbors[vix])
            vs[vix] = (1-n*alpha(n))*v + sum(self.vs[nvix] for nvix in neighbors[vix]) * alpha(n)

        return Mesh(vs, self.fs)

    def flatten(self):
        """
        Convert mesh to [N, 3, 3] vertex array.
        Topological information will be lost.
        """
        vs = []
        for (v0, v1, v2) in self.fs:
            vs.append([self.vs[v0], self.vs[v1], self.vs[v2]])

        return np.array(vs)

    def reverse(self):
        """ Turn mesh inside out. """
        return Mesh(self.vs, [list(reversed(f)) for f in self.fs])

    def apply_cont(self, f):
        """
        Apply continuous transform to all vertices.
        """
        return Mesh(f(self.vs), self.fs)

    def __len__(self):
        return len(self.fs)
