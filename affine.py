# coding=utf-8
"""
affine: provides 2d and 3d rotation and translation library
"""
from __future__ import division
import math
import numpy as np

class SE2(object):
    """
    Lie Group SE(2): translation and rotation in plane

    T(x)=R(self.z)x+self.d
    """
    def __init__(self, d, z):
        """ You should use identity, rotate or translate instead of __init__ """
        self.d = d
        self.z = z/abs(z)

    @staticmethod
    def identity():
        """ Return the identity transform. """
        return SE2(np.zeros(2), complex(1))

    @staticmethod
    def rotate(angle):
        """ Return a rotation transform with specified angle. """
        return SE2(np.zeros(2), complex(math.cos(angle), math.sin(angle)))
    
    @staticmethod
    def translate(d):
        assert(d.shape == (2,))
        return SE2(d, complex(1))

    @staticmethod
    def jacobian(p):
        """
        Return Jacobian: dp -> d(SE2.small(dp).apply(p))
        Actually dp = (dx, dy, dz.angle), but you can forget that.
        """
        return np.array([[1,0,-p[1]],[0,1,p[0]]])

    @staticmethod
    def small(dp):
        """ See help(SE2.jacobian). """
        assert(dp.shape==(3,))
        return SE2.translate(dp[:2]).apply(SE2.rotate(dp[2]))

    def inverse(self):
        return SE2(-self.complex_to_rot(1/self.z).dot(self.d), 1/self.z)

    @property
    def I(self):
        return self.inverse()

    def apply(self, x):
        rot = self.complex_to_rot(self.z)

        if type(x) is SE2:
            return SE2(self.d+rot.dot(x.d), self.z*x.z)
        elif x.shape == (2,):
            return np.dot(rot, x) + self.d
        elif x.ndim == 2 and x.shape[1] == 2:
            return dot_m_vs(rot, x) + self.d
        else:
            raise NotImplemented('unknown operand')

    def __mul__(self, other):
        return self.apply(other)

    @staticmethod
    def complex_to_rot(x):
        return np.array([
            [x.real, -x.imag],
            [x.imag,  x.real]])


class SE3(object):
    """
    Lie Group SE(3): translation and rotation in 3d space

    T(x)=R(self.q)x+self.d
    """
    
    def __init__(self, d, q):
        """ You shouldn't call this directly. """
        self.d = d
        self.q = q/abs(q)

    @staticmethod
    def identity():
        return SE3(np.zeros(3), Quaternion(1, np.zeros(3)))

    @staticmethod
    def rotate(axis, angle):
        return SE3(np.zeros(3), Quaternion(math.cos(angle*0.5), axis*math.sin(angle*0.5)))

    @staticmethod
    def translate(d):
        return SE3(d, Quaternion(1, np.zeros(3)))

    @staticmethod
    def jacobian(p):
        """
        Return Jacobian: dp -> d(SE3.small(dp).apply(p))
        Actually dp is complex 6-dim vector, but you can forget that.
        """
        if p.shape == (3,):
            x, y, z = p
            return np.array([
                [1,0,0,  0, z,-y],
                [0,1,0, -z, 0, x],
                [0,0,1,  y,-x, 0]])
        else:
            ret = np.zeros([len(p), 3, 6])
            ret[:,:,:3] = np.identity(3)
            ret[:,0,4] =  p[:,2]
            ret[:,0,5] = -p[:,1]
            ret[:,1,3] = -p[:,2]
            ret[:,1,5] =  p[:,0]
            ret[:,2,3] =  p[:,1]
            ret[:,2,4] = -p[:,0]
            return ret

    @staticmethod
    def small(dp):
        """ See help(SE3.jacobian). """
        assert(dp.shape==(6,))
        return SE3(dp[:3], Quaternion(1, dp[3:]*0.5))

    def extract_rotation(self):
        """ Return rotation component. (unlike translation, rotation is always well-defined) """
        return SE3(np.zeros(3), self.q)

    def to_matrix(self):
        """ Return 4x4 matrix that works on homegeneous coordinate. """
        m = np.identity(4)
        m[:3,:3] = self.q.to_rot()
        m[:3,3] = self.d
        return m
    
    def inverse(self):
        qc = self.q.conjugate()
        return SE3(-qc.to_rot().dot(self.d), qc)

    @property
    def I(self):
        return self.inverse()

    def apply(self, x):
        rot = self.q.to_rot()

        if type(x) is SE3:
            return SE3(self.d+rot.dot(x.d), self.q*x.q)
        elif x.shape == (3,):
            return np.dot(rot, x) + self.d
        elif x.ndim == 2 and x.shape[1] == 3:
            return dot_m_vs(rot, x) + self.d
        else:
            raise NotImplemented('unknown operand')

    def __mul__(self, other):
        return self.apply(other)
    
    def __repr__(self):
        return 'SE3(distance=%f, angle=%f)'%(
            math.sqrt((self.d**2).sum()),
            math.degrees(math.acos(self.q.s)*2))



class Quaternion(object):
    def __init__(self, s, v):
        self.s = s
        self.v = v

    def __abs__(self):
        return math.sqrt(self.s**2 + (self.v**2).sum())

    def __truediv__(self, other):
        return Quaternion(self.s/other, self.v/other)

    def __mul__(self, other):
        if type(other) is Quaternion:
            return Quaternion(
                self.s*other.s - np.dot(self.v,other.v),
                self.s*other.v + self.v*other.s + np.cross(self.v, other.v))
        else:
            return Quaternion(self.s*other, self.v*other)

    def conjugate(self):
        return Quaternion(self.s, -self.v)

    def to_rot(self):
        """
        Return 3x3 rotation matrix corresponding to the quaternion.
        This is only valid when abs(self)==1
        """
        vx, vy, vz = self.v

        m = np.identity(3) * (self.s**2-(self.v**2).sum())
        m += 2 * self.s * np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]])
        m += 2 * np.outer(self.v, self.v)

        return m


# utilitiy function
def dot_m_vs(mat, vecs):
    return np.einsum('ij,sj->si', mat, vecs)
