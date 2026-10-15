import math
import numpy as np
import pytest
import torch
from resize_right_optics import resize, Gaussian, IdealLowpass, Airy, GaussianCovariance, Camera


def signal(n, modes, camera):
    x=np.arange(n)-(n-1)/2
    return sum(a*camera(np.array(2*np.pi*k/n))*np.cos(2*np.pi*k/n*x+phase)
               for k,a,phase in modes)


@pytest.mark.parametrize("n,s",[(48,.5),(48,2/3),(40,1.5),(21,2),(40,.75),(60,.4)])
@pytest.mark.parametrize("cam",[Gaussian(.6),IdealLowpass(2.7)])
def test_physical_capture_independent_oracle(n,s,cam):
    modes=[(1,.8,.1),(4,.3,.7),(9,.2,.2)]
    x=signal(n,modes,cam)
    m=round(n*s); u=(np.arange(m)-(m-1)/2)/s
    expected=sum(a*cam(np.array(2*np.pi*k/n/s))*np.cos(2*np.pi*k/n*u+phase)
                 for k,a,phase in modes if abs(2*np.pi*k/n/s)<np.pi)
    y=resize(x,s,camera=cam,method='fourier')
    np.testing.assert_allclose(y,expected,atol=3e-13)


@pytest.mark.parametrize("s,t,n",[(.5,.5,64),(2/3,.75,48),(1.5,2,32),(2,1.5,32),
                                   (1.5,2/3,48),(.75,4/3,48)])
def test_composition_on_retained_modes(s,t,n):
    cam=Gaussian(.5); x=signal(n,[(1,1,.3),(3,.4,.7)],cam)
    a=resize(resize(x,s,camera=cam,method='fourier'),t,camera=cam,method='fourier')
    b=resize(x,s*t,camera=cam,method='fourier')
    np.testing.assert_allclose(a,b,atol=5e-13)


def test_down_up_is_projection_not_identity():
    cam=Gaussian(.5); modes=[(1,1,.1),(5,.4,.2),(17,.2,.3)]
    x=signal(48,modes,cam)
    y=resize(resize(x,.5,camera=cam,method='fourier'),2,camera=cam,method='fourier')
    expected=signal(48,modes[:2],cam)
    np.testing.assert_allclose(y,expected,atol=3e-13)
    assert np.max(abs(y-x))>.05


def test_up_down_preserves_all_visible_modes():
    cam=Gaussian(.5); x=signal(48,[(1,1,.1),(17,.2,.3),(23,.3,.8)],cam)
    y=resize(resize(x,1.5,camera=cam,method='fourier'),2/3,camera=cam,method='fourier')
    np.testing.assert_allclose(y,x,atol=3e-13)


def test_commutativity_with_safe_intermediate_bands():
    cam=Gaussian(.5); x=signal(48,[(1,1,.1),(3,.2,.3)],cam)
    a=resize(resize(x,.5,camera=cam,method='fourier'),1.5,camera=cam,method='fourier')
    b=resize(resize(x,1.5,camera=cam,method='fourier'),.5,camera=cam,method='fourier')
    np.testing.assert_allclose(a,b,atol=4e-13)


@pytest.mark.parametrize("cam",[Airy(2.8),GaussianCovariance([[.4,.1],[.1,.6]])])
def test_nd_capture_and_composition(cam):
    n=(24,32); scales=(.5,.75)
    x,y=np.meshgrid(np.arange(n[0])-(n[0]-1)/2,np.arange(n[1])-(n[1]-1)/2,indexing='ij')
    w=np.array([2*np.pi/n[0],4*np.pi/n[1]])
    source=cam(w)*np.cos(w[0]*x+w[1]*y+.3)
    m=(12,24)
    u,v=np.meshgrid((np.arange(m[0])-(m[0]-1)/2)/scales[0],
                    (np.arange(m[1])-(m[1]-1)/2)/scales[1],indexing='ij')
    expected=cam(w/np.array(scales))*np.cos(w[0]*u+w[1]*v+.3)
    got=resize(source,scales,camera=cam,method='fourier')
    np.testing.assert_allclose(got,expected,atol=3e-13)
    back=resize(got,[2,4/3],camera=cam,method='fourier')
    np.testing.assert_allclose(back,source,atol=5e-13)


@pytest.mark.parametrize("s",[math.sqrt(2),math.pi/4,.51,1.01])
def test_arbitrary_scale_fourier_sum_oracle(s):
    n=13; cam=Gaussian(.4)
    x=signal(n,[(1,1,.1),(2,.4,.3)],cam)
    m=math.ceil(n*s); u=(np.arange(m)-(m-1)/2)/s
    expected=sum(a*cam(np.array(2*np.pi*k/n/s))*np.cos(2*np.pi*k/n*u+p)
                 for k,a,p in [(1,1,.1),(2,.4,.3)])
    np.testing.assert_allclose(resize(x,s,camera=cam,method='fourier'),expected,atol=1e-13)


def test_nyquist_is_explicitly_excluded():
    x=(-1.)**np.arange(16)
    y=resize(x,2,method='fourier')
    np.testing.assert_allclose(y,0,atol=1e-15)


def test_regularization_is_not_claimed_exact():
    x=signal(48,[(1,1,.1),(17,.4,.2)],Gaussian(.7))
    a=resize(resize(x,2,camera=Gaussian(.7),method='fourier',regularization=.05),
             .5,camera=Gaussian(.7),method='fourier',regularization=.05)
    assert np.max(abs(a-x))>.005

