import math
import numpy as np
import pytest
import torch
from resize_right_optics import (resize, Gaussian, Airy, GaussianCovariance,
                                Separable, Camera, from_psf, Cubic)


@pytest.mark.parametrize("s",[.25,.5,2/3,.8,1,1.25,1.5,2,3,2.5])
@pytest.mark.parametrize("n",[1,2,7,12,17])
@pytest.mark.parametrize("mode",["constant","reflect","symmetric","wrap","edge"])
def test_polyphase_1d_numpy(s,n,mode):
    x=np.random.default_rng(n).normal(size=(2,n,3))
    args=dict(scale_factors=s,dims=[1],camera=Gaussian(.6),pad_mode=mode,support_sz=12,density=4)
    a=resize(x,**args)
    b=resize(x,by_convs=True,**args)
    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize("s,m",[(2/3,4),(1.5,19),(.5,1),(1,4),(3,2),(.25,9)])
@pytest.mark.parametrize("mode",["constant","reflect","symmetric","wrap","edge"])
def test_polyphase_torch_and_numpy(s,m,mode):
    x=np.random.default_rng(0).normal(size=(3,9))
    args=dict(scale_factors=s,out_shape=[m],dims=[1],camera=Gaussian(.6),pad_mode=mode,
              support_sz=10,density=4)
    n=resize(x,**args)
    for conv in (False,True):
        t=resize(torch.tensor(x),by_convs=conv,**args).numpy()
        np.testing.assert_allclose(t,n,atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize("mode",["constant","reflect","symmetric","wrap","edge"])
@pytest.mark.parametrize("scales",[(.5,2/3),(1.5,1.25),(2,1),(1,.5)])
def test_full_nd_polyphase_and_backends(mode,scales):
    x=np.random.default_rng(10).normal(size=(2,7,9))
    options=dict(scale_factors=scales,dims=[1,2],camera=GaussianCovariance([[.6,.13],[.13,.8]]),
                 support_sz=6,density=3,pad_mode=mode,chunk_size=7)
    expected=resize(x,**options)
    for v in (x,torch.tensor(x)):
        for conv in (False,True):
            y=resize(v,by_convs=conv,**options)
            if isinstance(y,torch.Tensor): y=y.numpy()
            np.testing.assert_allclose(y,expected,atol=3e-12,rtol=3e-12)


@pytest.mark.parametrize("ndim",[2,3,4])
def test_separable_equals_general_nd(ndim):
    sigma=.45
    c=GaussianCovariance(np.eye(ndim)*sigma**2)
    x=np.random.default_rng(4).normal(size=(3,)*ndim)
    args=dict(scale_factors=[.5]*ndim,dims=list(range(ndim)),support_sz=4,density=2,
              period_factor=2,pad_mode='wrap',max_kernel_points=2**21)
    sep=resize(x,camera=Gaussian(sigma),**args)
    for conv in (False,True):
        full=resize(x,camera=c,by_convs=conv,**args)
        np.testing.assert_allclose(full,sep,atol=1e-12,rtol=1e-12)


def test_separable_different_cameras_and_axes():
    x=np.random.default_rng(7).normal(size=(8,3,7))
    a=resize(x,[.5,1.5],dims=[2,0],camera=Separable(Gaussian(.4),Gaussian(.9)))
    b=resize(resize(x,.5,dims=[2],camera=Gaussian(.4)),1.5,dims=[0],camera=Gaussian(.9))
    np.testing.assert_allclose(a,b,atol=1e-13)


def test_nonseparable_is_not_axiswise():
    x=np.zeros((15,15)); x[7,7]=1
    a=resize(x,.8,camera=GaussianCovariance([[1,.7],[.7,1]]))
    b=resize(x,.8,camera=Gaussian())
    assert np.max(abs(a-b))>.01


@pytest.mark.parametrize("cam",[Airy(),Cubic(),from_psf(np.array([.1,.2,.5,.2]))])
def test_other_cameras(cam):
    x=np.random.default_rng(0).normal(size=(7,9))
    a=resize(x,.75,camera=cam)
    b=resize(torch.tensor(x),.75,camera=cam,by_convs=True).numpy()
    np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)


def test_irrational_fallback_does_not_snap():
    x=np.random.default_rng(0).normal(size=11)
    a=resize(x,math.sqrt(2))
    with pytest.warns(RuntimeWarning,match='without changing'):
        b=resize(x,math.sqrt(2),by_convs=True)
    np.testing.assert_array_equal(a,b)

