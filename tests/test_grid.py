import math
import numpy as np
import pytest
import torch
from resize_right_optics import resize, projected_grid, Camera, Gaussian
from resize_right_optics.resize_right import _pad_indices, _spatial_1d


class Triangle:
    support = (2,)
    def __call__(self, x):
        return (1 - abs(x)).clip(0, 1)


@pytest.mark.parametrize("n,m,s", [(1,1,1),(1,5,3),(2,3,1.5),(5,8,1.6),
                                   (6,9,1.5),(7,3,.4),(10,7,.66),(9,6,1),
                                   (17,25,math.sqrt(2)),(13,9,2/3)])
def test_centered_grid(n,m,s):
    g=projected_grid(n,m,s,np.zeros(1))
    np.testing.assert_allclose((g[0]+g[-1])/2,(n-1)/2,atol=1e-14)
    if m>1: np.testing.assert_allclose(np.diff(g),1/s,atol=5e-15)


@pytest.mark.parametrize("n,m,s",[(9,6,1),(7,4,.5),(5,9,2),(4,7,1.7),(1,4,2)])
@pytest.mark.parametrize("mode",["constant","reflect","symmetric","wrap","edge"])
def test_grid_against_independent_linear_oracle(n,m,s,mode):
    # Simple known weights, independent of camera/FFT construction. This
    # isolates the alignment question from the scale law.
    x=np.arange(n,dtype=float)**2
    y=_spatial_1d(x,0,m,s,Triangle(),mode,True,3)
    extended=np.pad(x,100,mode=mode if mode!='wrap' else 'wrap')
    points=[j/s+(n-1)/2-(m-1)/(2*s) for j in range(m)]
    expected=[]
    for p in points:
        j=math.floor(p); a=p-j
        expected.append((1-a)*extended[j+100]+a*extended[j+101])
    np.testing.assert_allclose(y,expected,atol=1e-13)


@pytest.mark.parametrize("mode",["constant","reflect","symmetric","wrap","edge"])
@pytest.mark.parametrize("n",[1,2,5])
def test_arbitrarily_large_padding(n,mode):
    x=np.arange(n,dtype=float)+1
    indices=np.arange(-37,n+37)
    mapped,valid=_pad_indices(indices,n,mode)
    np.testing.assert_array_equal(x[mapped]*valid,np.pad(x,37,mode=mode))


def test_explicit_scale_is_not_size_ratio():
    x=np.random.default_rng(0).normal(size=11)
    a=resize(x,.5,out_shape=[6],dims=[0])
    b=resize(x,out_shape=[6],dims=[0])
    assert np.max(abs(a-b))>.01


def test_default_axes_and_explicit_permutation():
    x=np.random.default_rng(2).normal(size=(7,9,3))
    n=resize(x,.6,camera=Gaussian(.5))
    t=resize(torch.tensor(x.transpose(2,0,1)),.6,camera=Gaussian(.5)).numpy().transpose(1,2,0)
    assert n.shape==(5,6,3)
    np.testing.assert_allclose(n,t,atol=1e-13)
    a=resize(x,[.7,1.5],dims=[1,0])
    b=resize(x.transpose(1,0,2),[.7,1.5],dims=[0,1]).transpose(1,0,2)
    np.testing.assert_allclose(a,b,atol=1e-13)


def test_identity_crop_and_even_odd():
    x=np.arange(9,dtype=float)
    # Unit scale with a different output shape is a centered crop, not no-op.
    y=resize(x,1,out_shape=[5],dims=[0])
    np.testing.assert_allclose(y,x[2:7],atol=1e-13)
    z=resize(x,1,out_shape=[6],dims=[0])
    assert z.shape==(6,)

