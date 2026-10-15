"""Accuracy tests deliberately include finite-implementation limitations."""
import math
import numpy as np
import pytest
import torch
from resize_right_optics import *


@pytest.mark.parametrize("s,t,n",[(.5,.5,256),(2/3,.75,240),(1.5,2/3,256),(.5,2,256)])
@pytest.mark.parametrize("camera",[Gaussian(.5),Cubic()])
def test_finite_kernel_consistency(s,t,n,camera):
    x=np.cos(2*np.pi*3/n*(np.arange(n)-(n-1)/2))
    options=dict(camera=camera,pad_mode='wrap',support_sz=64,density=16,period_factor=8)
    chain=resize(resize(x,s,**options),t,**options)
    direct=resize(x,s*t,**options)
    assert np.max(abs(chain-direct))<.002


def test_finite_kernel_convergence_cubic():
    n=256;x=np.cos(2*np.pi*5/n*(np.arange(n)-(n-1)/2))
    errors=[]
    for width in [16,32,64]:
        options=dict(camera=Cubic(),pad_mode='wrap',support_sz=width,density=16,period_factor=8)
        y=resize(resize(x,.5,**options),2,**options)
        errors.append(np.max(abs(y-x)))
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<1.2e-4


def test_plain_truncation_is_not_mistaken_for_exactness():
    n=256;x=np.cos(2*np.pi*5/n*(np.arange(n)-(n-1)/2))
    opts=dict(camera=Gaussian(1),pad_mode='wrap',support_sz=32,density=16,period_factor=8)
    errors=[]
    for window in [None,'hann']:
        y=resize(resize(x,.5,window=window,**opts),2,window=window,**opts)
        errors.append(np.max(abs(y-x)))
    assert errors[1]<errors[0]/100


@pytest.mark.parametrize("n,m",[(7,14),(8,12),(9,6),(12,9),(12,8),(7,7),(8,8)])
def test_fft_fast_path_against_general_fourier_sum(n,m):
    x=np.random.default_rng(42).normal(size=n)
    scale=m/n;cam=Gaussian(.4)
    fast=resize(x,scale,out_shape=[m],camera=cam,method='fourier')
    # A differentiable scale selects the independent Fourier-sum execution.
    slow=resize(torch.tensor(x),torch.tensor(scale,dtype=torch.double,requires_grad=True),
                out_shape=[m],camera=cam,method='fourier').detach().numpy()
    np.testing.assert_allclose(fast,slow,atol=1e-13)


def test_asymmetric_camera_captures_phase_not_only_magnitude():
    cam=from_psf(np.array([.1,.2,.7]),spacing=.5)
    n=32;s=.75;m=24;w=2*np.pi*3/n
    x=(cam(np.array(w))*np.exp(1j*w*(np.arange(n)-(n-1)/2))).real
    expected=(cam(np.array(w/s))*np.exp(1j*w*(np.arange(m)-(m-1)/2)/s)).real
    np.testing.assert_allclose(resize(x,s,camera=cam,method='fourier'),expected,atol=2e-14)
    # A longer spatial approximation should respect the same handedness.
    approx=resize(x,s,camera=cam,pad_mode='wrap',support_sz=64,density=16)
    assert np.max(abs(approx-expected))<.003


def test_rounded_extent_is_not_an_exact_cycle():
    x=np.random.default_rng(7).normal(size=17)
    y=resize(resize(x,.7,method='fourier'),1/.7,out_shape=[17],method='fourier')
    # The intermediate samples do not cover the same periodic extent. This
    # is a documented finite-array issue, not an identity asserted by tests.
    assert np.max(abs(y-x))>.01


@pytest.mark.parametrize("cam",[Airy(2.7),GaussianCovariance([[.4,.1],[.1,.6]])])
def test_full_nd_fft_fast_against_sum(cam):
    x=np.random.default_rng(17).normal(size=(8,12))
    s=[.75,2/3];m=[6,8]
    fast=resize(x,s,camera=cam,method='fourier')
    slow=resize(torch.tensor(x),[torch.tensor(v,dtype=torch.double,requires_grad=True) for v in s],
                out_shape=m,camera=cam,method='fourier').detach().numpy()
    np.testing.assert_allclose(fast,slow,atol=2e-14)


@pytest.mark.parametrize("device",['cpu','mps'])
def test_torch_fourier_device(device):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('MPS unavailable')
    x=torch.randn(8,12,dtype=torch.float32,device=device,requires_grad=True)
    y=resize(x,[.75,2/3],camera=Airy(2.7),method='fourier')
    expected=resize(x.detach().cpu().numpy(),[.75,2/3],camera=Airy(2.7),method='fourier')
    np.testing.assert_allclose(y.detach().cpu().numpy(),expected,atol=3e-6)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_alias_free_nd_chain_to_machine_precision():
    n=48;x,y=np.meshgrid(np.arange(n)-(n-1)/2,np.arange(n)-(n-1)/2,indexing='ij')
    source=np.cos(2*np.pi*(x+y)/n)+.4*np.sin(2*np.pi*(3*x-y)/n)
    cam=Airy(2.7)
    chain=resize(resize(source,[.5,.75],camera=cam,method='fourier'),[2,4/3],camera=cam,method='fourier')
    np.testing.assert_allclose(chain,source,atol=3e-13)


def test_badly_conditioned_refinement_warns():
    with pytest.warns(RuntimeWarning,match='inverse gain'):
        resize(np.ones(16),2,camera=Gaussian(2),support_sz=8)


@pytest.mark.parametrize('scales',[(math.sqrt(2),1/math.sqrt(2)),(1/math.sqrt(2),math.sqrt(2))])
def test_irrational_spatial_cycle_away_from_boundaries(scales):
    n=257;u=np.arange(n)-(n-1)/2
    x=np.cos(.07*u)+.2*np.sin(.13*u)
    opts=dict(camera=Gaussian(.5),support_sz=64,density=16,pad_mode='reflect')
    y=resize(resize(x,scales[0],**opts),scales[1],out_shape=[n],**opts)
    # Compare only where neither stage consulted a boundary extension.
    assert np.max(abs(y[80:-80]-x[80:-80]))<.001
