import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from resize_right_optics import (resize, Gaussian, GaussianCovariance, Airy,
                                from_psf, from_function, from_kernel)


@pytest.mark.parametrize("method,conv",[("spatial",False),("spatial",True),("fourier",False)])
def test_image_and_gaussian_gradients(method,conv):
    x=torch.randn(7,dtype=torch.double,requires_grad=True)
    sigma=torch.tensor(.63,dtype=torch.double,requires_grad=True)
    def f(x,sigma):
        return resize(x,2/3,camera=Gaussian(sigma),method=method,by_convs=conv,
                      support_sz=8,density=4,pad_mode='reflect')
    assert gradcheck(f,(x,sigma),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize("conv",[False,True])
def test_learned_sampled_psf(conv):
    x=torch.randn(9,dtype=torch.double,requires_grad=True)
    psf=torch.tensor([.1,.7,1.,.8,.2],dtype=torch.double,requires_grad=True)
    def f(x,p):
        return resize(x,.5,camera=from_psf(p,spacing=.25),support_sz=8,density=4,by_convs=conv)
    assert gradcheck(f,(x,psf),atol=1e-5,rtol=1e-4)


def test_continuous_psf_parameters():
    x=torch.randn(7,dtype=torch.double)
    sigma=torch.tensor(.7,dtype=torch.double,requires_grad=True)
    def f(sigma):
        cam=from_function(lambda u: torch.exp(-u**2/(2*sigma**2)),support=6,spacing=.25)
        return resize(x,.6,camera=cam,support_sz=6,density=3)
    assert gradcheck(f,(sigma,),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize("conv",[False,True])
def test_nonseparable_covariance_gradients(conv):
    x=torch.randn(3,4,dtype=torch.double,requires_grad=True)
    # Learn a factor, so the covariance stays positive definite.
    L=torch.tensor([[.6,0.],[.1,.7]],dtype=torch.double,requires_grad=True)
    def f(x,L):
        return resize(x,[.5,1.5],camera=GaussianCovariance(L@L.T),support_sz=4,
                      density=2,period_factor=2,by_convs=conv)
    assert gradcheck(f,(x,L),atol=1e-5,rtol=1e-4)


def test_airy_cutoff_gradient():
    x=torch.randn(3,4,dtype=torch.double)
    cutoff=torch.tensor(2.71,dtype=torch.double,requires_grad=True)
    def f(c):
        return resize(x,.75,camera=Airy(c),support_sz=4,density=2,regularization=.01)
    assert gradcheck(f,(cutoff,),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize("method",['spatial','fourier'])
def test_scale_gradient_with_fixed_output_shape(method):
    x=torch.randn(7,dtype=torch.double)
    scale=torch.tensor(.731,dtype=torch.double,requires_grad=True)
    def f(s):
        return resize(x,s,out_shape=[5],dims=[0],camera=Gaussian(.6),method=method,
                      support_sz=8,density=4)
    assert gradcheck(f,(scale,),atol=1e-5,rtol=1e-4)


def test_repeated_training_steps_no_cached_graph():
    x=torch.randn(8,dtype=torch.double)
    target=resize(x,.75,camera=Gaussian(.9))
    log_sigma=torch.tensor(-.7,dtype=torch.double,requires_grad=True)
    optimizer=torch.optim.Adam([log_sigma],lr=.05)
    losses=[]
    # Reuse the camera object, but its callback recomputes the parameter.
    cam=from_function(lambda u: torch.exp(-u**2/(2*log_sigma.exp()**2)),support=8)
    for _ in range(15):
        optimizer.zero_grad()
        y=resize(x,.75,camera=cam)
        loss=(y-target).square().mean()
        loss.backward(); optimizer.step()
        assert torch.isfinite(log_sigma.grad)
        losses.append(loss.item())
    assert losses[-1]<losses[0]


def test_reference_kernel_is_differentiable():
    x=torch.randn(6,dtype=torch.double)
    values=torch.tensor([.2,1.,.3],dtype=torch.double,requires_grad=True)
    def f(v):
        return resize(x,.75,camera=from_kernel(v,.5,spacing=.5,terms=8),support_sz=6,density=2)
    assert gradcheck(f,(values,),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_dtype_preserved(dtype):
    x=torch.randn(7,dtype=dtype,requires_grad=True)
    y=resize(x,1.5,camera=Gaussian(.4))
    assert y.dtype==dtype
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
@pytest.mark.parametrize("conv",[False,True])
@pytest.mark.parametrize("camera",[Gaussian(.5),GaussianCovariance([[.5,.1],[.1,.6]])])
def test_mps_forward_backward(conv,camera):
    x=torch.randn(7,9,dtype=torch.float32)
    options=dict(camera=camera,support_sz=6,density=2,by_convs=conv)
    expected=resize(x,.75,**options)
    gpu=x.to('mps').requires_grad_()
    got=resize(gpu,.75,**options)
    torch.testing.assert_close(got.cpu(),expected,atol=2e-5,rtol=2e-5)
    got.square().mean().backward()
    assert torch.isfinite(gpu.grad).all()

