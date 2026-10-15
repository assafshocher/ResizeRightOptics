import numpy as np
import pytest
import torch
from resize_right_optics import (Gaussian, Airy, GaussianCovariance, from_psf,
                                from_function, Cubic, from_kernel_spectrum, resize, kernel_at)
from resize_right_optics.cameras import keys
from resize_right_optics.kernels import transfer


def test_sampled_psf_fourier_convention():
    psf=np.array([.1,.3,.6])
    w=np.linspace(-3,3,53)
    h=from_psf(psf,spacing=.5,center=[1.2])(w)
    expected=sum(v*np.exp(-1j*w*(j-1.2)*.5) for j,v in enumerate(psf))
    np.testing.assert_allclose(h,expected,atol=4e-16)
    np.testing.assert_allclose(h,h[::-1].conj(),atol=6e-16)


def test_continuous_psf_quadrature_against_analytic():
    gaussian=from_function(lambda x: np.exp(-x*x/(2*.7**2)),support=10,spacing=.05)
    w=np.linspace(-np.pi,np.pi,33)
    np.testing.assert_allclose(gaussian(w),Gaussian(.7)(w),atol=2e-12)


def test_nd_sampled_psf_and_offset():
    psf=np.array([[.1,.2,.1],[.05,.3,.25]])
    w=np.array([[.2,.5],[.4,-.6]])
    camera=from_psf(psf,spacing=[.5,.25])
    expected=sum(psf[i,j]*np.exp(-1j*(w[:,0]*(i-.5)*.5+w[:,1]*(j-1)*.25))
                 for i in range(2) for j in range(3))
    np.testing.assert_allclose(camera(w),expected,atol=2e-16)


def test_airy_endpoints_and_radial_symmetry():
    c=Airy(2.7)
    w=np.array([[0.,0.],[2.7,0.],[3.,0.],[1.,1.],[np.sqrt(2),0.]])
    v=c(w)
    np.testing.assert_allclose(v[:3],[1,0,0],atol=1e-15)
    np.testing.assert_allclose(v[3],v[4],atol=2e-16)


def test_cubic_anchor_ratio_against_spatial_quadrature():
    r=.5; cam=Cubic(r)
    w=np.linspace(-1.4,1.4,35)
    x=np.linspace(-2,2,10001)
    trapezoid=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    u=trapezoid(keys(x)[None,:]*np.cos(w[:,None]*x[None,:]/r),x,axis=1)
    np.testing.assert_allclose(cam(w/r)/cam(w),u,atol=3e-13)


def test_gaussian_anchor_recovers_camera():
    r=.6; sigma=.7
    anchor=lambda w: np.exp(-.5*sigma**2*(1/r**2-1)*w*w)
    cam=from_kernel_spectrum(anchor,r,terms=32)
    w=np.linspace(-3,3,49)
    np.testing.assert_allclose(cam(w),Gaussian(sigma)(w),atol=1e-14)


def test_kernel_identity_at_integer_offsets():
    k=kernel_at(Gaussian(.7),1,np.zeros(1),support=32,density=8)
    x=np.arange(-15,16,dtype=float)
    np.testing.assert_allclose(k(x),x==0,atol=3e-15)


def test_kernel_offsets_are_continuous_and_usable_in_torch():
    x=torch.linspace(-3,3,41,dtype=torch.double,requires_grad=True)
    k=kernel_at(Gaussian(.6),.75,x,support=12)
    k(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert abs(float(k(torch.tensor([0.],dtype=torch.double)))-.75)>.01


def test_interior_zeros_are_projected_not_divided():
    from resize_right_optics import Camera
    cam=Camera(lambda w: np.where(abs(w)<.2,1.,0.))
    w=np.linspace(-3,3,61)
    v=transfer(cam,w,[2.])
    assert np.isfinite(v).all()
    assert (v[abs(w)>.2]==0).all()
