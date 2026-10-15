import numpy as np
import pytest
from resize_right_optics import *


@pytest.mark.parametrize('scale',[(.5,.5),(1.5,.5),(.75,1.5)])
@pytest.mark.parametrize('camera',[Airy(2.7),GaussianCovariance([[.4,.1],[.1,.6]])])
def test_nd_mixed_chain_explicit_projector(scale,camera):
    shape=(24,32);s=np.array(scale)
    rng=np.random.default_rng(102)
    raw=rng.normal(size=shape)
    w=np.stack(np.meshgrid(*(2*np.pi*np.fft.fftfreq(n) for n in shape),indexing='ij'),axis=-1)
    source=(np.abs(w)<np.pi).all(-1)&(abs(camera(w))>1e-12)
    target=(np.abs(w/s)<np.pi).all(-1)&(abs(camera(w/s))>1e-12)
    image=np.fft.ifftn(np.fft.fftn(raw)*source).real
    expected=np.fft.ifftn(np.fft.fftn(image)*target).real
    scaled=resize(image,scale,camera=camera,method='fourier')
    got=resize(scaled,list(1/s),out_shape=shape,camera=camera,method='fourier')
    np.testing.assert_allclose(got,expected,atol=5e-13)


def test_separable_regularization_is_not_joint_tikhonov():
    # This is intentional and documented: regularizing each factor retains
    # separability, whereas a joint denominator generally does not.
    cam=Separable(Gaussian(.8),Gaussian(.8))
    x=np.random.default_rng(3).normal(size=(8,12))
    a=resize(x,1.5,camera=cam,method='fourier',regularization=.1)
    b=resize(x,1.5,camera=cam.as_full(),method='fourier',regularization=.1)
    assert np.max(abs(a-b))>.01


def test_regularized_inverse_reduces_noise_amplification():
    n=64;cam=Gaussian(1.)
    x=np.cos(2*np.pi*2/n*(np.arange(n)-(n-1)/2))
    noise=np.random.default_rng(4).normal(0,.01,size=n)
    clean=resize(x,2,camera=cam,method='fourier')
    exact=resize(x+noise,2,camera=cam,method='fourier')
    stable=resize(x+noise,2,camera=cam,method='fourier',regularization=.05)
    assert np.mean((stable-clean)**2)<np.mean((exact-clean)**2)/10

