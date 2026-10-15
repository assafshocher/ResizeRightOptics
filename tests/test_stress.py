import math
import subprocess
import sys
import numpy as np
import pytest
import torch
from resize_right_optics import *


@pytest.mark.parametrize('seed',range(100))
def test_randomized_polyphase_and_grid(seed):
    rng=np.random.default_rng(seed)
    ndim=int(rng.integers(1,5));shape=tuple(int(x) for x in rng.integers(1,11,size=ndim))
    count=int(rng.integers(1,ndim+1));dims=rng.choice(ndim,size=count,replace=False).tolist()
    scales=[float(rng.choice([.25,.4,.5,.6,2/3,.75,1,1.25,1.5,2,3])) for _ in dims]
    sizes=[max(1,math.ceil(shape[d]*s)+int(rng.integers(-1,2))) for d,s in zip(dims,scales)]
    mode=str(rng.choice(['constant','reflect','symmetric','wrap','edge']))
    image=rng.normal(size=shape)
    kwargs=dict(scale_factors=scales,out_shape=sizes,dims=dims,camera=Gaussian(.45),
                support_sz=6,density=3,pad_mode=mode)
    normal=resize(image,**kwargs)
    conv=resize(image,by_convs=True,**kwargs)
    np.testing.assert_allclose(normal,conv,atol=4e-12,rtol=4e-12)
    t=resize(torch.tensor(image),by_convs=bool(seed%2),**kwargs).numpy()
    np.testing.assert_allclose(normal,t,atol=4e-12,rtol=4e-12)


@pytest.mark.parametrize('ndim',[2,3,4])
@pytest.mark.parametrize('conv',[False,True])
def test_general_nd_torch_numpy(ndim,conv):
    x=np.random.default_rng(ndim).normal(size=(2,)+(3,)*ndim)
    cov=np.eye(ndim)*.3+np.ones((ndim,ndim))*.02
    opts=dict(scale_factors=[.5]*ndim,dims=list(range(1,ndim+1)),camera=GaussianCovariance(cov),
              support_sz=3,density=1,period_factor=2,by_convs=conv,pad_mode='reflect')
    a=resize(x,**opts);t=torch.tensor(x,requires_grad=True)
    b=resize(t,**opts)
    np.testing.assert_allclose(a,b.detach().numpy(),atol=3e-12,rtol=3e-12)
    b.sum().backward()
    assert torch.isfinite(t.grad).all()


def test_numpy_works_without_importable_torch():
    code="import sys; sys.modules['torch']=None; import numpy as np; from resize_right_optics import resize; assert resize(np.ones((3,4)),.5).shape==(2,2)"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_recover_gaussian_from_upscale_anchor():
    s=2;sigma=.5
    anchor=lambda w:np.exp(.5*sigma**2*(1-1/s**2)*w*w)
    cam=from_kernel_spectrum(anchor,s)
    w=np.linspace(-3,3,41)
    np.testing.assert_allclose(cam(w),Gaussian(sigma)(w),atol=5e-15)


def test_same_separable_camera_full_path():
    camera=Separable(Gaussian(.4),Gaussian(.7))
    x=np.random.default_rng(1).normal(size=(7,8))
    options=dict(scale_factors=[.75,.5],support_sz=6,density=2)
    np.testing.assert_allclose(resize(x,camera=camera,**options),
                               resize(x,camera=camera.as_full(),**options),atol=2e-13)


def test_constant_normalization_all_paths():
    x=np.ones((5,7))
    for cam in [Gaussian(.7),Airy(2.7),Cubic()]:
        for conv in [False,True]:
            y=resize(x,[.6,.75],camera=cam,by_convs=conv,support_sz=6,density=2)
            np.testing.assert_allclose(y,1,atol=1e-13)


def test_explicit_linear_lut_option():
    x=np.random.default_rng(0).normal(size=11)
    a=resize(x,.6,table_interp='linear')
    b=resize(x,.6,table_interp='linear',by_convs=True)
    np.testing.assert_allclose(a,b,atol=3e-13)


def test_invalid_camera_parameters():
    for cam in [Gaussian(0),Gaussian(-1),Airy(0),Airy(-1),from_kernel_spectrum(lambda w: w*0+.8,.5)]:
        with pytest.raises(ValueError): resize(np.ones((3,4)),.5,camera=cam)
    with pytest.raises(TypeError,match='real'):
        from_psf(np.ones(3,dtype=complex))

