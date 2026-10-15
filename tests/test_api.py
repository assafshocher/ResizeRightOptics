import math
import numpy as np
import pytest
import torch
from resize_right_optics import *


@pytest.mark.parametrize("options",[
 {},{'scale_factors':0},{'scale_factors':-1},{'scale_factors':float('nan')},
 {'scale_factors':float('inf')},{'scale_factors':True},
 {'out_shape':[0]},{'out_shape':[2.5]},{'out_shape':[True]},
 {'scale_factors':[.5,.5],'dims':[0]}, {'scale_factors':2,'dims':[0,0]},
 {'scale_factors':2,'dims':[4]}, {'scale_factors':2,'dims':[]},
 {'scale_factors':2,'camera':Airy(),'dims':[0]},
 {'scale_factors':2,'by_convs':[True]},
 {'scale_factors':2,'method':'bad'}, {'scale_factors':2,'regularization':-1},
 {'scale_factors':2,'support_sz':0}, {'scale_factors':2,'density':0},
 {'scale_factors':2,'method':'fourier','by_convs':True},
 {'scale_factors':2,'camera':Airy(),'by_convs':[True,False]},
 {'scale_factors':2,'chunk_size':0}, {'scale_factors':2,'max_numerator':0},
 {'scale_factors':2,'pad_mode':'unknown'},
])
def test_bad_arguments(options):
    with pytest.raises((ValueError,TypeError)):
        resize(np.ones((3,4)),**options)


@pytest.mark.parametrize('x',[[],np.ones(3,dtype=int),np.ones(3,dtype=complex),np.ones((0,3)),np.array(1.)])
def test_bad_inputs(x):
    with pytest.raises((ValueError,TypeError)):
        resize(x,.5)


def test_no_antialiasing_switch():
    with pytest.raises(TypeError,match='antialiasing'):
        resize(np.ones(3),.5,antialiasing=True)


def test_memory_guards():
    with pytest.raises(ValueError,match='Kernel table needs'):
        resize(np.ones((5,5)),2,camera=Airy(),support_sz=50,max_kernel_points=100)
    with pytest.raises(ValueError,match='field needs'):
        resize(np.ones((5,5)),2,camera=Airy(),support_sz=4,density=2,max_work_elements=10)


def test_noncontiguous_and_readonly():
    x=np.random.default_rng(7).normal(size=(9,7))[::-1,::2]
    x.flags.writeable=False
    a=resize(x,.75)
    b=resize(x.copy(),.75)
    np.testing.assert_array_equal(a,b)
    t=torch.tensor(x.copy()).T
    torch.testing.assert_close(resize(t,.75),resize(t.contiguous(),.75))


def test_numpy_does_not_detach_learned_camera():
    with pytest.raises(TypeError,match='not detached'):
        resize(np.ones(7),.5,camera=Gaussian(torch.tensor(.6,requires_grad=True)))


def test_polyphase_rejects_learned_scale():
    with pytest.raises(ValueError,match='scale gradients'):
        resize(torch.ones(7),torch.tensor(.5,requires_grad=True),by_convs=True)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_numpy_dtype(dtype):
    x=np.arange(9,dtype=dtype)
    assert resize(x,.75).dtype==dtype

