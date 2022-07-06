#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from grpc import dynamic_ssl_server_credentials
import numpy as np

import paddle
from .. import Layer
from .. import functional as F
__all__ = []

class PairwiseDistance(Layer):
    r"""
    This operator computes the pairwise distance between two vectors. The
    distance is calculated by p-oreder norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        p (float): The order of norm. The default value is 2.
        epsilon (float, optional): Add small value to avoid division by zero,
            default value is 1e-6.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``'x-y'`` unless :attr:`keepdim` is True, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: :math:`[N, D]` where `D` is the dimension of vector, available dtype
            is float32, float64.
        y: :math:`[N, D]`, y have the same shape and dtype as x.
        out: :math:`[N]`. If :attr:`keepdim` is ``True``, the out shape is :math:`[N, 1]`.
            The same dtype as input tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            paddle.disable_static()
            x_np = np.array([[1., 3.], [3., 5.]]).astype(np.float64)
            y_np = np.array([[5., 6.], [7., 8.]]).astype(np.float64)
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            dist = paddle.nn.PairwiseDistance()
            distance = dist(x, y)
            print(distance.numpy()) # [5. 5.]

    """

    def __init__(self, p=2., epsilon=1e-6, keepdim=False, name=None):
        super(PairwiseDistance, self).__init__()
        self.p = p
        self.epsilon = epsilon
        self.keepdim = keepdim
        self.name = name

    def forward(self, x, y):
        
        return F.pairwise_distance(x, y, self.p, self.epsilon, self.keepdim, self.name)

    def extra_repr(self):
        main_str = 'p={p}'
        if self.epsilon != 1e-6:
            main_str += ', epsilon={epsilon}'
        if self.keepdim != False:
            main_str += ', keepdim={keepdim}'
        if self.name != None:
            main_str += ', name={name}'
        return main_str.format(**self.__dict__)


#单测
import unittest
import paddle
def call_pairwise_distance_layer(x, y, p=2., epsilon=1e-6, keepdim='False', name='name'):
    pairwise_distance = paddle.nn.PairwiseDistance(
        p=p, 
        epsilon=epsilon, 
        keepdim=keepdim, 
        name=name)
    distance = pairwise_distance(x=x, y=y)
    return distance

def call_pairwise_distance_functional(x, y, p=2., epsilon=1e-6, keepdim='False', name='name'):
    distance = paddle.nn.functional.pairwise_distance(
        x=x,
        y=y,
        p=p,
        epsilon=epsilon,
        keepdim=keepdim,
        name=name)
    return distance

def test_static(place,
                x_np,
                y_np,
                p=2,
                epsilon=1e-6,
                keepdim=False,
                name='name',
                functional=False):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        x = paddle.static.data(name='x',
                               shape=x_np.shape,
                               dtype='float64')
        y = paddle.static.data(name='y',
                               shape=y_np.shape,
                               dtype='float64')   
        feed_dict = {
            "x": x,
            "y": y,   
        }   
        if functional:
            distance = call_pairwise_distance_functional(
                x=x,
                y=y,
                p=p,
                epsilon=epsilon,
                keepdim=keepdim,
                name=name)                     
        else:
            distance = call_pairwise_distance_layer(
                x=x,
                y=y,
                p=p,
                epsilon=epsilon,
                keepdim=keepdim,
                name=name)
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[distance])
    return static_result

def test_dygraph(place,
                 x,
                 y,
                 p=2,
                 epsilon=1e-6,
                 keepdim=False,
                 name='name',
                 functional=False):
    paddle.disable_static()
    x = paddle.to_tensor(x)
    y = paddle.to_tensor(y)

    if functional:
        dy_distance = call_pairwise_distance_functional(
            x=x,
            y=y,
            p=p,
            epsilon=epsilon,
            keepdim=keepdim,
            name=name)
    else:
        dy_distance = call_pairwise_distance_layer(
            x=x,
            y=y,
            p=p,
            epsilon=epsilon,
            keepdim=keepdim,
            name=name)
    dy_distance = dy_distance.numpy()
    paddle.enable_static()
    return dy_distance

def cala_pairwise_distance(x, y, p=2.0, epsilon=1e-6, keepdim=False, name='name'):
    
    distance = np.linalg.norm(x, y, p=p, epsilon=epsilon, keepdim=keepdim)

    return distance

class TestPairwiseDistance(unittest.TestCase):
    def test_pairwisedistance(self):
        shape = (3, 3)
        x = np.random.uniform(0, 1, size=shape).astype(np.float64)
        y = np.random.uniform(1, 2, size=shape).astype(np.float64)

        places = [paddle.CPUPlace()]
        keepdims = ['True', 'False']
        p_list = [0, 1, 2, 'inf', '-inf']
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            for p in p_list:
                for keepdim in keepdims:
                    expected = cala_pairwise_distance(
                        x=x,
                        y=y,
                        p=p,
                        keepdim=keepdim)

                    dy_distance = test_dygraph(
                        place=place,
                        x=x,
                        y=y,
                        p=p,
                        keepdim=keepdim
                    )

                    static_distance = test_static(
                        place=place,
                        x=x,
                        y=y,
                        p=p,
                        keepdim=keepdim)
                    self.assertTrue(np.allclose(static_distance, expected))
                    self.assertTrue(np.allclose(static_distance, dy_distance))
                    self.assertTrue(np.allclose(dy_distance, expected))

                    static_functional = test_static(
                        place=place,
                        x=x,
                        y=y,
                        p=p,
                        keepdim=keepdim,
                        functional=True)
                    dy_functional = test_dygraph(
                        place=place,
                        x=x,
                        y=y,
                        p=p,
                        keepdim=keepdim,
                        functional=True)
                self.assertTrue(np.allclose(static_functional, expected))
                self.assertTrue(np.allclose(static_functional, dy_functional))
                self.assertTrue(np.allclose(dy_functional, expected))
    def test_pairwise_distance_error(self):

        paddle.disable_static()
        self.assertRaises(ValueError,
                        paddle.nn.PairwiseDistance,
                        keepdim="unsupport keepdim")
        x = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        y = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.triplet_margin_with_distance_loss,
            x=x,
            y=y,
            keepdim="unsupport keepdim")
        
        self.assertRaises(ValueError,
                        paddle.nn.PairwiseDistance,
                        p="unsupport keepdim")
        x = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        y = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.triplet_margin_with_distance_loss,
            x=x,
            y=y,
            p="unsupport keepdim")
        paddle.enable_static()

    # def test_pairwise_distance_dimension(self):
    #     paddle.disable_static()

    #     x = paddle.to_tensor(np.random.randn(3, 4))
    #     y = paddle.to_tensor(np.random.randn(4, ))
    #     test_dygraph(x, y)
