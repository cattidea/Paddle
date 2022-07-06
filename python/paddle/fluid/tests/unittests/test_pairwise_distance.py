# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


def pairwise_distance(x, y, p=2.0, epsilon=1e-6, keepdim=False):
    return np.linalg.norm(x - y, ord=p, axis=1, keepdims=keepdim)


def test_static(x_np, y_np, p=2.0, epsilon=1e-6, keepdim=False):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()

    place = fluid.CUDAPlace(
        0) if paddle.fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    with paddle.static.program_guard(prog, startup_prog):
        x = paddle.fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
        y = paddle.fluid.data(name='y', shape=y_np.shape, dtype=x_np.dtype)
        dist = paddle.nn.layer.distance.PairwiseDistance(p=p,
                                                         epsilon=epsilon,
                                                         keepdim=keepdim)
        distance = dist(x, y)
        exe = paddle.static.Executor(place)
        static_ret = exe.run(prog,
                             feed={
                                 'x': x_np,
                                 'y': y_np
                             },
                             fetch_list=[distance])
        static_ret = static_ret[0]
    return static_ret


def test_dygraph(x_np, y_np, p=2.0, epsilon=1e-6, keepdim=False):
    paddle.disable_static()
    x = paddle.to_tensor(x_np)
    y = paddle.to_tensor(y_np)
    dist = paddle.nn.layer.distance.PairwiseDistance(p=p,
                                                     epsilon=epsilon,
                                                     keepdim=keepdim)
    distance = dist(x, y)
    dygraph_ret = distance.numpy()
    paddle.enable_static()
    return dygraph_ret


class TestPairwiseDistance(unittest.TestCase):

    def test_pairwise_distance(self):
        all_shape = [[100, 100], [4, 5, 6, 7]]
        dtypes = ['float32', 'float64']
        keeps = [False, True]
        for shape in all_shape:
            for dtype in dtypes:
                for keepdim in keeps:
                    x_np = np.random.random(shape).astype(dtype)
                    y_np = np.random.random(shape).astype(dtype)

                    static_ret = test_static(x_np, y_np, keepdim=keepdim)
                    dygraph_ret = test_dygraph(x_np, y_np, keepdim=keepdim)
                    excepted_value = pairwise_distance(x_np,
                                                       y_np,
                                                       keepdim=keepdim)

                    self.assertTrue(np.allclose(static_ret, dygraph_ret))
                    self.assertTrue(np.allclose(static_ret, excepted_value))
                    self.assertTrue(np.allclose(dygraph_ret, excepted_value))

    def test_pairwise_distance_broadcast(self):
        shape_x = [100, 100]
        shape_y = [100, 1]
        keepdim = False
        x_np = np.random.random(shape_x).astype('float32')
        y_np = np.random.random(shape_y).astype('float32')
        static_ret = test_static(x_np, y_np, keepdim=keepdim)
        dygraph_ret = test_dygraph(x_np, y_np, keepdim=keepdim)
        excepted_value = pairwise_distance(x_np, y_np, keepdim=keepdim)
        self.assertTrue(np.allclose(static_ret, dygraph_ret))
        self.assertTrue(np.allclose(static_ret, excepted_value))
        self.assertTrue(np.allclose(dygraph_ret, excepted_value))

    def test_pairwise_distance_different_p(self):
        shape = [100, 100]
        keepdim = False
        p = 3.0
        x_np = np.random.random(shape).astype('float32')
        y_np = np.random.random(shape).astype('float32')
        static_ret = test_static(x_np, y_np, p=p, keepdim=keepdim)
        dygraph_ret = test_dygraph(x_np, y_np, p=p, keepdim=keepdim)
        excepted_value = pairwise_distance(x_np, y_np, p=p, keepdim=keepdim)
        self.assertTrue(np.allclose(static_ret, dygraph_ret))
        self.assertTrue(np.allclose(static_ret, excepted_value))
        self.assertTrue(np.allclose(dygraph_ret, excepted_value))

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

if __name__ == "__main__":
    unittest.main()
