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


def call_pairwise_distance_layer(x,
                                 y,
                                 p=2.,
                                 epsilon=1e-6,
                                 keepdim='False',
                                 name='name'):
    pairwise_distance = paddle.nn.PairwiseDistance(p=p,
                                                   epsilon=epsilon,
                                                   keepdim=keepdim,
                                                   name=name)
    distance = pairwise_distance(x=x, y=y)
    return distance


def call_pairwise_distance_functional(x,
                                      y,
                                      p=2.,
                                      epsilon=1e-6,
                                      keepdim='False',
                                      name='name'):
    distance = paddle.nn.functional.pairwise_distance(x=x,
                                                      y=y,
                                                      p=p,
                                                      epsilon=epsilon,
                                                      keepdim=keepdim,
                                                      name=name)
    return distance


def test_static(place,
                x_np,
                y_np,
                p=2.0,
                epsilon=1e-6,
                keepdim=False,
                functional=False):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()

    place = fluid.CUDAPlace(
        0) if paddle.fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    with paddle.static.program_guard(prog, startup_prog):
        x = paddle.fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
        y = paddle.fluid.data(name='y', shape=y_np.shape, dtype=x_np.dtype)

        if functional:
            distance = call_pairwise_distance_functional(x=x,
                                                         y=y,
                                                         p=p,
                                                         epsilon=epsilon,
                                                         keepdim=keepdim)
        else:
            distance = call_pairwise_distance_layer(x=x,
                                                    y=y,
                                                    p=p,
                                                    epsilon=epsilon,
                                                    keepdim=keepdim)
        exe = paddle.static.Executor(place)
        static_ret = exe.run(prog,
                             feed={
                                 'x': x_np,
                                 'y': y_np
                             },
                             fetch_list=[distance])
        static_ret = static_ret[0]
    return static_ret


def test_dygraph(place,
                 x_np,
                 y_np,
                 p=2.0,
                 epsilon=1e-6,
                 keepdim=False,
                 functional=False):
    paddle.disable_static()
    x = paddle.to_tensor(x_np)
    y = paddle.to_tensor(y_np)
    if functional:
        dy_distance = call_pairwise_distance_functional(x=x,
                                                        y=y,
                                                        p=p,
                                                        epsilon=epsilon,
                                                        keepdim=keepdim)
    else:
        dy_distance = call_pairwise_distance_layer(x=x,
                                                   y=y,
                                                   p=p,
                                                   epsilon=epsilon,
                                                   keepdim=keepdim)
    dygraph_ret = dy_distance.numpy()
    paddle.enable_static()
    return dygraph_ret


class TestPairwiseDistance(unittest.TestCase):

    def test_pairwise_distance(self):
        all_shape = [[5], [100, 100], [4, 5, 6, 7]]
        dtypes = ['float32', 'float64']
        p_list = [0, 1, 2, 'inf', '-inf']
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        keeps = [False, True]
        for place in places:
            for shape in all_shape:
                for dtype in dtypes:
                    for p in p_list:
                        for keepdim in keeps:
                            x_np = np.random.random(shape).astype(dtype)
                            y_np = np.random.random(shape).astype(dtype)

                            static_ret = test_static(place,
                                                     x_np,
                                                     y_np,
                                                     p,
                                                     keepdim=keepdim)
                            dygraph_ret = test_dygraph(place,
                                                       x_np,
                                                       y_np,
                                                       p,
                                                       keepdim=keepdim)
                            excepted_value = pairwise_distance(x_np,
                                                               y_np,
                                                               p,
                                                               keepdim=keepdim)

                            self.assertTrue(np.allclose(static_ret,
                                                        dygraph_ret))
                            self.assertTrue(
                                np.allclose(static_ret, excepted_value))
                            self.assertTrue(
                                np.allclose(dygraph_ret, excepted_value))

                            static_functional_ret = test_static(place,
                                                                x_np,
                                                                y_np,
                                                                p,
                                                                keepdim=keepdim)
                            dygraph_functional_ret = test_dygraph(
                                place, x_np, y_np, p, keepdim=keepdim)
                            self.assertTrue(
                                np.allclose(static_functional_ret,
                                            dygraph_functional_ret))
                            self.assertTrue(
                                np.allclose(static_functional_ret,
                                            excepted_value))
                            self.assertTrue(
                                np.allclose(dygraph_functional_ret,
                                            excepted_value))

    def test_pairwise_distance_broadcast1(self):
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

        static_founctional_ret = test_static(x_np,
                                             y_np,
                                             keepdim=keepdim,
                                             functional=True)
        dygraph_founctional_ret = test_dygraph(x_np,
                                               y_np,
                                               keepdim=keepdim,
                                               functional=True)
        self.assertTrue(
            np.allclose(static_founctional_ret, dygraph_founctional_ret))
        self.assertTrue(np.allclose(static_founctional_ret, excepted_value))
        self.assertTrue(np.allclose(dygraph_founctional_ret, excepted_value))

    def test_pairwise_distance_broadcast2(self):
        shape_x = [100, 100]
        shape_y = [100]
        keepdim = False
        x_np = np.random.random(shape_x).astype('float32')
        y_np = np.random.random(shape_y).astype('float32')
        static_ret = test_static(x_np, y_np, keepdim=keepdim)
        dygraph_ret = test_dygraph(x_np, y_np, keepdim=keepdim)
        excepted_value = pairwise_distance(x_np, y_np, keepdim=keepdim)
        self.assertTrue(np.allclose(static_ret, dygraph_ret))
        self.assertTrue(np.allclose(static_ret, excepted_value))
        self.assertTrue(np.allclose(dygraph_ret, excepted_value))

        static_founctional_ret = test_static(x_np,
                                             y_np,
                                             keepdim=keepdim,
                                             functional=True)
        dygraph_founctional_ret = test_dygraph(x_np,
                                               y_np,
                                               keepdim=keepdim,
                                               functional=True)
        self.assertTrue(
            np.allclose(static_founctional_ret, dygraph_founctional_ret))
        self.assertTrue(np.allclose(static_founctional_ret, excepted_value))
        self.assertTrue(np.allclose(dygraph_founctional_ret, excepted_value))


if __name__ == "__main__":
    unittest.main()
