#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-28 12:03:15
#   Description : test_where
#
# ================================================================
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr


a0 = np.zeros((2, 2), 'float32')
a0[0][0] = 3.0
a0[0][1] = 4.0
a0[1][0] = -1.0
a0[1][1] = 4.0


cond = P.data(name='xxx', shape=[-1, -1], append_batch_size=False, dtype='float32')
keep = P.where(cond > 0.5)


# Create an executor using CPU as an example
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

aaa00 = exe.run(fluid.default_main_program(),
                  feed={'xxx': a0, },
               fetch_list=keep)
print(aaa00)
print()




