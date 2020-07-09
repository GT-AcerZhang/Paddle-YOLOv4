/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <functional>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct WhereIndexFunctor {
    // true_index，[M, ]
    // true_num，M
    // stride，[N, ]，表示的是进制
    // rank，N
    // out_ptr，[M, N]
  WhereIndexFunctor(const T* true_index, int true_num, const T* stride,
                    int rank, T* out)
      : true_index_(true_index),
        true_num_(true_num),
        stride_(stride),
        rank_(rank),
        out_ptr_(out) {}

  HOSTDEVICE void operator()(size_t idx) const {
    // 将true_index里的1维坐标转换成4维坐标，写入到out_ptr_里。
    T index = true_index_[idx];
    for (int j = 0; j < rank_; j++) {
      out_ptr_[idx * rank_ + j] = index / stride_[j];   // out_ptr_依然是一个一维数组
      index -= out_ptr_[idx * rank_ + j] * stride_[j];
    }
  }

  const T* true_index_;
  int true_num_;
  const T* stride_;
  int rank_;
  T* out_ptr_;
};

using CPUDeviceContext = paddle::platform::CPUDeviceContext;



// where_index op是没有反向传播的。这很好理解，因为op返回的是坐标，坐标没有参与到损失的计算，而是通过这些坐标gather/gather_nd得到的元素参与了损失的计算。
// 所以where_index op没有反向传播。联想一下，最大池化也是保存了最大值的坐标，坐标没有直接参与损失的计算。


// CPU、CUDA共享Kernel实现在.h 文件中，否则，CPU 实现在.cc 文件中，CUDA 实现在.cu 文件中。

// 前向。C++的前向
// where_index op会返回一个形状为(M, N)的张量，其中M是符合条件的元素的个数，N是坐标的维数。
template <typename T>
class CPUWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition"); // 实际上输入张量condition是一个一维数组（指针）
    auto* out = context.Output<framework::Tensor>("Out");

    const T* cond_data = condition->data<T>();
    auto numel = condition->numel();   // numel()函数返回condition的元素个数

    // dims()函数返回张量的形状。dims的类型是std::vector<int64_t>。
    // 比如若condition在python端是一个形状为[8, 3, 13, 13]的张量时，那么这里的dims是一个数组vector，值为[8, 3, 13, 13]
    auto dims = condition->dims();
    const int rank = dims.size();    // 坐标的维数，即N。vector对象的函数，返回数组大小。这里是4。

    std::vector<int64_t> true_index;
    for (auto i = 0; i < numel; i++) {
      if (static_cast<bool>(cond_data[i])) {
        true_index.push_back(i);   // 加入符合条件的下标
      }
    }
    auto true_num = true_index.size();   // true_index的大小。即符合条件的元素个数，即M

    // 将out reshape成(M, N)
    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_ptr = out->mutable_data<int64_t>(context.GetPlace());   // 调用mutable_data()函数获取指针，可以指定形状

    if (true_num == 0) {   // 如果M==0，什么都不返回
      return;
    }

    // 创建一个std::vector<int64_t>对象stride，里面有rank（即N）个int64_t。
    // 相当于
    // stride = np.zeros((rank, ), 'int64')
    // stride[rank - 1] = 1
    // for i in range(rank-2, -1, -1):
    //     stride[i] = stride[i + 1] * dims[i + 1]
    // stride表示的是进制，
    // 若dims == [8, 3, 13, 13]，则stride == [3*13*13, 13*13, 13, 1]
    // 第0维的单位1表示的是其实是3*13*13，第1维的单位1表示的是其实是13*13，第2维的单位1表示的是其实是13，第3维的单位1表示的是其实是1。
    // stride用于将true_index里的1维坐标转换成4维坐标。
    std::vector<int64_t> stride(rank);
    stride[rank - 1] = 1;   // 数组stride最后一个元素设为1
    for (int i = rank - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * dims[i + 1];   // stride == [3*13*13, 13*13, 13, 1]，表示的是进制
    }

    auto& dev_ctx = context.template device_context<CPUDeviceContext>();
    // true_index，[M, ]
    // true_num，M
    // stride，[N, ]，表示的是进制
    // rank，N
    // out_ptr，[M, N]
    // 创建WhereIndexFunctor<int64_t>对象functor
    WhereIndexFunctor<int64_t> functor(true_index.data(), true_num,
                                       stride.data(), rank, out_ptr);
    // 创建ForRange<CPUDeviceContext>对象for_range
    platform::ForRange<CPUDeviceContext> for_range(dev_ctx, true_num);
    // 会调用WhereIndexFunctor里的operator()
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle