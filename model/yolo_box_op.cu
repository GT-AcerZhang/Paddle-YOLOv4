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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detection/yolo_box_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>


// input: 输入张量的指针，形状是[bz, 255, 13, 13]
// imgsize: img_size张量的指针，形状是[bz, 2]
// boxes:  输出张量boxes的指针
// scores: 输出张量scores的指针
// conf_thresh: 0.0
// anchors: 指针，指向[142, 110, 192, 243, 459, 401]
// n: 8  批大小
// h: 13 一列的格子数；格子行数
// w: 13 一行的格子数；格子列数
// an_num: 3 每个格子有几个预测框
// class_num: 80
// box_num: 3*13*13
// input_size: 32*13=416
// clip_bbox: True
// scale:
// bias: -0.5 * (scale - 1.)
__global__ void KeYoloBoxFw(const T* input, const int* imgsize, T* boxes,
                            T* scores, const float conf_thresh,
                            const int* anchors, const int n, const int h,
                            const int w, const int an_num, const int class_num,
                            const int box_num, int input_size, bool clip_bbox,
                            const float scale, const float bias) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T box[4];

  // 遍历所有预测框，共bz*3*13*13个
  //           8 * 3*13*13
  for (; tid < n * box_num; tid += stride) {
    int grid_num = h * w;    // 这张特征图格子总数，比如13x13=169

    // 获得4个坐标[i, j, ?, k, l]。输出张量的形状是[bz, 3, 85, 13, 13]，用来取预测框。
    int i = tid / box_num;   // batch_size维的下标，第几张图片。box_num=3*13*13
    int j = (tid % box_num) / grid_num;  // 预测框的下标
    int k = (tid % grid_num) / w;        // h的坐标，网格内的y坐标
    int l = tid % w;                     // w的坐标，网格内的x坐标

    // 输出张量的形状是[bz, 3, 85, 13, 13]，那么同一个格子相邻两个预测框的步长 an_stride 就是 85*13*13
    int an_stride = (5 + class_num) * grid_num;
    int img_height = imgsize[2 * i];     // 原图的高
    int img_width = imgsize[2 * i + 1];  // 原图的宽

    // 输出张量的形状是[bz, 3, 85, 13, 13]
    // an_num 就是 3
    // an_stride 就是 85*13*13
    // grid_num 就是 13*13
    // 实际上输出张量input是一个一维数组（指针），所以将坐标[i, j, 4, k, l]转换成真实位置obj_idx
    int obj_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 4);
    T conf = sigmoid<T>(input[obj_idx]);   // 置信位经过sigmoid()激活
    if (conf < conf_thresh) {    // 置信位数值低于conf_thresh，就过滤这个预测框
      continue;
    }

    // 实际上输出张量input是一个一维数组（指针），所以将坐标[i, j, 0, k, l]转换成真实位置box_idx
    int box_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 0);
    // 将坐标为box_idx处的box解码，用那个公式。box作为返回值。
    GetYoloBox<T>(box, input, anchors, l, k, j, h, input_size, box_idx,
                  grid_num, img_height, img_width, scale, bias);
    box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
    CalcDetectionBox<T>(boxes, box, box_idx, img_height, img_width, clip_bbox);

    // 实际上输出张量input是一个一维数组（指针），所以将坐标[i, j, 5, k, l]转换成真实位置label_idx
    int label_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 5);
    int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
    CalcLabelScore<T>(scores, input, label_idx, score_idx, class_num, conf,
                      grid_num);
  }
}

// x = fluid.layers.data(name='x', shape=[255, 13, 13], dtype='float32')
// img_size = fluid.layers.data(name='img_size',shape=[2],dtype='int64')
// anchors = [142, 110, 192, 243, 459, 401]
// boxes, scores = fluid.layers.yolo_box(x=x, img_size=img_size, class_num=80, anchors=anchors,
//                                 conf_thresh=0.0, downsample_ratio=32)

template <typename T>
class YoloBoxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");    // [bz, 255, 13, 13]
    auto* img_size = ctx.Input<Tensor>("ImgSize");   // [bz, 2]
    auto* boxes = ctx.Output<Tensor>("Boxes");     // [bz, -1, 4]
    auto* scores = ctx.Output<Tensor>("Scores");   // [bz, -1, 80]

    auto anchors = ctx.Attr<std::vector<int>>("anchors");   //  [142, 110, 192, 243, 459, 401]
    int class_num = ctx.Attr<int>("class_num");   // 80
    float conf_thresh = ctx.Attr<float>("conf_thresh");   // 0.0
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");  // 32
    bool clip_bbox = ctx.Attr<bool>("clip_bbox");   // True
    float scale = ctx.Attr<float>("scale_x_y");
    float bias = -0.5 * (scale - 1.);

    const int n = input->dims()[0];   // bz，批大小
    const int h = input->dims()[2];   // 13，格子行数
    const int w = input->dims()[3];   // 13，格子列数
    const int box_num = boxes->dims()[1];   // 3*13*13
    const int an_num = anchors.size() / 2;  // 3
    int input_size = downsample_ratio * h;  // 32*13=416

    auto& dev_ctx = ctx.cuda_device_context();
    int bytes = sizeof(int) * anchors.size();   // ?*6
    auto anchors_ptr = memory::Alloc(dev_ctx, sizeof(int) * anchors.size());
    int* anchors_data = reinterpret_cast<int*>(anchors_ptr->ptr());
    const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    const auto cplace = platform::CPUPlace();
    memory::Copy(gplace, anchors_data, cplace, anchors.data(), bytes,
                 dev_ctx.stream());

    const T* input_data = input->data<T>();   // 获得输入张量的指针，形状是[bz, 255, 13, 13]
    const int* imgsize_data = img_size->data<int>();   // 获得img_size张量的指针，形状是[bz, 2]
    T* boxes_data = boxes->mutable_data<T>({n, box_num, 4}, ctx.GetPlace());   // 获得输出张量boxes的指针
    T* scores_data = scores->mutable_data<T>({n, box_num, class_num}, ctx.GetPlace());   // 获得输出张量scores的指针
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    set_zero(dev_ctx, boxes, static_cast<T>(0));   // 将boxes初始化为0
    set_zero(dev_ctx, scores, static_cast<T>(0));  // 将scores初始化为0

    int grid_dim = (n * box_num + 512 - 1) / 512;   // 暂时不明白什么意思
    grid_dim = grid_dim > 8 ? 8 : grid_dim;   // 三个输出层，都是得8

    KeYoloBoxFw<T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        input_data, imgsize_data, boxes_data, scores_data, conf_thresh,
        anchors_data, n, h, w, an_num, class_num, box_num, input_size,
        clip_bbox, scale, bias);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(yolo_box, ops::YoloBoxOpCUDAKernel<float>,
                        ops::YoloBoxOpCUDAKernel<double>);



