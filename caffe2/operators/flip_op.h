#ifndef CAFFE2_OPERATORS_FLIP_OP_H_
#define CAFFE2_OPERATORS_FLIP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

  template <class Context>
  class FlipOp final : public Operator<Context> {
  public:
    USE_OPERATOR_CONTEXT_FUNCTIONS;
    USE_DISPATCH_HELPER;
    FlipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
      // 
      CAFFE_ENFORCE(OperatorBase::HasArgument("axes"), "Argument `axes` is missing");
      // We will check the legality of axes_: it should be monotone increasing tuple between 0 and X.ndim().
      for (int i = 1; i < axes_.size(); ++i) {
        CAFFE_ENFORCE(axes_[i] == axes_[0] + i, "Argument `axes` has invalid dimension:", axes_[i]);
      }
    }
    ~FlipOp() {}

    bool RunOnDevice() override {
      const auto& X = Input(0);
      auto* Y = Output(0);
      Y->ResizeLike(X);
      // Do the actual flip, which is implemented in DoRunWithType().
      return DispatchHelper<TensorTypes<float, double, int, long>>::call(
        this, Input(0));
    }

  protected:
    template <typename T>
    bool DoRunWithType();

    std::vector<int> axes_;
    // buffer_ is used in FlipOp<CUDAContext> so we can obtain a consistent
    // buffer on the GPU. It is not used in the CPUContext implementation.
    Tensor<Context> buffer_;
    TensorCPU buffer_cpu_;
  };

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLIP_OP_H_
