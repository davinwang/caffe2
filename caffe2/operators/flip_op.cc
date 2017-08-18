#include "caffe2/operators/flip_op.h"

namespace caffe2 {

  template <>
  template <typename T>
  bool FlipOp<CPUContext>::DoRunWithType() {
    const auto& input = Input(0);
    auto* output = Output(0);
    size_t count = input.size();
    int num_axes = axes_.size();
    CAFFE_ENFORCE(OperatorBase::HasArgument("axes"), "argument axes is missing");
    const T* from_data = input.template data<T>();
    T* to_data = output->template mutable_data<T>();
    auto in_dims = input.dims();

    // Measure amount of contiguous data we can copy at once
    // Suppose input.dims()=(N,C,H,W),
    //   if axes=(1,) or (0,1) then blocksize = H * W
    //   if axes=(2,) or (1,2) or (0,1,2) then blocksize = W
    //   if axes=(3,) or (2,3) or (1,2,3) or (0,1,2,3) then blocksize = 1
    // Calculate stride
    //   if axes=(1,) or (1,2) or (1,2,3) then stride = C * H * W
    //   if axes=(2,) or (2,3) then stride = H * W
    //   if axes=(3,) then stride = W
    TIndex blocksize = 1;
    TIndex stride = 1;
    for (int i = input.ndim() - 1; i >= 0; --i) {
      if (axes_[num_axes - 1] < i) {
        blocksize *= in_dims[i];
      }
      if (axes_[0] <= i) {
        stride *= in_dims[i];
      }
      else {
        break;
      }
    }

    // Now, for every stride, reverse data in blocksize
    // Branch here to avoid branching within the loop
    if (blocksize > 1) {
      for (size_t index = 0; index < count; index += stride) {
        for (size_t i = 0; i < stride; i += blocksize) {
          memcpy(
            to_data + blocksize * (index + i),
            from_data + blocksize * (index + stride - 1 - i),
            blocksize * sizeof(T));
        }
      }
    }
    else {
      for (size_t index = 0; index < count; index += stride) {
        for (size_t i = 0; i < stride; i++) {
          *(to_data + index + i) = *(from_data + index + stride - 1 - i);
        }
      }
    }

    return true;
  }

  namespace {
    REGISTER_CPU_OPERATOR(Flip, FlipOp<CPUContext>);

    OPERATOR_SCHEMA(Flip)
      .NumInputs(1)
      .NumOutputs(1)
      .TensorInferenceFunction([](
        const OperatorDef& def,
        const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        auto axis = in[0].dims().rend() - 1;
        out[0].add_dims(*axis);
      }
      else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
          std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
          return axis >= 0 && axis < tensor_size;
        });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
          axes.size() <= tensor_size,
          "Axes argument passed in had the incorrect size");

        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    })
      .SetDoc(R"DOC(
Flip the input tensor similar to numpy.flip. For example, when axes=(3,) or 
None, given an input tensor M of shape (N, C, H, W), the output will be 
similar as numpy.flip(M, 3) or numpy.fliplr(M).
)DOC")
      .Arg(
        "axes",
        "A list of integers. By default, flip the last dimension, "
        "otherwise flip the axes according to the values given.")
      .Input(0, "data", "An input tensor.")
      .Output(0, "flipped", "Flipped output.");

    class GetFlipGradient : public GradientMakerBase {
      using GradientMakerBase::GradientMakerBase;
      // We will create our own arguments.
      bool CopyArguments() const override {
        return false;
      }
      vector<OperatorDef> GetGradientDefs() override {
        auto ops = SingleGradientDef(
          "Flip", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        ops[0].mutable_arg()->CopyFrom(Def().arg());
        if (ArgumentHelper::HasArgument(Def(), "axes")) {
          // If axes is specified, we will need to figure out the inverse index.
          const Argument& old_axes = GetArgument(Def(), "axes");
          const int axes_size = old_axes.ints_size();
          Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
          for (int i = 0; i < axes_size; ++i) {
            new_arg->set_ints(old_axes.ints(i), i);
          }
        }
        return ops;
      }
    };
    REGISTER_GRADIENT(Flip, GetFlipGradient);
  } // namespace
} // namespace caffe2
