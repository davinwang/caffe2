#include "caffe2/operators/mirror_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool MirrorOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

#ifdef CAFFE2_USE_ACCELERATE
  const float zero = 0.0f;
  vDSP_vthres(X.data<float>(), 1, &zero, Y->mutable_data<float>(), 1, X.size());
#else
  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size()).cwiseMax(0.f);
#endif
  /* Naive implementation
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
  }
  */
  return true;
}

template <>
bool MirrorGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // TODO: proper vectorization with Eigen
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * Yvec.cwiseSign();
  /* Previous implementation
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
  }
  */
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Mirror, MirrorOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MirrorGradient, MirrorGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Mirror)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Mirror takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
  .Input(0, "X", "1D input tensor")
  .Output(0, "Y", "1D input tensor");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(MirrorGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
MirrorGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetMirrorGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Mirror, GetMirrorGradient);
REGISTER_GRADIENT(MirrorFp16, GetMirrorGradient);

}  // namespace
}  // namespace caffe2
