#ifndef CAFFE2_OPERATORS_MIRROR_OP_H_
#define CAFFE2_OPERATORS_MIRROR_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class MirrorOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MirrorOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
};

template <typename T, class Context>
class MirrorGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MirrorGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // Input: Y, dY; Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MIRROR_OP_H_
