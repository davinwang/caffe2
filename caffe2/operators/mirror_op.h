#ifndef CAFFE2_OPERATORS_MIRROR_OP_H_
#define CAFFE2_OPERATORS_MIRROR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

	template <class Context>
	class MirrorOp final : public Operator<Context> {
	public:
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		USE_DISPATCH_HELPER;
		MirrorOp(const OperatorDef& operator_def, Workspace* ws)
			: Operator<Context>(operator_def, ws),
			axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
			// We will check the legality of axes_: it should be monotone increasing tuple between 0 and X.ndim().
			for (int i = 1; i < axes_.size(); ++i) {
				if (axes_[i] != axes_[0] + i) {
					CAFFE_THROW("Axes should be a monotone increasing tuple between 0 and ndim.");
				}
			}
		}
		~MirrorOp() {}

		bool RunOnDevice() override {
			const auto& X = Input(0);
			auto* Y = Output(0);
			Y->ResizeLike(X);
			// Do the actual mirror, which is implemented in DoRunWithType().
			return DispatchHelper<TensorTypes<float, double, int, long>>::call(
				this, Input(0));
		}

	protected:
		template <typename T>
		bool DoRunWithType();

		std::vector<int> axes_;
		std::vector<TIndex> new_dims_;
		// buffer_ is used in MirrorOp<CUDAContext> so we can obtain a consistent
		// buffer on the GPU. It is not used in the CPUContext implementation.
		Tensor<Context> buffer_;
		TensorCPU buffer_cpu_;
	};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MIRROR_OP_H_
