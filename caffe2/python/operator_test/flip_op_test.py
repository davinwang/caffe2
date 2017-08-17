from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu
import numpy as np

import unittest


class TestFlip(hu.HypothesisTestCase):

    @given(H=st.sampled_from([19]),
           W=st.sampled_from([19]),
           engine=st.sampled_from([None, "CUDNN"]),
           **hu.gcs)
    def test_flip(self, H, W, engine, gc, dc):
        X = np.random.rand(H, W).astype(np.float32)
        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(1), engine=engine)
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=np.flip,
        )


if __name__ == "__main__":
    unittest.main()
