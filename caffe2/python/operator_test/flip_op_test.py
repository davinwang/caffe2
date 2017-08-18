from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestFlip(hu.HypothesisTestCase):

    @given(H=st.sampled_from([1,3,8]),
           W=st.sampled_from([2,5,11]),
           engine=st.sampled_from([None]),
           **hu.gcs)
    def test_flip(self, H, W, engine, gc, dc):
        X = np.random.rand(H, W).astype(np.float32)
        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(1,), engine=engine)
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=np.fliplr,
        )


if __name__ == "__main__":
    unittest.main()
