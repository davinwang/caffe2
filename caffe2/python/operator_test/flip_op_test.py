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

        # test1: compare with numpy.fliplr
        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(1,), engine=engine)
        
        def ref_fliplr(X):
            return [np.fliplr(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref_fliplr,
        )

    @given(H=st.sampled_from([1,3,8]),
           W=st.sampled_from([2,5,11]),
           engine=st.sampled_from([None]),
           **hu.gcs)
    def test_flip(self, H, W, engine, gc, dc):
        X = np.random.rand(H, W).astype(np.float32)

        # test2: compare with numpy.flipud
        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(0,), engine=engine)


        def ref_flipud(X):
            return [np.flipud(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref_flipud,
        )

if __name__ == "__main__":
    unittest.main()
