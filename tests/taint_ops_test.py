#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

"""Unit test configuration for taint propagation operators."""

from absl.testing import absltest

from batching_security_checker.core import config_class as taint_config
from tests import data_labeler
from tests import runner as label_runner


taint_config.config.update('taint_dtype', 'uint64')

runner = label_runner.LabelTestRunner(__name__)
expect_fail_patterns = []
exclude_patterns = []

runner.include('test_constant(_|$)(?!pad)', expected_n_tests=0)
runner.include(
    'test_constantofshape(_|$)',
    inputs_public=[0],
    inputs_unchanged=[0],
    expected_n_tests=0,
)


runner.include('test_abs_')
runner.include('test_add_')
runner.include('test_and_')
runner.include('test_acos_')
runner.include('test_acosh_')

runner.include('test_asin_')
runner.include('test_asinh_')
runner.include('test_atan_')
runner.include('test_atanh_')
runner.include('test_averagepool_')
runner.include('test_batchnormalization_')
# runner.include('test_bitshift_')
# runner.include('test_cast_')
# runner.include('test_castlike_')


runner.include('test_cast_')
expect_fail_patterns.extend([
    # not sure why these are expected to fail in jaxonnxruntime tests
    'test_cast_FLOAT_to_STRING',
    'test_cast_STRING_to_FLOAT',
    'test_cast_FLOAT_to_BFLOAT16_',
])
runner.include('test_ceil_')
runner.include('test_clip_')
runner.include('test_concat_')

runner.include('test_conv(_|$)', labeler=data_labeler.ConvLabeler())
runner.include('test_convinteger_(_|$)')

runner.include('test_cos_')
runner.include('test_cosh_')
runner.include('test_dequantizelinear_')
runner.include('test_dynamicquantizelinear_')
runner.include('test_div_')
runner.include('test_equal_')
runner.include('test_erf_')
runner.include('test_exp_')

runner.include('test_expand', inputs_public=[1], inputs_unchanged=[1])

runner.include('test_flatten_')

runner.include(
    'test_gather(_|$)(?!elements)'
)
runner.include(
    'test_gather_negative_indices', inputs_public=[1], inputs_unchanged=[1]
)

runner.include('test_gemm(_|$)', labeler=data_labeler.GemmLabeler())
runner.include('test_globalaveragepool', labeler=data_labeler.MaxPoolLabeler())

runner.include('test_greater_')
exclude_patterns.append('test_greater_*expanded')
runner.include('test_greaterorequal_')
exclude_patterns.append('test_greaterorequal_*expanded')
exclude_patterns.append('test_greater_equal_*expanded')
exclude_patterns.append('test_greater_equal_bcast_expanded_cpu')

runner.include('test_identity_')
exclude_patterns.append(
    'test_identity_sequence_'
)  # NOTE: not supported currently becuase supports list input
exclude_patterns.append('test_identity_opt_')

runner.include('test_leakyrelu_')
exclude_patterns.append('test_leakyrelu_*expanded')


runner.include('test_less_')
exclude_patterns.append('test_less_*expanded')
runner.include('test_lessorequal_')
exclude_patterns.append('test_less_equal_*expanded')
exclude_patterns.append('test_less_equal_bcast_expanded_cpu')


runner.include('test_log_')

runner.include('test_matmul(_|$)', labeler=data_labeler.MatMulLabeler())

runner.include('test_matmulinteger')

runner.include('test_maxpool', labeler=data_labeler.MaxPoolLabeler())
exclude_patterns.append(
    'test_maxpool_with_argmax_2d_'
)  # not implemented yet (requires support for indices as output)
exclude_patterns.append(
    'test_maxpool_2d_ceil_'
)  # not implemented yet (requires support for ceil)

runner.include('test_mul_')
runner.include('test_neg_')
runner.include('test_not_')
runner.include('test_or_')

runner.include('test_pow_')
runner.include('test_prelu_')

runner.include('test_range(_|$)(?!float|int32)', expected_n_tests=0)
runner.include('test_reciprocal_')


runner.include('test_reduce_mean(_|$)', inputs_public=[1], inputs_unchanged=[1])
# reduce to scalar
runner.include('test_reduce_mean_default_axes_keepdims', expected_n_tests=0)

runner.include('test_reduce_sum(_|$)', inputs_public=[1], inputs_unchanged=[1])
# reduce to scalar
runner.include('test_reduce_sum_default_axes_keepdims', expected_n_tests=0)
exclude_patterns.append('test_reduce_sum_square_')  # Op ReduceSumSquare

runner.include('test_relu_')

runner.include('test_reshape', inputs_public=[1], inputs_unchanged=[1])
runner.include('test_reshape_allowzero_reordered', expected_n_tests=0)  # empty

runner.include('test_scatternd(_|$)', inputs_public=[1], inputs_unchanged=[1])
runner.include('test_selu_')
runner.include('test_shape(_|$)', expected_n_tests=0)
runner.include('test_sigmoid_')
runner.include('test_sin(_|$)')
runner.include('test_sinh(_|$)')


runner.include(
    'test_slice', inputs_public=range(1, 5), inputs_unchanged=range(1, 5)
)
runner.include(
    'test_slice_start_out_of_bounds', expected_n_tests=0
)  # empty output

runner.include('test_softmax_')
runner.include(
    'test_softmax_example', expected_n_tests=0
)  # single dim -> everything has interference
exclude_patterns.append(
    'test_softmax_.*expanded'
)
runner.include('test_softplus_')
runner.include('test_squeeze_')
runner.include('test_sqrt_')
runner.include('test_sub_')
runner.include('test_tan_')
runner.include('test_tanh_')
runner.include('test_tile_')
runner.include('test_transpose_')

runner.include('test_tri[ul]', inputs_public=[1], inputs_unchanged=[1])
# expect some zero tests because outputs are unlabled (or empty)
runner.include('test_tri[ul]_zero', expected_n_tests=0)
runner.include('test_triu_out_pos', expected_n_tests=0)
runner.include('test_tril_out_neg', expected_n_tests=0)

runner.include('test_unsqueeze', inputs_public=[1], inputs_unchanged=[1])

runner.include('test_where_')

# runner = label_runner.LabelTestRunner(__name__)


# only run on cpu
exclude_patterns.append('_tpu')
exclude_patterns.append('_gpu')

for pattern in exclude_patterns:
  runner.exclude(pattern)

for pattern in expect_fail_patterns:
  runner.xfail(pattern)

globals().update(runner.test_cases)


if __name__ == '__main__':
  absltest.main()
