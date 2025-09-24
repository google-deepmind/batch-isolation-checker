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

"""Config class for ONNX model checker."""

from jaxonnxruntime.core import config_class


config = config_class.Config()


taint_dtype = config.define_enum_state(
    name='taint_dtype',
    enum_values=['uint64', 'uint32', 'uint16'],
    default='uint16',
    help='The dtype to use for taint propagation',
)
