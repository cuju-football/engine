# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the cuju Authors in 2026, in the import lines that ci/port_gfootball.py lists.
# SPDX-License-Identifier: Apache-2.0

"""Google Research Football."""

from cuju.gfootball.env import scenario_builder

from cuju.gfootball import _gym as gym
from cuju.gfootball._gym import register


for env_name in scenario_builder.all_scenarios():
  register(
      id='GFootball-{env_name}-SMM-v0'.format(env_name=env_name),
      entry_point='cuju.gfootball.env:create_environment',
      kwargs={
          'env_name': env_name,
          'representation': 'extracted'
      },
  )

  register(
      id='GFootball-{env_name}-Pixels-v0'.format(env_name=env_name),
      entry_point='cuju.gfootball.env:create_environment',
      kwargs={
          'env_name': env_name,
          'representation': 'pixels'
      },
  )

  register(
      id='GFootball-{env_name}-simple115-v0'.format(env_name=env_name),
      entry_point='cuju.gfootball.env:create_environment',
      kwargs={
          'env_name': env_name,
          'representation': 'simple115'
      },
  )

  register(
      id='GFootball-{env_name}-simple115v2-v0'.format(env_name=env_name),
      entry_point='cuju.gfootball.env:create_environment',
      kwargs={
          'env_name': env_name,
          'representation': 'simple115v2'
      },
  )
