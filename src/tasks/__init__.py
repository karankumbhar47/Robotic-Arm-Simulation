# coding=utf-8
# Copyright 2023 The Ravens Authors.
#
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

"""Ravens tasks."""


from ravens.tasks.block_insertion import BlockInsertion
from ravens.tasks.block_insertion import BlockInsertionEasy
from ravens.tasks.block_insertion import BlockInsertionNoFixture
from ravens.tasks.block_insertion import BlockInsertionSixDof
from ravens.tasks.block_insertion import BlockInsertionTranslation

from ravens.tasks.task import Task


names = {

    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation
    
}
