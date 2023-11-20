"""tasks names"""

from src.tasks.block_insertion import BlockInsertion
from src.tasks.block_insertion import BlockInsertionEasy
from src.tasks.block_insertion import BlockInsertionNoFixture
from src.tasks.block_insertion import BlockInsertionSixDof
from src.tasks.block_insertion import BlockInsertionTranslation
from src.tasks.task import Task

names = {
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation
}