from tests import (
    test_checkpoint,
    test_merge,
    test_split,
    test_parallel_merge,
    test_parallel_split,
)


if __name__ == '__main__':
    test_checkpoint.run_all()
    test_merge.run_all()
    test_split.run_all()
    test_parallel_merge.run_all()
    test_parallel_split.run_all()
