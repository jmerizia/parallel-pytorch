"""
just a test for me to learn how 1f1b pipeline scheduler works
"""

from collections import deque
from fire import Fire

# some printing utils
def green(s):
    return '\033[0;32m' + s + '\033[0m'

def blue(s):
    return '\033[0;34m' + s + '\033[0m'

def run_schedule(
    use_1f1b=True,
    pipeline_stages=4,  # number of divisions of network vertically == numberof pipeline parallel dims
    micro_batches=4,
):
    micro_batches = list(range(micro_batches))
    QFs = [deque() for _ in range(pipeline_stages)]
    QBs = [deque() for _ in range(pipeline_stages)]
    lasts = ['backward' for _ in range(pipeline_stages)]
    # append all elements in the micro batch to the head of the pipeline
    for micro_batch in micro_batches:
        QFs[0].append(micro_batch)
    steps = 0
    while any(len(QF) > 0 for QF in QFs) or any(len(QB) > 0 for QB in QBs):
        processed = []
        to_push_later = []
        for rank in range(pipeline_stages):
            QF = QFs[rank]
            QB = QBs[rank]
            last = lasts[rank]
            todo = None

            if use_1f1b:
                # here we want to alternative (if possible)
                if last == 'backward':
                    if len(QF) > 0:
                        todo = 'forward'
                    elif len(QB) > 0:
                        todo = 'backward'
                if last == 'forward':
                    if len(QB) > 0:
                        todo = 'backward'
                    elif len(QF) > 0:
                        todo = 'forward'
            else:
                # here we want to greedily choose F if possible, otherwise B
                if len(QF) > 0:
                    todo = 'forward'
                elif len(QB) > 0:
                    todo = 'backward'

            if todo == 'forward':
                micro_batch = QF.popleft()
                processed.append((micro_batch, 'F'))
                if rank == pipeline_stages - 1:
                    # we are at the end, so push a backwards op
                    to_push_later.append(('backward', rank, micro_batch))
                else:
                    # we haven't reached the end, so push this micro batch on to the next pipeline stage
                    to_push_later.append(('forward', rank + 1, micro_batch))
                lasts[rank] = 'forward'

            elif todo == 'backward':
                micro_batch = QB.popleft()
                processed.append((micro_batch, 'B'))
                if rank == 0:
                    # we reached the end, so there's nothing left to do
                    pass
                else:
                    # we haven't reached the end, so push the next pipeline stage
                    to_push_later.append(('backward', rank - 1, micro_batch))
                lasts[rank] = 'backward'

            else:
                processed.append(None)

        # communicate the batches and the changes to the queues
        for t, rank, micro_batch in to_push_later:
            if t == 'forward':
                QFs[rank].append(micro_batch)
            else:
                QBs[rank].append(micro_batch)

        def fmt(e):
            if not e:
                return '   '
            elif e[1] == 'F':
                return green(f' {e[0]} ')
            else:
                return blue(f' {e[0]} ')

        print(' '.join(fmt(e) for e in processed))
        steps += 1

    print('Took', steps, 'steps')

if __name__ == '__main__':
    Fire(run_schedule)
