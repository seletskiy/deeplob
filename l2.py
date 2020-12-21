import sys
import datetime

step  = 0.25
depth = 10

def init():
    book = {
        'A': [(0, 0) for i in range(depth)],
        'B': [(0, 0) for i in range(depth)],
    }

    index = {
        'A': [0 for i in range(depth)],
        'B': [0 for i in range(depth)],
    }

    return book, index

book, index = init()

format = '%Y-%m-%d %H:%M:%S.%f'
tick = None

steps = 0

for line in open(sys.argv[1]):
    if line.startswith("E"):
        print(line, file=sys.stderr)
        continue

    if line.startswith("R"):
        book, index = init()
        continue

    try:
        [_, date, time, pos, op, side, price, size] = line.strip().split(" ")
    except:
        print('E malformed line: %s' % line.strip(), file=sys.stderr)
        continue

    if time < '14:35' or time > '20:55':
        tick = None
        continue

    if len(time) == len('00:00:00'):
        time += '.000000'

    pos = int(pos)
    price = float(price)
    size = int(size)

    if op == 'U' or op == 'I':
        book[side][pos] = (price, size)
        index[side][pos] = 1

    if op == 'D':
        book[side][pos] = (0, 0)

    # wait until each level will be changed at least once
    if sum(index['A'])+sum(index['B']) < depth * 2:
        continue

    # timestamp = datetime.datetime.strptime(
    #     date + ' ' + time, format
    # )

    # if tick is None:
    #     tick = datetime.datetime(
    #         timestamp.year,
    #         timestamp.month,
    #         timestamp.day,
    #         timestamp.hour,
    #         timestamp.minute,
    #         timestamp.second,
    #         timestamp.microsecond // 250000 * 250000
    #     )

    # timedelta = abs(timestamp - tick)

    steps += 1

    if steps % 10 == 0:
        steps = 0
        print('%s %s' % (date, time), end='')

        for i in range(depth):
            for j in ['A', 'B']:
                entry = book[j][i]
                if entry is not None:
                    price, size = entry
                    print(',%.2f,%d' % (price, size), end='')

        print()


    # steps = timedelta // datetime.timedelta(seconds=step)

    # if steps > 10:
    #     print(
    #         'E gap detected: %s -- %s' % (
    #             tick.strftime(format),
    #             timestamp.strftime(format)
    #         ),
    #         file=sys.stderr
    #     )

    #     tick = None

    #     continue

    # for i in range(steps):
    #     tick = tick + datetime.timedelta(seconds=step)

    #     # print('%s %s' % (date, time), end='')
    #     print(tick.strftime(format), end='')

    #     for i in range(depth):
    #         for j in ['A', 'B']:
    #             entry = book[j][i]
    #             if entry is not None:
    #                 price, size = entry
    #                 print(',%.2f,%d' % (price, size), end='')

    #     print()
