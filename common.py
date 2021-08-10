import csv
import gzip
import json

from pathlib import Path
from collections import defaultdict


def open_maybe_compressed(path, mode='rt'):
    if path.suffix == '.gz':
        return gzip.open(path, mode)
    else:
        return open(path, mode)


def load_election(data_path):
    filepath = Path(data_path)
    basedir = filepath.parents[0]

    with open_maybe_compressed(filepath) as f:
        election = json.loads(f.read())

    data_filename = basedir / election['data']
    with open_maybe_compressed(data_filename) as f:
        reader = csv.DictReader(f, delimiter=',')
        election['data'] = list(reader)

    return election


def apportion_highest_averages(weights, seats, divisor, min_seats=0,
                               starting_seats=None):
    reps = {district: min_seats for district in weights}
    if starting_seats:
        reps.update(starting_seats)

    def score(district):
        return weights[district] * (1 / (1 + (reps[district] / divisor)))

    allocated = sum(reps.values())

    for i in range(allocated, seats):
        next_district = max(reps, key=score)
        reps[next_district] += 1

    return reps

def flip_hierarchy(hierarchy_votes):
    remapped = defaultdict(dict)
    for a, a_votes in hierarchy_votes.items():
        for b, votes in a_votes.items():
            remapped[b][a] = votes

    return dict(remapped)


def flatten_data(data):
    for k, v in data.items():
        try:
            first = True
            depth = 0
            for t in flatten_data(v):
                if first:
                    yield (k, *t)
                    depth = len(t)
                else:
                    yield ('', *t)
                first = False
        except AttributeError:
            yield (k, v)


def render_data(name, data, fields):
    flat = list(flatten_data(data))
    data_widths = [max(len(str(row[col])) for row in flat)
                   for col in range(len(flat[0]))]
    col_widths = [max(len(field), data_len)
                  for field, data_len in zip(fields, data_widths)]

    header_line = ' | '.join(f'{f:^{w}}' for f, w in zip(fields, col_widths))

    print(f'{name:^{len(header_line)}}')
    print()

    print(header_line)
    print('=' * len(header_line))

    for row in flat:
        print(' | '.join(f'{item:>{w}}' for item, w in zip(row, col_widths)))
