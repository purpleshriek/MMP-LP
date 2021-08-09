import csv
import json

from pathlib import Path
from collections import defaultdict


def load_election(data_path):
    filepath = Path(data_path)
    basedir = filepath.parents[0]

    with open(filepath) as f:
        election = json.loads(f.read())

    data_filename = basedir / election['data']
    with open(data_filename) as f:
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

