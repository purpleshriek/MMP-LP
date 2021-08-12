import itertools

import common

from collections import Counter, defaultdict
from pprint import pprint

def process_params(supplied):
    defaults = {
        'divisor': 1,
        'list_seats': 7,
    }
    if supplied:
        defaults.update(supplied)

    defaults['divisor'] = float(defaults['divisor'])
    defaults['list_seats'] = int(defaults['list_seats'])

    return defaults


def winning_party(row):
    cons_keys = [name for name in row if name.endswith('Cons')]

    vote_counts = {
        key.rstrip('Cons').strip(): int(row[key])
        for key in cons_keys
        if row[key] != ''
    }
    if vote_counts:
        return max(vote_counts, key=vote_counts.get)
    else:
        return None


def cons_seats_by_region(data):
    cons_keys = [name for name in data[0] if name.endswith('List')]

    cons_seats = defaultdict(Counter)
    for row in data:
        party = winning_party(row)
        if party:
            cons_seats[row['Region']][party] += 1

    return {
        k: dict(v) for k, v in cons_seats.items()
    }




def total_list_votes_by_region(data):

    list_keys = [name for name in data[0] if name.endswith('List')]
    as_regions = [(row['Region'], {key: int(row[key] or 0) for key in list_keys})
                  for row in data]

    grouped = itertools.groupby(as_regions, lambda row: row[0])

    list_votes = {}
    for region, rows in grouped:
        region_votes = Counter()
        for row in rows:
            region_votes.update(row[1])
        list_votes[region] = {
            key.rstrip('List').strip(): votes
            for key, votes in region_votes.items()
        }

    return list_votes


def run_election(election, params, csv=False):
    data = election['data']

    cons_seats = cons_seats_by_region(data)

    region_seats = {}
    for region, list_votes in total_list_votes_by_region(data).items():
        region_total_seats = sum(cons_seats[region].values()) +\
            params['list_seats']

        region_seats[region] = common.apportion_highest_averages(
            list_votes, region_total_seats, params['divisor'],
            starting_seats=cons_seats[region])

    total_seats = Counter()
    for v in region_seats.values():
        total_seats.update(v)

    final_results = {
        region: {
            party:
                (cons_seats[region].get(party, 0),
                 seats - cons_seats[region].get(party, 0),
                 seats,)
            for party, seats in party_seats.items()
        }
        for region, party_seats in region_seats.items()
    }

    labels=['Region', 'Party', 'Cons Seats', 'List Seats', 'Total Seats']
    if csv:
        common.render_csv(final_results, labels)
    else:
        common.render_data(election['name'], final_results, labels)


