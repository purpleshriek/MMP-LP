import itertools

from collections import defaultdict, Counter

import common


def process_votes_german(row):
    list_keys = [name for name in row if name.endswith('List')]
    cons_keys = [name for name in row if name.endswith('Cons')]

    list_votes = {key.rstrip('List').strip(): int(row[key])
                  for key in list_keys}
    cons_votes = {key.rstrip('Cons').strip(): int(row[key])
                  for key in cons_keys}

    list_keys = set(list_keys)
    cons_keys = set(cons_keys)

    processed_row = {k: v for k, v in row.items()
                     if k not in list_keys and k not in cons_keys}

    processed_row['List Votes'] = list_votes
    processed_row['Cons Votes'] = cons_votes

    return processed_row


def apportion_national_seats(weights, minimum_total, minimum_party, divisor=0.5):
    reps = {party: 0 for party in weights}

    def score(party):
        return weights[party] * (1 / (1 + (reps[party] / divisor)))

    while sum(reps.values()) < minimum_total or \
            any(reps[party] < minimum_party[party] for party in
                weights):
        next_party = max(reps, key=score)
        reps[next_party] += 1

    return reps


def group_party_list_votes_by_state(rows):
    state_key = lambda row: row['State']
    rows = sorted(rows, key=state_key)
    state_groups = itertools.groupby(rows, key=state_key)

    state_votes = {}

    for state, rows in state_groups:
        party_votes = defaultdict(lambda: 0)
        for row in rows:
            for party in row['List Votes']:
                party_votes[party] += row['List Votes'][party]
        state_votes[state] = dict(party_votes)

    return state_votes


def apportion_state_seats_by_party(reps, rows, **kwargs):
    state_seats = {}
    state_party_votes = group_party_list_votes_by_state(rows)

    for state, party_votes in state_party_votes.items():
        state_seats[state] = \
            common.apportion_highest_averages(party_votes, reps[state],
                                              **kwargs)

    return state_seats


def determine_overhang(constituency, proportional):
    overhang_votes = {}

    for state, state_votes in proportional.items():
        counts = dict(state_votes)
        for party, cons_seats in constituency[state].items():
            counts[party] = max(counts[party], cons_seats)
        overhang_votes[state] = counts

    return overhang_votes


def run_german_election(election):
    election['data'] = [process_votes_german(row) for row in election['data']]

    reps = common.apportion_highest_averages(election['census'],
                                             election['seats'], divisor=0.5)

    for row in election['data']:
        row['Cons Party'] = max(row['Cons Votes'], key=row['Cons Votes'].get)

    constituency_seats = {
        state: dict(Counter(row['Cons Party'] for row in rows))
        for state, rows in itertools.groupby(election['data'], lambda row: row['State'])
    }

    proportion_seats = apportion_state_seats_by_party(reps, election['data'],
                                                      divisor=0.5)

    overhang_seats = determine_overhang(constituency_seats, proportion_seats)

    total_overhang = sum(sum(state_votes.values())
                              for state_votes in overhang_seats.values())

    national_overhang = Counter()
    for state_seats in overhang_seats.values():
        national_overhang.update(state_seats)
    national_overhang = dict(national_overhang)

    national_party_votes = Counter()
    for row in election['data']:
        national_party_votes.update(row['List Votes'])
    national_party_votes = dict(national_party_votes)

    national_seats = apportion_national_seats(
        national_party_votes,
        total_overhang,
        national_overhang,
    )

    total_national = sum(national_seats.values())

    state_votes = group_party_list_votes_by_state(election['data'])

    state_votes_by_party = common.flip_hierarchy(state_votes)

    constituency_party = common.flip_hierarchy(constituency_seats)

    final_vote = {}

    for party, state_votes in state_votes_by_party.items():
        starting_seats = constituency_party.get(party, {})
        total_party = national_seats.get(party, 0)

        alloc = common.apportion_highest_averages(state_votes, total_party,
                                                  starting_seats=starting_seats,
                                                  divisor=0.5)
        final_vote[party] = alloc

    final_vote = common.flip_hierarchy(final_vote)
    return final_vote


