from collections import Counter

import common


def process_params(supplied):
    defaults = {
        'divisor': 0.5
    }

    if supplied:
        defaults.update(supplied)

    defaults['divisor'] = float(defaults['divisor'])
    return defaults


def run_election(election, params, csv=False):
    list_fields = [f for f in election['data'][0].keys() if f.endswith('List')]

    party_votes = {}
    for field in list_fields:
        party = field.strip().rstrip('List').rstrip()
        party_votes[party] = sum(int(entry[field]) for entry in election['data']
                                 if entry[field])

    cons_seats = Counter(row['Cons Seat Party'] for row in election['data']
                         if row['Cons Seat Party'])
    cons_seats = dict(cons_seats)

    combined_seats = common.indefinite_highest_averages(party_votes,
                                                        election['seats'],
                                                        cons_seats,
                                                        divisor=params['divisor'])

    total_seats = sum(combined_seats.values())
    if total_seats != election['seats']:
        print(f'Warning: Expected {election["seats"]} seats, got {total_seats}')
        print()

    labels = ['Party', 'Seats']
    if csv:
        common.render_csv(combined_seats, labels)
    else:
        common.render_data(election['name'], combined_seats, labels)
