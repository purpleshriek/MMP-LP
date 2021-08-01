# MMP-LP

from pulp import *
from pyomo.environ import *
import os, sys, time

print 'Start at', time.asctime()

# file
args = sys.argv[1:]
if (args) :
    fn = args[0]
else :
    fn = './ballots/2019.pr'
os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/bin'
f = {x.splitlines()[0][2:]: '\n'.join(x.splitlines()[1:]) for x in open(fn, 'r').read().replace(',', '').split('\n\n')}

# apportionment methods
if 'apportionment method' not in f :
    f['apportionment method'] = 'Germany'
for m in ['initial census apportionment method',
          'state party apportionment method',
          'national party apportionment method',
          'leveling seat apportionment method',
          'revised census apportionment method',
          'final party apportionment method'] :
    if m not in f :
        f[m] = f['apportionment method']

# census
if 'census' in f :
    C = {line.split('\t')[0].lstrip().rstrip(): int(line.split('\t')[1]) for line in f['census'].splitlines()}
else :
    C = {f['votes'].splitlines()[-1].split('\t')[0]: 0}
L = C.keys()

# ballots
if f['ballot type'] == 'Approval' :
    f['votes'] = f['votes'].splitlines()[1:]
    P = list(set('\t'.join(['\t'.join([t for t in line.rstrip().split('\t')[5:]]) for line in f['votes']]).split('\t')))
    B = [[0, '', []] for b in range(len(list(set(line.split('\t')[0] for line in f['votes']))))]
    for line in f['votes'] :
        if line.split('\t')[4] == 'List' :
            B[int(line.split('\t')[0])][0] += int(line.split('\t')[1])
            B[int(line.split('\t')[0])][1] = line.split('\t')[2]
            for t in line.split('\t')[5:] :
                if t not in B[int(line.split('\t')[0])][2] and t not in ['', '\t'] :
                    B[int(line.split('\t')[0])][2].append(t)
    D = {s: {c: {p: sum([B[b][0] for b in range(len(B)) if B[b][1] == s and p in B[b][2]]) for p in P} for c in list(set(line.split('\t')[3] for line in f['votes'] if line.split('\t')[2] == s))} for s in list(set(line.split('\t')[2] for line in f['votes']))}
    D = {s: {p: sum([1 for c in D[s] if D[s][c][p] == max([D[s][c][q] for q in D[s][c]])]) for p in P} for s in L}
if f['ballot type'] == 'FPTP' :
    P = f['votes'].splitlines()[0].split('\t')[3::3]
    n = 1
    f['votes'] = f['votes'].splitlines()
    while len(f['votes'][n].split('\t')) != (len(P) + 1) * 3 :
        n += 1
    f['votes'] = '\n'.join(f['votes'])
    B = '\n'.join([line.lstrip().rstrip() for line in f['votes'].splitlines()[n:]]).split('\n\n')[:-1]
    D = {B[b].splitlines()[-1].split('\t')[0]: {B[b].splitlines()[n].split('\t')[0]: B[b].splitlines()[n].split('\t')[1] for n in range(len(B[b].splitlines()))[:-1]} for b in range(len(B))}
    D = {s: {p: sum([1 for c in D[s] if D[s][c] == p]) for p in P} for s in L}
    LV = {B[b].splitlines()[-1].split('\t')[0]: {P[p]: int(B[b].splitlines()[-1].split('\t')[(p * 3) + 4]) for p in range(len(P))} for b in range(len(B))}
    B = {x.splitlines()[-1].split('\t')[0]: {c.split('\t')[0]: {P[p]: [y for y in c.split('\t')[3:][(p * 3):][:2]] for p in range(len(P))} for c in x.splitlines()[:-1]} for x in B}
    B = [[LV[s][p], s, [p]] for s in L for p in P]
    B = [b for b in B if b[0] > 0]

# divisor (0.5 = Sainte-Lague; 1.0 = D'Hondt)
if 'divisor' in f :
    f['divisor'] = float(f['divisor'])
else :
    f['divisor'] = 0.5
for m in ['initial census apportionment divisor',
          'state party apportionment divisor',
          'national party apportionment divisor',
          'leveling seat apportionment divisor',
          'revised census apportionment divisor',
          'final party apportionment divisor'] :
    if m not in f :
        f[m] = f['divisor']
    else :
        f[m] = float(f[m])

# seats
if 'seats' in f :
    if f['seats'] == 'Constituencies' :
        N = sum([sum([D[s][p] for p in P]) for s in L])
    else :
        N = int(f['seats'])
else :
    N = (50 * (int((sum([C[s] for s in L]) ** (1.0 / 3.0)) / 50) + 1))
if 'minimum state seats' in f :
    if f['minimum state seats'] == 'Constituencies' :
        f['minimum state seats'] = -1
    else :
        f['minimum state seats'] = int(f['minimum state seats'])
else :
    if f['apportionment method'] == 'Huntington-Hill' :
        f['minimum state seats'] = 1
    else :
        f['minimum state seats'] = 0
for m in ['minimum initial census state seats',
          'minimum revised census state seats'] :
    if m not in f :
        f[m] = f['minimum state seats']
    else :
        if f[m] == 'Constituencies' :
            f[m] = -1
        else :
            f[m] = int(f[m])
if 'maximum state seats' in f :
    f['maximum state seats'] = int(f['maximum state seats'])
else :
    f['maximum state seats'] = N
for m in ['maximum initial census state seats',
          'maximum revised census state seats'] :
    if m not in f :
        f[m] = f['maximum state seats']
    else :
        f[m] = int(f[m])

# Index
I = [(s, p) for s in L for p in P]

# initial census apportionment
if f['initial census apportionment method'] not in ['New Zealand', 'Scotland'] :
    print 'Initial census apportionment at'.rjust(32), time.asctime()
K = f['initial census apportionment divisor']
mns = f['minimum initial census state seats']
mxs = f['maximum initial census state seats']
if f['initial census apportionment method'] in ['Ebert', 'VarPhrag'] :
    mA = ConcreteModel()
    # Census, Loads, Seats
    mA.C = Param(L, mutable = True)
    for s in L :
        mA.C[s] = C[s]
    mA.Loads = Var(L, bounds = (0.0, 1.0), within = NonNegativeReals)
    if 'fixed regional list tier' in f :
        mA.N = sum([D[i[0]][i[1]] for i in I])
    else :
        mA.N = N
    # constraints
    mA.Consts = ConstraintList()
    for s in L :
        if mns >= sum(D[s][p] for p in P) :
            mA.Consts.add(expr = mA.C[s] * (mA.Loads[s] / K) >= mns)
        else :
            mA.Consts.add(expr = mA.C[s] * (mA.Loads[s] / K) >= sum(D[s][p] for p in P))
        mA.Consts.add(expr = mA.C[s] * (mA.Loads[s] / K) <= mxs)
    mA.Consts.add(expr = sum(mA.C[s] * (mA.Loads[s] / K) for s in L) == mA.N)
    # objective
    mA.O = Objective(expr = sum(mA.C[s] * (mA.Loads[s] ** 2.0) for s in L), sense = minimize)
    # solve
    solver_mA = SolverFactory('ipopt')
    solver_mA.solve(mA)
    censusSeats = {s: int(value(mA.C[s] * (mA.Loads[s] / K))) for s in L}
    while sum([censusSeats[s] for s in L]) < mA.N :
        R = {}
        for s in L :
            if value(mA.C[s] * (mA.Loads[s] / K)) - censusSeats[s] not in R :
                R[value(mA.C[s] * (mA.Loads[s] / K)) - censusSeats[s]] = []
            R[value(mA.C[s] * (mA.Loads[s] / K)) - censusSeats[s]].append(s)
        Rk = R.keys()
        Rk.sort()
        Rk.reverse()
        for s in R[Rk[0]] :
            censusSeats[s] += 1
if f['initial census apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['initial census apportionment method'] :
    censusSeats = {s: [0, 0.0] for s in L}
    for s in L :
        if mns > sum(D[s][p] for p in P) :
            censusSeats[s][0] = mns
        else :
            censusSeats[s][0] = sum(D[s][p] for p in P)
    if 'fixed regional list tier' in f :
        z = sum([D[i[0]][i[1]] for i in I])
    else :
        z = N
    while sum([censusSeats[s][0] for s in L]) < z :
        for s in L :
            censusSeats[s][1] = 0.0
            if censusSeats[s][0] < mxs :
                censusSeats[s][1] = C[s] * (1.0 / (1.0 + (censusSeats[s][0] / K)))
        A = max([censusSeats[s][1] for s in L])
        for s in L :
            if censusSeats[s][1] == A :
                censusSeats[s][0] += 1
    censusSeats = {s: censusSeats[s][0] for s in L}
if f['initial census apportionment method'] == 'HarePhrag' :
    pass
if f['initial census apportionment method'] == 'MaximinKS' :
    mA = {}
    mA['MaximinKS'] = LpProblem('MaximinKS', LpMaximize)
    # Apportionment, Initial Bargaining Points, Minimum Relative Benefit, Relative Benefits
    mA['A'] = LpVariable.dicts('Apportionment', range(len(L)), lowBound = 0, upBound = N)
    mA['IBP'] = {s: float(mxs) * C[L[s]] / sum([C[t] for t in L]) for s in range(len(L))}
    mA['MRB'] = LpVariable('MinimumRelativeBenefit', lowBound = 0.0, upBound = 1.0)
    mA['RB'] = {s: (mA['A'][s] - mA['IBP'][s]) / (mxs - mA['IBP'][s]) for s in range(len(L))}
    # constraints
    for s in range(len(L)) :
        if mns >= sum(D[L[s]][p] for p in P) :
            mA['MaximinKS'] += mA['A'][s] >= mns
        else :
            mA['MaximinKS'] += mA['A'][s] >= sum(D[L[s]][p] for p in P)
        mA['MaximinKS'] += mA['A'][s] <= mxs
    if 'fixed regional list tier' in f :
        mA['MaximinKS'] += sum(mA['A'][s] for s in range(len(L))) == sum(D[i[0]][i[1]] for i in I)
    else :
        mA['MaximinKS'] += sum(mA['A'][s] for s in range(len(L))) == N
    for s in range(len(L)) :
        mA['MaximinKS'] += mA['MRB'] <= mA['RB'][s]
    # objective
    mA['MaximinKS'] += lpSum(mA['MRB'])
    # solve
    mA['sol'] = mA['MaximinKS'].solve(GLPK_CMD())
    censusSeats = {L[s]: int(round(mA['A'][s].varValue)) for s in range(len(L))}
    if 'fixed regional list tier' in f :
        z = sum([D[i[0]][i[1]] for i in I])
    else :
        z = N
    while sum([censusSeats[s] for s in L]) < z :
        R = {}
        for s in range(len(L)) :
            if mA['A'][s].varValue - censusSeats[L[s]] not in R :
                R[mA['A'][s].varValue - censusSeats[L[s]]] = []
            R[mA['A'][s].varValue - censusSeats[L[s]]].append(s)
        Rk = R.keys()
        Rk.sort()
        Rk.reverse()
        for s in R[Rk[0]] :
            censusSeats[L[s]] += 1
if f['initial census apportionment method'] == 'MaxPhrag' :
    mA = {}
    mA['MaxPhrag'] = LpProblem('MaxPhragmen', LpMinimize)
    # Loads, MaxLoad
    mA['Loads'] = LpVariable.dicts('Loads', range(len(L)), lowBound = 0.0, upBound = 1.0)
    mA['MaxLoad'] = LpVariable('MaxLoad', lowBound = 0.0)
    # constraints
    for s in range(len(L)) :
        if mns >= sum(D[L[s]][p] for p in P) :
            mA['MaxPhrag'] += C[L[s]] * (mA['Loads'][s] / K) >= mns
        else :
            mA['MaxPhrag'] += C[L[s]] * (mA['Loads'][s] / K) >= sum(D[L[s]][p] for p in P)
        mA['MaxPhrag'] += C[L[s]] * (mA['Loads'][s] / K) <= mxs
    if 'fixed regional list tier' in f :
        mA['MaxPhrag'] += sum(C[L[s]] * (mA['Loads'][s] / K) for s in range(len(L))) == sum(D[i[0]][i[1]] for i in I)
    else :
        mA['MaxPhrag'] += sum(C[L[s]] * (mA['Loads'][s] / K) for s in range(len(L))) == N
    for s in range(len(L)) :
        mA['MaxPhrag'] += mA['MaxLoad'] >= mA['Loads'][s]
    # objective
    mA['MaxPhrag'] += lpSum(mA['MaxLoad'])
    # solve
    mA['sol'] = mA['MaxPhrag'].solve(GLPK_CMD())
    censusSeats = {L[s]: int(C[L[s]] * (mA['Loads'][s].varValue / K)) for s in range(len(L))}
    if 'fixed regional list tier' in f :
        z = sum([D[i[0]][i[1]] for i in I])
    else :
        z = N
    while sum([censusSeats[s] for s in L]) < z :
        R = {}
        for s in range(len(L)) :
            if C[L[s]] * (mA['Loads'][s].varValue / K) - censusSeats[L[s]] not in R :
                R[C[L[s]] * (mA['Loads'][s].varValue / K) - censusSeats[L[s]]] = []
            R[C[L[s]] * (mA['Loads'][s].varValue / K) - censusSeats[L[s]]].append(s)
        Rk = R.keys()
        Rk.sort()
        Rk.reverse()
        for s in R[Rk[0]] :
            censusSeats[L[s]] += 1
if f['initial census apportionment method'] == 'New Zealand' :
    censusSeats = {L[0]: N}
if f['initial census apportionment method'] == 'PAV' :
    mA = ConcreteModel()
    # Range, Rangeset, Apportionment, Census, Scoreset, Scores
    mA.R = range(min([int(float(max([C[s] for s in L])) / sum([C[s] for s in L]) * N) + 2, mxs + 1]))
    mA.Rset = Set(initialize = mA.R)
    mA.A = Var(L, within = mA.Rset)
    mA.C = Param(L, mutable = True)
    for s in L :
        mA.C[s] = C[s]
    mA.ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mA.R]
    mA.Scores = Var(L, domain = NonNegativeReals)
    # constraints
    mA.Consts = ConstraintList()
    for s in L :
        if mns > sum(D[s][p] for p in P) :
            mA.Consts.add(expr = mA.A[s] >= mns)
        else :
            mA.Consts.add(expr = mA.A[s] >= sum(D[s][p] for p in P))
        mA.Consts.add(expr = mA.A[s] <= mxs)
    if 'fixed regional list tier' in f :
        mA.Consts.add(expr = sum(mA.A[s] for s in L) == sum(D[i[0]][i[1]] for i in I))
    else :
        mA.Consts.add(expr = sum(mA.A[s] for s in L) == N)
    mA.Piece = Piecewise(L, mA.Scores, mA.A, pw_pts = mA.R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mA.ScoreSet)
    # objective
    mA.O = Objective(expr = summation(mA.C, mA.Scores), sense = maximize)
    # solve
    solver_mA = SolverFactory('ipopt')
    solver_mA.solve(mA)
    censusSeats = {s: int(round(value(mA.A[s]))) for s in L}
if f['initial census apportionment method'] == 'Scotland' :
    censusSeats = {s: sum([D[s][p] for p in P]) for s in L}
if f['initial census apportionment method'] == 'SeqPhrag' :
    censusSeats = {s: [0, 0.0, 0.0] for s in L}
    for s in L :
        if mns > sum(D[s][p] for p in P) :
            z = mns
        else :
            z = sum(D[s][p] for p in P)
        while censusSeats[s][0] < z :
            censusSeats[s][0] += 1
            censusSeats[s][2] = ((1.0 / K) + (C[s] * censusSeats[s][2])) / C[s]
    if 'fixed regional list tier' in f :
        z = sum([D[i[0]][i[1]] for i in I])
    else :
        z = N
    while sum([censusSeats[s][0] for s in L]) < z :
        for s in [x for x in L if censusSeats[x][0] < mxs] :
            censusSeats[s][1] = ((1.0 / K) + (C[s] * censusSeats[s][2])) / C[s]
        A = min([censusSeats[s][1] for s in [x for x in L if censusSeats[x][0] < mxs]])
        for s in [x for x in L if censusSeats[x][0] < mxs] :
            if censusSeats[s][1] == A :
                censusSeats[s][0] += 1
                censusSeats[s][2] = A
    censusSeats = {s: censusSeats[s][0] for s in L}
if f['initial census apportionment method'] == 'VarKS' :
    mA = ConcreteModel()
    # Apportionment, Census, Initial Bargaining Points, Relative Benefits, Seats
    mA.A = Var(L, bounds = (0, N), within = Integers)
    mA.C = Param(L, mutable = True)
    for s in L :
        mA.C[s] = C[s]
    mA.IBP = Param(L, mutable = True)
    for s in L :
        mA.IBP[s] = C[s] * float(mxs) / sum([C[t] for t in L])
    mA.RB = Var(L, bounds = (0.0, 1.0), within = NonNegativeReals)
    if 'fixed regional list tier' in f :
        mA.N = sum([D[i[0]][i[1]] for i in I])
    else :
        mA.N = N
    # constraints
    mA.Consts = ConstraintList()
    for s in L :
        if mns >= sum(D[s][p] for p in P) :
            mA.Consts.add(expr = mA.A[s] >= mns)
        else :
            mA.Consts.add(expr = mA.A[s] >= sum(D[s][p] for p in P))
        mA.Consts.add(expr = mA.A[s] <= mxs)
        mA.Consts.add(expr = mA.RB[s] == (mA.A[s] - mA.IBP[s]) / (mxs - mA.IBP[s]))
    mA.Consts.add(expr = sum(mA.A[s] for s in L) == mA.N)
    # objective
    mA.O = Objective(expr = sum(C[s] * (mA.RB[s] ** 2.0) for s in L), sense = minimize)
    # solve
    solver_mA = SolverFactory('ipopt')
    solver_mA.solve(mA)
    censusSeats = {s: int(value(mA.A[s])) for s in L}
    while sum([censusSeats[s] for s in L]) < mA.N :
        R = {}
        for s in L :
            if value(mA.A[s]) - censusSeats[s] not in R :
                R[value(mA.A[s]) - censusSeats[s]] = []
            R[value(mA.A[s]) - censusSeats[s]].append(s)
        Rk = R.keys()
        Rk.sort()
        Rk.reverse()
        for s in R[Rk[0]] :
            censusSeats[s] += 1
if 'fixed regional list tier' in f :
    for s in L :
        censusSeats[s] += int(f['fixed regional list tier'])

# state party apportionments
if f['state party apportionment method'] != 'New Zealand' and len(L) > 1 :
    print 'State party apportionments at'.rjust(32), time.asctime()
else :
    print 'Party apportionment at'.rjust(32), time.asctime()
K = f['state party apportionment divisor']
if f['state party apportionment method'] in ['Ebert', 'VarPhrag'] :
    mB = []
    partyMins_B = {s: {p: 0 for p in P} for s in L}
    solvers_mB = []
    for s in L :
        mB.append(ConcreteModel())
        # Ballots subsets, Index, Loads
        mB[-1].Lsub = [b for b in range(len(B)) if B[b][1] == s]
        mB[-1].Bsub = {b: [p for p in B[b][2]] for b in mB[-1].Lsub}
        mB[-1].Psub = {p: [b for b in mB[-1].Bsub if p in mB[-1].Bsub[b]] for p in P}
        mB[-1].I = []
        for b in mB[-1].Bsub :
            for p in mB[-1].Bsub[b] :
                mB[-1].I.append((b, p))
        mB[-1].Loads = Var(mB[-1].I, bounds = (0.0, 1.0), within = NonNegativeReals)
        # constraints
        mB[-1].Consts = ConstraintList()
        mB[-1].Consts.add(expr = sum(B[i[0]][0] * mB[-1].Loads[i] for i in mB[-1].I) == censusSeats[s])
        # objective
        mB[-1].O = Objective(expr = sum(B[b][0] * (sum(mB[-1].Loads[(b, p)] for p in mB[-1].Bsub[b]) ** 2.0) for b in mB[-1].Bsub), sense = minimize)
        # solve
        solvers_mB.append(SolverFactory('ipopt'))
        solvers_mB[-1].solve(mB[-1])
        for p in P :
            partyMins_B[s][p] = int(sum(B[b][0] * value(mB[-1].Loads[(b, p)]) for b in mB[-1].Psub[p]))
        while sum([partyMins_B[s][p] for p in P]) < censusSeats[s] :
            mB[-1].R = {}
            for p in P :
                if sum([B[b][0] * value(mB[-1].Loads[(b, p)]) for b in mB[-1].Psub[p]]) - partyMins_B[s][p] not in mB[-1].R :
                    mB[-1].R[sum([B[b][0] * value(mB[-1].Loads[(b, p)]) for b in mB[-1].Psub[p]]) - partyMins_B[s][p]] = []
                mB[-1].R[sum([B[b][0] * value(mB[-1].Loads[(b, p)]) for b in mB[-1].Psub[p]]) - partyMins_B[s][p]].append(p)
            mB[-1].Rk = mB[-1].R.keys()
            mB[-1].Rk.sort()
            mB[-1].Rk.reverse()
            for p in mB[-1].R[mB[-1].Rk[0]] :
                partyMins_B[s][p] += 1
        for p in P :
            partyMins_B[s][p] = max([D[s][p], partyMins_B[s][p]])
    om = sum([max([0, sum([partyMins_B[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['apportionment method'] :
    partyMins = {s: {p: [sum([B[b][0] for b in range(len(B)) if B[b][1] == s and p in B[b][2]]), 0, 0.0] for p in P} for s in L}
    for s in L :
        while sum([partyMins[s][p][1] for p in P]) < censusSeats[s] :
            for p in P :
                partyMins[s][p][2] = partyMins[s][p][0] * (1.0 / (1.0 + (partyMins[s][p][1] / K)))
            A = max([partyMins[s][p][2] for p in P])
            for p in P :
                if partyMins[s][p][2] == A :
                    partyMins[s][p][1] += 1
    partyMins = {s: {p: max([D[s][p], partyMins[s][p][1]]) for p in P} for s in L}
    om = sum([max([0, sum([partyMins[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'HarePhrag' :
    pass
if f['state party apportionment method'] == 'MaximinKS' :
    mB = []
    partyMins_B = {s: {p: 0 for p in P} for s in L}
    solvers_mB = []
    for s in L :
        mB.append({})
        mB[-1]['MaximinKS'] = LpProblem('MaximinKS', LpMaximize)
        # Ballots subset, Elected, Initial Bargaining Points, Minimum Relative Benefit, Relative Benefits
        mB[-1]['Bsub'] = [n for n in range(len(B)) if B[n][1] == s]
        mB[-1]['Elected'] = LpVariable.dicts('Elected', range(len(P)), lowBound = 0, upBound = censusSeats[s])
        mB[-1]['IBP'] = {b: censusSeats[s] * sum([B[c][0] * (float(sum([1 for p in B[c][2] if p in B[b][2]])) / len(B[c][2])) for c in mB[-1]['Bsub']]) / sum([B[d][0] for d in mB[-1]['Bsub']]) for b in mB[-1]['Bsub']}
        mB[-1]['MRB'] = LpVariable('MinimumRelativeBenefit', lowBound = 0.0, upBound = 1.0)
        mB[-1]['RB'] = {b: (sum(mB[-1]['Elected'][p] for p in range(len(P)) if P[p] in B[b][2]) - mB[-1]['IBP'][b]) / (float(censusSeats[s]) - mB[-1]['IBP'][b]) for b in mB[-1]['Bsub']}
        # constraints
        for b in mB[-1]['Bsub'] :
            mB[-1]['MaximinKS'] += mB[-1]['MRB'] <= mB[-1]['RB'][b]
        mB[-1]['MaximinKS'] += sum(mB[-1]['Elected'][p] for p in range(len(P))) == censusSeats[s]
        # objective
        mB[-1]['MaximinKS'] += lpSum(mB[-1]['MRB'])
        # solve
        mB[-1]['sol'] = mB[-1]['MaximinKS'].solve(GLPK_CMD())
        for p in range(len(P)) :
            partyMins_B[s][P[p]] = max([D[s][P[p]], int(round(mB[-1]['Elected'][p].varValue))])
    om = sum([max([0, sum([partyMins_B[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'MaxPhrag' :
    mB = []
    partyMins_B = {s: {p: 0 for p in P} for s in L}
    solvers_mB = []
    for s in L :
        mB.append({})
        mB[-1]['MaxPhrag'] = LpProblem('MaxPhragmen', LpMinimize)
        # Ballots subsets, Elected, Index, Loads, MaxLoad
        mB[-1]['Bsub'] = {b: [p for p in range(len(P)) if P[p] in B[b][2]] for b in range(len(B)) if B[b][1] == s}
        mB[-1]['Psub'] = {p: [b for b in mB[-1]['Bsub'] if p in mB[-1]['Bsub'][b]] for p in range(len(P))}
        mB[-1]['Elected'] = LpVariable.dicts('Elected', range(len(P)), 0, censusSeats[s], LpInteger)
        mB[-1]['I'] = []
        for b in mB[-1]['Bsub'] :
            for p in mB[-1]['Bsub'][b] :
                mB[-1]['I'].append((b, p))
        mB[-1]['Loads'] = LpVariable.dicts('Loads', mB[-1]['I'], lowBound = 0.0, upBound = 1.0)
        mB[-1]['MaxLoad'] = LpVariable('MaxLoad', lowBound = 0.0, upBound = 1.0)
        # constraints
        mB[-1]['MaxPhrag'] += sum(B[i[0]][0] * mB[-1]['Loads'][i] for i in mB[-1]['I']) == censusSeats[s]
        for b in mB[-1]['Bsub'] :
            mB[-1]['MaxPhrag'] += mB[-1]['MaxLoad'] >= sum(mB[-1]['Loads'][i] for i in mB[-1]['I'] if i[0] == b)
        for p in range(len(P)) :
            mB[-1]['MaxPhrag'] += mB[-1]['Elected'][p] == sum(B[i[0]][0] * mB[-1]['Loads'][i] for i in mB[-1]['I'] if i[1] == p)
        # objective
        mB[-1]['MaxPhrag'] += lpSum(mB[-1]['MaxLoad'])
        solvers_mB.append(mB[-1]['MaxPhrag'].solve(GLPK_CMD()))
        for p in range(len(P)) :
            partyMins_B[s][P[p]] = max([D[s][P[p]], int(round(mB[-1]['Elected'][p].varValue))])
    om = sum([max([0, sum([partyMins_B[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'New Zealand' :
    partySeats = {s: {p: [sum([B[b][0] for b in range(len(B)) if B[b][1] == s and p in B[b][2]]), 0, 0.0] for p in P} for s in L}
    for s in L :
        while sum([1 for p in P if partySeats[s][p][1] < D[s][p]]) > 0 or sum([partySeats[s][p][1] for p in P]) < censusSeats[s] :
            for p in P :
                partySeats[s][p][2] = partySeats[s][p][0] * (1.0 / (1.0 + (partySeats[s][p][1] / K)))
            A = max([partySeats[s][p][2] for p in P])
            for p in P :
                if partySeats[s][p][2] == A :
                    partySeats[s][p][1] += 1
    finalSeats = {i: partySeats[i[0]][i[1]][1] for i in I}
    om = sum([max([0, sum([finalSeats[(s, p)] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'PAV' :
    mB = []
    partyMins_B = {s: {p: 0 for p in P} for s in L}
    solvers_mB = []
    for s in L :
        mB.append(ConcreteModel())
        # Range, Rangeset, Ballots subset, Ballots, Elected, Scoreset, Scores, Votes
        mB[-1].R = range(int(float(max([sum([B[b][0] for b in range(len(B)) if B[b][1] == s and p in B[b][2]]) for p in P])) / sum([B[b][0] for b in range(len(B)) if B[b][1] == s]) * N) + 2)
        mB[-1].Rset = Set(initialize = mB[-1].R)
        mB[-1].Bsub = [n for n in range(len(B)) if B[n][1] == s]
        mB[-1].B = Var(mB[-1].Bsub, within = mB[-1].Rset)
        mB[-1].E = Var(P, within = mB[-1].Rset)
        mB[-1].ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mB[-1].R]
        mB[-1].Scores = Var(mB[-1].Bsub, domain = NonNegativeReals)
        mB[-1].V = Param(mB[-1].Bsub, mutable = True)
        for b in mB[-1].Bsub :
            mB[-1].V[b] = B[b][0]
        # constraints
        mB[-1].Consts = ConstraintList()
        for b in mB[-1].Bsub :
            mB[-1].Consts.add(expr = mB[-1].B[b] == sum(mB[-1].E[p] for p in P if p in B[b][2]))
        mB[-1].Consts.add(expr = sum(mB[-1].E[p] for p in P) == censusSeats[s])
        mB[-1].Piece = Piecewise(mB[-1].Bsub, mB[-1].Scores, mB[-1].B, pw_pts = mB[-1].R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mB[-1].ScoreSet)
        # objective
        mB[-1].O = Objective(expr = summation(mB[-1].V, mB[-1].Scores), sense = maximize)
        # solve
        solvers_mB.append(SolverFactory('ipopt'))
        solvers_mB[-1].solve(mB[-1])
        for p in P :
            partyMins_B[s][p] = max([D[s][p], int(round(value(mB[-1].E[p])))])
    om = sum([max([0, sum([partyMins_B[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'Scotland' :
    partySeats = {s: {p: [sum([B[b][0] for b in range(len(B)) if B[b][1] == s and p in B[b][2]]), D[s][p], 0.0] for p in P} for s in L}
    for s in L :
        while sum([partySeats[s][p][1] for p in P]) < censusSeats[s] :
            for p in P :
                partySeats[s][p][2] = partySeats[s][p][0] * (1.0 / (1.0 + (partySeats[s][p][1] / K)))
            A = max([partySeats[s][p][2] for p in P])
            for p in P :
                if partySeats[s][p][2] == A :
                    partySeats[s][p][1] += 1
    finalSeats = {i: partySeats[i[0]][i[1]][1] for i in I}
    om = 0
if f['state party apportionment method'] == 'SeqPhrag' :
    partyMins = {s: {p: [D[s][p], 0.0] for p in P} for s in L}
    for s in L :
        Loads = {n: 0.0 for n in range(len(B)) if B[n][1] == s}
        while sum([partyMins[s][p][0] for p in P]) < censusSeats[s] :
            for p in [x for x in P if sum([B[b][0] for b in range(len(B)) if B[b][1] == s and x in B[b][2]]) > 0] :
                partyMins[s][p][1] = (1.0 + sum([(B[b][0] * Loads[b]) for b in Loads if p in B[b][2]])) / sum([B[b][0] for b in Loads if p in B[b][2]])
            A = min([partyMins[s][p][1] for p in P if 'elimination' not in f or partyMins[s][p][0] == 0])
            for p in P :
                if partyMins[s][p][1] == A :
                    partyMins[s][p][0] += 1
                    for b in Loads :
                        if p in B[b][2] :
                            Loads[b] == A
    partyMins = {s: {p: max([D[s][p], partyMins[s][p][0]]) for p in P} for s in L}
    om = sum([max([0, sum([partyMins[s][p] for p in P]) - censusSeats[s]]) for s in L])
if f['state party apportionment method'] == 'VarKS' :
    pass

# national party apportionment
K = f['national party apportionment divisor']
if f['national party apportionment method'] not in ['New Zealand', 'Scotland'] :
    print 'National party apportionemnt at'.rjust(32), time.asctime()
if f['national party apportionment method'] in ['Ebert', 'VarPhrag'] :
    mC = ConcreteModel()
    # Index, Loads
    mC.I = [(b, p) for b in range(len(B)) for p in P]
    mC.Loads = Var(mC.I, bounds = (0.0, 1.0), within = NonNegativeReals)
    # constraints
    mC.Consts = ConstraintList()
    for i in mC.I :
        if i[1] not in B[i[0]][2] :
            mC.Consts.add(expr = mC.Loads[i] == 0.0)
    mC.Consts.add(expr = sum(B[i[0]][0] * mC.Loads[i] for i in mC.I) == (N + om))
    # objective
    mC.O = Objective(expr = sum(B[b][0] * (sum([mC.Loads[(b, p)] for p in P]) ** 2.0) for b in range(len(B))), sense = minimize)
    # solve
    solver_mC = SolverFactory('ipopt')
    solver_mC.solve(mC)
    partyMins_C = {p: [float(sum([partyMins_B[s][p] for s in L])), round(sum([B[b][0] * value(mC.Loads[(b, p)]) for b in range(len(B))]))] for p in P}
    for p in P :
        partyMins_C[p].append(partyMins_C[p][0] / partyMins_C[p][1])
    for p in P :
        partyMins_C[p].append(int(partyMins_C[p][1] * max([partyMins_C[q][2] for q in P])))
    lm = sum([partyMins_C[p][3] for p in P]) - (N + om)
if f['national party apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['national party apportionment method'] :
    partySeats = {p: [sum([B[b][0] for b in range(len(B)) if p in B[b][2]]), 0, 0.0] for p in P}
    while sum([1 for p in P if partySeats[p][1] < sum([partyMins[s][p] for s in L])]) > 0 or sum([partySeats[p][1] for p in P]) < sum([partyMins[i[0]][i[1]] for i in I]) :
        for p in P :
            partySeats[p][2] = partySeats[p][0] * (1.0 / (1.0 + (partySeats[p][1] / K)))
        A = max([partySeats[p][2] for p in P])
        for p in P :
            if partySeats[p][2] == A :
                partySeats[p][1] += 1
    lm = sum([partySeats[p][1] for p in P]) - (N + om)
if f['national party apportionment method'] == 'HarePhrag' :
    pass
if f['national party apportionment method'] == 'MaximinKS' :
    pass
if f['national party apportionment method'] == 'MaxPhrag' :
    mC = {}
    mC['MaxPhrag'] = LpProblem('MaxPhragmen', LpMinimize)
    # Elected, Index, Loads, MaxLoad
    mC['Elected'] = LpVariable.dicts('Elected', range(len(P)), 0, N + om, LpInteger)
    mC['I'] = [(b, p) for b in range(len(B)) for p in range(len(P))]
    mC['Loads'] = LpVariable.dicts('Loads', mC['I'], lowBound = 0.0, upBound = 1.0)
    mC['MaxLoad'] = LpVariable('MaxLoad', lowBound = 0.0, upBound = 1.0)
    # constraints
    for i in mC['I'] :
        if P[i[1]] not in B[i[0]][2] :
            mC['MaxPhrag'] += mC['Loads'][i] == 0.0
    mC['MaxPhrag'] += sum(B[i[0]][0] * mC['Loads'][i] for i in mC['I']) == (N + om)
    for b in range(len(B)) :
        mC['MaxPhrag'] += mC['MaxLoad'] >= sum(mC['Loads'][(b, p)] for p in range(len(P)))
    for p in range(len(P)) :
        mC['MaxPhrag'] += mC['Elected'][p] == sum(B[b][0] * mC['Loads'][(b, p)] for b in range(len(B)))
    mC['MaxPhrag'] += lpSum(mC['MaxLoad'])
    mC['sol'] = mC['MaxPhrag'].solve(GLPK_CMD())
    partyMins_C = {p: [float(sum([partyMins_B[s][P[p]] for s in L])), round(sum([B[b][0] * mC['Loads'][(b, p)].varValue for b in range(len(B))]))] for p in range(len(P))}
    for p in range(len(P)) :
        partyMins_C[p].append(partyMins_C[p][0] / partyMins_C[p][1])
    for p in range(len(P)) :
        partyMins_C[p].append(int(partyMins_C[p][1] * max([partyMins_C[q][2] for q in range(len(P))])))
    lm = sum([partyMins_C[p][3] for p in range(len(P))]) - (N + om)
if f['national party apportionment method'] == 'New Zealand' :
    lm = 0      # overhangs compensated but unleveled in New Zealand system
if f['national party apportionment method'] == 'PAV' :
    mC = ConcreteModel()
    # Range, Rangeset, Ballots, Elected, Scoreset, Scores, Votes
    mC.R = range(int(float(max([sum([B[b][0] for b in range(len(B)) if p in B[b][2]]) for p in P])) / sum([B[b][0] for b in range(len(B))]) * (N + om)) + 2)
    mC.Rset = Set(initialize = mC.R)
    mC.B = Var(range(len(B)), within = mC.Rset)
    mC.E = Var(P, within = mC.Rset)
    mC.ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mC.R]
    mC.Scores = Var(range(len(B)), domain = NonNegativeReals)
    mC.V = Param(range(len(B)), mutable = True)
    for b in range(len(B)) :
        mC.V[b] = B[b][0]
    # constraints
    mC.Consts = ConstraintList()
    for b in range(len(B)) :
        mC.Consts.add(expr = mC.B[b] == sum(mC.E[p] for p in P if p in B[b][2]))
    mC.Consts.add(expr = sum(mC.E[p] for p in P) == (N + om))
    mC.Piece = Piecewise(range(len(B)), mC.Scores, mC.B, pw_pts = mC.R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mC.ScoreSet)
    # objective
    mC.O = Objective(expr = summation(mC.V, mC.Scores), sense = maximize)
    # solve
    solver_mC = SolverFactory('ipopt')
    solver_mC.solve(mC)
    partyMins_C = {p: [float(sum([partyMins_B[s][p] for s in L])), round(value(mC.E[p]))] for p in P}
    for p in P :
        partyMins_C[p].append(partyMins_C[p][0] / partyMins_C[p][1])
    for p in P :
        partyMins_C[p].append(int(partyMins_C[p][1] * max([partyMins_C[q][2] for q in P])))
    lm = sum([partyMins_C[p][3] for p in P]) - (N + om)
if f['national party apportionment method'] == 'Scotland' :
    # overhangs uncompensated in Scottish system
    lm = 0      # overhangs uncompensated in Scottish system
if f['national party apportionment method'] == 'SeqPhrag' :
    partySeats = {p: [0, 0.0] for p in P}
    Loads = {b: 0.0 for b in range(len(B))}
    while sum([1 for p in P if partySeats[p][0] < sum([partyMins[s][p] for s in L])]) > 0 or sum([partySeats[p][0] for p in P]) < sum([partyMins[i[0]][i[1]] for i in I]) :
        for p in P :
            partySeats[p][1] = (1.0 + sum([(B[b][0] * Loads[b]) for b in Loads if p in B[b][2]])) / sum([B[b][0] for b in Loads if p in B[b][2]])
        A = min([partySeats[p][1] for p in P if 'elimination' not in f or partySeats[p][0] == 0])
        for p in P :
            if partySeats[p][1] == A :
                partySeats[p][0] += 1
                for b in Loads :
                    if p in B[b][2] :
                        Loads[b] == A
    partySeats = {p: partySeats[p][0] for p in P}
    lm = sum([partySeats[p] for p in P]) - (N + om)
if f['national party apportionment method'] == 'VarKS' :
    pass

# leveling seat apportionment
K = f['leveling seat apportionment divisor']
if f['leveling seat apportionment method'] not in ['New Zealand', 'Scotland'] :
    print 'Leveling seat apportionment at'.rjust(32), time.asctime()
if f['leveling seat apportionment method'] in ['Ebert', 'VarPhrag'] :
    mD = ConcreteModel()
    # Index, Loads
    mD.I = [(b, p) for b in range(len(B)) for p in P]
    mD.Loads = Var(mD.I, bounds = (0.0, 1.0), within = NonNegativeReals)
    # constraints
    mD.Consts = ConstraintList()
    for i in mD.I :
        if i[1] not in B[i[0]][2] :
            mD.Consts.add(expr = mD.Loads[i] == 0.0)
    mD.Consts.add(expr = sum(B[i[0]][0] * mD.Loads[i] for i in mD.I) == (N + om + lm))
    # objective
    mD.O = Objective(expr = sum(B[b][0] * (sum([mD.Loads[(b, p)] for p in P]) ** 2.0) for b in range(len(B))), sense = minimize)
    # solve
    solver_mD = SolverFactory('ipopt')
    solver_mD.solve(mD)
if f['leveling seat apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['leveling seat apportionment method'] :
    partyMins_D = {p: [sum([B[b][0] for b in range(len(B)) if p in B[b][2]]), 0, 0.0] for p in P}
    while sum([1 for p in P if partyMins_D[p][1] < partySeats[p][1]]) > 0 or sum([partyMins_D[p][1] for p in P]) < (N + om + lm) :
        for p in P :
            partyMins_D[p][2] = partyMins_D[p][0] * (1.0 / (1.0 + (partyMins_D[p][1] / K)))
        A = max([partyMins_D[p][2] for p in P])
        for p in P :
            if partyMins_D[p][2] == A :
                partyMins_D[p][1] += 1
if f['leveling seat apportionment method'] == 'HarePhrag' :
    pass
if f['leveling seat apportionment method'] == 'MaximinKS' :
    pass
if f['leveling seat apportionment method'] == 'MaxPhrag' :
    mD = {}
    mD['MaxPhrag'] = LpProblem('MaxPhragmen', LpMinimize)
    # Elected, Index, Loads, MaxLoad
    mD['Elected'] = LpVariable.dicts('Elected', range(len(P)), 0, N + om + lm, LpInteger)
    mD['I'] = [(b, p) for b in range(len(B)) for p in range(len(P))]
    mD['Loads'] = LpVariable.dicts('Loads', mD['I'], lowBound = 0.0, upBound = 1.0)
    mD['MaxLoad'] = LpVariable('MaxLoad', lowBound = 0.0, upBound = 1.0)
    # constraints
    for i in mD['I'] :
        if P[i[1]] not in B[i[0]][2] :
            mD['MaxPhrag'] += mD['Loads'][i] == 0.0
    mD['MaxPhrag'] += sum(B[i[0]][0] * mD['Loads'][i] for i in mD['I']) == (N + om + lm)
    for b in range(len(B)) :
        mD['MaxPhrag'] += mD['MaxLoad'] >= sum(mD['Loads'][(b, p)] for p in range(len(P)))
    for p in range(len(P)) :
        mD['MaxPhrag'] += mD['Elected'][p] == sum(B[b][0] * mD['Loads'][(b, p)] for b in range(len(B)))
    mD['MaxPhrag'] += lpSum(mD['MaxLoad'])
    mD['sol'] = mD['MaxPhrag'].solve(GLPK_CMD())
if f['leveling seat apportionment method'] == 'PAV' :
    mD = ConcreteModel()
    # Range, Rangeset, Ballots, Elected, Scoreset, Scores, Votes
    mD.R = range(int(float(max([sum([B[b][0] for b in range(len(B)) if p in B[b][2]]) for p in P])) / sum([B[b][0] for b in range(len(B))]) * (N + om + lm)) + 2)
    mD.Rset = Set(initialize = mD.R)
    mD.B = Var(range(len(B)), within = mD.Rset)
    mD.E = Var(P, within = mD.Rset)
    mD.ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mD.R]
    mD.Scores = Var(range(len(B)), domain = NonNegativeReals)
    mD.V = Param(range(len(B)), mutable = True)
    for b in range(len(B)) :
        mD.V[b] = B[b][0]
    # constraints
    mD.Consts = ConstraintList()
    for b in range(len(B)) :
        mD.Consts.add(expr = mD.B[b] == sum(mD.E[p] for p in P if p in B[b][2]))
    mD.Consts.add(expr = sum(mD.E[p] for p in P) == (N + om + lm))
    mD.Piece = Piecewise(range(len(B)), mD.Scores, mD.B, pw_pts = mD.R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mD.ScoreSet)
    # objective
    mD.O = Objective(expr = summation(mD.V, mD.Scores), sense = maximize)
    # solve
    solver_mD = SolverFactory('ipopt')
    solver_mD.solve(mD)
if f['leveling seat apportionment method'] == 'SeqPhrag' :
    partyMins_D = {p: [sum([B[b][0] for b in range(len(B)) if p in B[b][2]]), 0, 0.0] for p in P}
    D_Loads = {b: 0.0 for b in range(len(B))}
    while sum([1 for p in P if partyMins_D[p][1] < partySeats[p][1]]) > 0 or sum([partyMins_D[p][1] for p in P]) < (N + om + lm) :
        for p in P :
            partyMins_D[p][2] = (1.0 + sum([(B[b][0] * D_Loads[b]) for b in D_Loads if p in B[b][2]])) / sum([B[b][0] for b in D_Loads if p in B[b][2]])
        A = min([partyMins_D[p][2] for p in P if 'elimination' not in f or partyMins_D[p][1] == 0])
        for p in P :
            if partyMins_D[p][2] == A :
                partyMins_D[p][1] += 1
if f['leveling seat apportionment method'] == 'VarKS' :
    pass

# revised census apportionment
if 'census revision' in f and f['revised census apportionment method'] not in ['New Zealand', 'Scotland'] :
    print 'Revised census apportionment at'.rjust(32), time.asctime()
    K = f['revised census apportionment divisor']
    mns = f['minimum revised census state seats']
    mxs = f['maximum revised census state seats']
    if f['revised census apportionment method'] in ['Ebert', 'VarPhrag'] :
        pass
    if f['revised census apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['revised census apportionment method'] :
        pass
    if f['revised census apportionment method'] == 'HarePhrag' :
        pass
    if f['revised census apportionment method'] == 'MaximinKS' :
        pass
    if f['revised census apportionment method'] == 'MaxPhrag' :
        pass
    if f['revised census apportionment method'] == 'PAV' :
        mE = ConcreteModel()
        # Range, Rangeset, Apportionment, Census, Party minimums, Scoreset, Scores
        mE.R = range(int(float(max([C[s] for s in L])) / sum([C[s] for s in L]) * (N + om + lm)) + 2)
        mE.Rset = Set(initialize = mE.R)
        mE.A = Var(L, within = mE.Rset)
        mE.C = Param(L, mutable = True)
        for s in L :
            mE.C[s] = C[s]
        mE.partyMins_B = Param(I, mutable = True)
        for i in I :
            mE.partyMins_B[i] = partyMins_B[i[0]][i[1]]
        mE.ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mE.R]
        mE.Scores = Var(L, domain = NonNegativeReals)
        # constraints
        mE.Consts = ConstraintList()
        for s in L :
            mE.Consts.add(expr = mE.A[s] >= sum(mE.partyMins_B[(s, p)] for p in P))
        mE.Consts.add(expr = sum(mE.A[s] for s in L) == (N + om + lm))
        mE.Piece = Piecewise(L, mE.Scores, mE.A, pw_pts = mE.R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mE.ScoreSet)
        # objective
        mE.O = Objective(expr = summation(mE.C, mE.Scores), sense = maximize)
        # solve
        solver_mE = SolverFactory('ipopt')
        solver_mE.solve(mE)
    if f['revised census apportionment method'] == 'SeqPhrag' :
        pass
    if f['revised census apportionment method'] == 'VarKS' :
        pass

# final party apportionment
K = f['final party apportionment divisor']
if f['final party apportionment method'] not in ['New Zealand', 'Scotland'] :
    print 'Final party apportionment at'.rjust(32), time.asctime()
if f['final party apportionment method'] in ['Ebert', 'VarPhrag'] :
    mF = ConcreteModel()
if f['final party apportionment method'] == 'Germany' or 'Sainte-Lagu' in f['final party apportionment method'] :
    finalSeats = {i: [sum([B[b][0] for b in range(len(B)) if B[b][1] == i[0] and i[1] in B[b][2]]), D[i[0]][i[1]], 0.0] for i in I}
    for p in P :
        while sum([finalSeats[(s, p)][1] for s in L]) < partyMins_D[p][1] :
            for s in L :
                finalSeats[(s, p)][2] = finalSeats[(s, p)][0] * (1.0 / (1.0 + (finalSeats[(s, p)][1] / K)))
            A = max([finalSeats[(s, p)][2] for s in L])
            for s in L :
                if finalSeats[(s, p)][2] == A :
                    finalSeats[(s, p)][1] += 1
    finalSeats = {i: finalSeats[i][1] for i in I}
if f['final party apportionment method'] == 'HarePhrag' :
    pass
if f['final party apportionment method'] == 'MaximinKS' :
    pass
if f['final party apportionment method'] == 'MaxPhrag' :
    mF = {}
    mF['MaxPhrag'] = LpProblem('MaxPhragmen', LpMinimize)
    # Elected, Index, Loads, MaxLoad, partySeats, stateSeats
    mF['E'] = [(s, p) for s in range(len(L)) for p in range(len(P))]
    mF['Elected'] = LpVariable.dicts('Elected', mF['E'], 0, N + om + lm, LpInteger)
    mF['I'] = [(b, p) for b in range(len(B)) for p in range(len(P))]
    mF['Loads'] = LpVariable.dicts('Loads', mF['I'], lowBound = 0.0, upBound = 1.0)
    mF['MaxLoad'] = LpVariable('MaxLoad', lowBound = 0.0)
    mF['partySeats'] = [mD['Elected'][p].varValue for p in range(len(P))]
    if 'census revision' in f :
        mF['stateSeats'] = {s: revCensusSeats[L[s]] for s in range(len(L))}
    else :
        mF['stateSeats'] = {s: censusSeats[L[s]] for s in range(len(L))}
    # constraints
    for i in mF['I'] :
        if P[i[1]] not in B[i[0]][2] :
            mF['MaxPhrag'] += mF['Loads'][i] == 0.0
    mF['MaxPhrag'] += sum(mF['Elected'][e] for e in mF['E']) == (N + om + lm)
    for b in range(len(B)) :
        mF['MaxPhrag'] += mF['MaxLoad'] >= sum(mF['Loads'][(b, p)] for p in range(len(P)) if P[p] in B[b][2])
    for p in range(len(P)) :
        for s in range(len(L)) :
            mF['MaxPhrag'] += mF['Elected'][(s, p)] >= D[L[s]][P[p]]
            mF['MaxPhrag'] += sum(B[b][0] * mF['Loads'][(b, p)] for b in range(len(B)) if B[b][1] == L[s] and P[p] in B[b][2]) == mF['Elected'][(s, p)]
        mF['MaxPhrag'] += sum(mF['Elected'][(s, p)] for s in range(len(L))) == mF['partySeats'][p]
    for s in range(len(L)) :
        if 'census revision' in f :
            mF['MaxPhrag'] += sum(mF['Elected'][(s, p)] for p in range(len(P))) == mF['stateSeats'][s]
        else :
            mF['MaxPhrag'] += sum(mF['Elected'][(s, p)] for p in range(len(P))) >= mF['stateSeats'][s]
    mF['MaxPhrag'] += lpSum(mF['MaxLoad'])
    mF['sol'] = mF['MaxPhrag'].solve(GLPK_CMD())
    finalSeats = {(L[e[0]], P[e[1]]): mF['Elected'][e].varValue for e in mF['E']}
if f['final party apportionment method'] == 'PAV' :
    mF = ConcreteModel()
    # Range, Rangeset, Ballots, Elected, Party minimums, Scoreset, Scores, State seats, Votes
    mF.R = range(int(float(max([sum([B[b][0] for b in range(len(B)) if B[b][1] == i[0] and i[1] in B[b][2]]) for i in I])) / sum([B[b][0] for b in range(len(B))]) * (N + om + lm)) + 2)
    mF.Rset = Set(initialize = mF.R)
    mF.B = Var(range(len(B)), within = mF.Rset)
    mF.E = Var(I, within = mF.Rset)
    mF.partySeats = Param(P, mutable = True)
    for p in P :
        mF.partySeats[p] = int(round(value(mD.E[p])))
    mF.ScoreSet = [sum([1.0 / (1.0 + (v / K)) for v in range(n)]) for n in mF.R]
    mF.Scores = Var(range(len(B)), domain = NonNegativeReals)
    mF.stateSeats = Param(L, mutable = True)
    if 'census revision' in f :
        for s in L :
            mF.stateSeats[s] = int(round(value(mE.A[s])))
    else :
        for s in L :
            mF.stateSeats[s] = censusSeats[s]
    mF.V = Param(range(len(B)), mutable = True)
    for b in range(len(B)) :
        mF.V[b] = B[b][0]
    # constraints
    mF.Consts = ConstraintList()
    for b in range(len(B)) :
        mF.Consts.add(expr = mF.B[b] == sum(mF.E[i] for i in I if B[b][1] == i[0] and i[1] in B[b][2]))
    for i in I :
        mF.Consts.add(expr = mF.E[i] >= D[i[0]][i[1]])
    for p in P :
        mF.Consts.add(expr = sum(mF.E[i] for i in I if i[1] == p) == mF.partySeats[p])
    for s in L :
        mF.Consts.add(expr = sum(mF.E[i] for i in I if i[0] == s) == mF.stateSeats[s])
    mF.Consts.add(expr = sum(mF.E[i] for i in I) == (N + om + lm))
    mF.Piece = Piecewise(range(len(B)), mF.Scores, mF.B, pw_pts = mF.R, pw_repn = 'INC', pw_constr_type = 'EQ', f_rule = mF.ScoreSet)
    # objective
    mF.O = Objective(expr = summation(mF.V, mF.Scores), sense = maximize)
    # solve
    solver_mF = SolverFactory('ipopt')
    solver_mF.solve(mF)
    finalSeats = {i: int(round(value(mF.E[i]))) for i in I}
if f['final party apportionment method'] == 'SeqPhrag' :
    finalSeats = {i: [0, 0.0] for i in I}
    Loads = {b: 0.0 for b in range(len(B))}
    # constraints: constituency seats; party seat totals; total
    while sum([1 for i in I if finalSeats[i][0] < D[i[0]][i[1]]]) > 0 or sum([1 for p in P if sum([finalSeats[(s, p)][0] for s in L]) < partySeats[p]]) > 0 or sum([finalSeats[i][0] for i in I]) < (N + om + lm) :
        for i in I :
            finalSeats[i][1] = (1.0 + sum([(B[b][0] * Loads[b]) for b in Loads if B[b][1] == i[0] and i[1] in B[b][2]])) / sum([B[b][0] for b in Loads if B[b][1] == i[0] and i[1] in B[b][2]])
        A = min([finalSeats[i][1] for i in I if 'elimination' not in f or finalSeats[i][0] == 0])
        for i in I :
            if finalSeats[i][1] == A :
                finalSeats[i][0] += 1
                for b in Loads :
                    if p in B[b][2] :
                        Loads[b] == A
    finalSeats = {i: finalSeats[i][0] for i in I}
    if 'elimination' in f :
        P.sort()
if f['final party apportionment method'] == 'VarKS' :
    pass

# output solution
L.sort()
o = ['State\tCons. Seats\tList Seats\tTotal']
for s in L :
    o.append('\t'.join([s, str(sum([D[s][p] for p in P])), str(sum([finalSeats[(s, p)] - D[s][p] for p in P])), str(sum([finalSeats[(s, p)] for p in P]))]))
o.append('')
o.append('\t'.join(['', str(sum([sum([D[s][p] for p in P]) for s in L])), str((N + om + lm) - sum([sum([D[s][p] for p in P]) for s in L])), str(N + om + lm)]))
o.append('')
o.append('Party\tCons. Seats\tList Seats\tTotal')
for p in P :
    o.append('\t'.join([p, str(sum([D[s][p] for s in L])), str(sum([finalSeats[(s, p)] for s in L]) - sum([D[s][p] for s in L])), str(sum([finalSeats[(s, p)] for s in L]))]))
o.append('')
o.append('\t'.join(['', str(sum([sum([D[s][p] for p in P]) for s in L])), str((N + om + lm) - sum([sum([D[s][p] for p in P]) for s in L])), str(N + om + lm)]))
o.append('')
o.append('\t' + '\t\t'.join([p for p in P]))
o.append('\t' + '\t'.join(['\t'.join(['Cons. Seats', 'List Seats']) for p in P]) + '\t\tTotal')
o.append('')
for s in L :
    o.append('\t'.join([s, '\t'.join(['\t'.join([str(D[s][p]), str(finalSeats[(s, p)] - D[s][p])]) for p in P]), '', str(sum([finalSeats[(s, p)] for p in P]))]))
o.append('')
o.append('\t' + '\t'.join(['\t'.join([str(sum([D[s][p] for s in L])), str(sum([finalSeats[(s, p)] for s in L]) - sum([D[s][p] for s in L]))]) for p in P]) + '\t\t' + str(sum([finalSeats[i] for i in I])))
o.append('')
o.append('\t\t' + '\t\t'.join([str(sum([finalSeats[(s, p)] for s in L])) for p in P]))
o.append('')
o.append(''.join(['\t' for n in range(len(P) * 2)]) + str(sum([finalSeats[i] for i in I])))
f['election name'] += ' (' + f['apportionment method'] + ', ' + str(N)
if 'elimination' not in f and f['apportionment method'] != 'Scotland' :
    f['election name'] += '+'
f['election name'] += ').txt'
open('./results/' + f['election name'], 'w').write('\n'.join(o))

print 'Done  at', time.asctime()
