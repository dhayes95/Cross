"""Summarise the per-rank CSV written by TT_PROFILE_OUT.

    python3 profile_summary.py run_a.csv [run_b.csv ...]

Prints, for each phase, how the wall time splits into local compute, MPI
communication and idle wait, plus the spread across ranks.  Idle wait is the
load imbalance: a rank that finishes its share early sits in the Barrier ahead
of the next collective until the slowest rank catches up.  Comparing two runs
side by side is the way to tell whether a layout change actually helped.
"""
import sys
import csv


def load(path):
    with open(path) as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def pct(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def summarise(path):
    rows = load(path)
    n = len(rows)
    print('=' * 78)
    print('{}   ({} ranks)'.format(path, n))
    for phase, label in (('piv', 'PIVOT SEARCH'), ('core', 'CORE CONSTRUCTION')):
        total = [r[phase + '_total'] for r in rows]
        wall = max(total)
        if wall <= 0:
            continue
        print('  {}   wall = {:.4f}s'.format(label, wall))
        for part in ('compute', 'comm', 'wait'):
            v = [r[phase + '_' + part] for r in rows]
            avg = sum(v) / n
            print('    {:<8} avg={:8.4f}s ({:5.1f}% of wall)  min={:8.4f}  p50={:8.4f}'
                  '  p95={:8.4f}  max={:8.4f}  max/avg={:5.2f}'.format(
                      part, avg, 100 * avg / wall, min(v), pct(v, 0.50),
                      pct(v, 0.95), max(v), (max(v) / avg) if avg else 0.0))
        comp = sorted(rows, key=lambda r: r[phase + '_compute'])
        busiest = comp[-1]
        idlest = comp[0]
        print('    busiest rank {:>5}: compute {:.4f}s, wait {:.4f}s'.format(
              int(busiest['rank']), busiest[phase + '_compute'], busiest[phase + '_wait']))
        print('    idlest  rank {:>5}: compute {:.4f}s, wait {:.4f}s'.format(
              int(idlest['rank']), idlest[phase + '_compute'], idlest[phase + '_wait']))
        wasted = sum(r[phase + '_wait'] for r in rows)
        print('    aggregate idle = {:.4f} rank-seconds '
              '({:.1f}% of the {:.4f} rank-seconds the allocation was held)'.format(
                  wasted, 100 * wasted / (wall * n), wall * n))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for arg in sys.argv[1:]:
        summarise(arg)
