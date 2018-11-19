# coding: utf-8
import numpy as np
import json

# what runlog to use
jname = 'output/181117-deglared/runlog_181118_0610.json'

with open(jname) as f:
    runlog = json.load(f)

M = {k:np.round(v,decimals=3) for k,v in runlog['MCCS'].items()}
avgs = [m.mean() for m in M.values()]

# get divisions for the four sections
# good and bad will be half as big as the other two
p1 = np.percentile(avgs, 12.5)
p2 = np.percentile(avgs, 50.0)
p3 = np.percentile(avgs, 87.5)

print(f'thresholds calculated at {p1:.3%}, {p2:.3}, {p3:.3}')
qualities = dict()
poor = {k:m.mean() for k,m in M.items() if m.mean() <= p1}
fair = {k:m.mean() for k,m in M.items() if m.mean() > p1 and m.mean() <= p2}
okay = {k:m.mean() for k,m in M.items() if m.mean() > p2 and m.mean() <= p3}
good = {k:m.mean() for k,m in M.items() if m.mean() > p3}

qualities['good'] = good
qualities['okay'] = okay
qualities['fair'] = fair
qualities['poor'] = poor

print("""
      run:
    with open('sample-qualities.json', 'w') as f:
        json.dump(qualities, f, indent=True)

      ...to save this. won't do it for you!
      """
      )

