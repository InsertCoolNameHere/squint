from collections import  defaultdict

errors_accum = defaultdict(list)

errors_accum['l1x1'].append(11)
errors_accum['l1x1'].append(13)

print(errors_accum)
