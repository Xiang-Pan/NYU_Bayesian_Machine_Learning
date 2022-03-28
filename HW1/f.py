D = [(4,21),(9,59),(7,25),(15,127)]
print(D)

a = 0
for d in D:
    x,y = d
    a += y/x
a /= len(D)

print(a)

v = 0.0
for d in D:
    x,y = d
    t = (y - a * x)** 2
    v += t
v /= len(D)
print(v)

S = 0.0
for d in D:
    x,y = d
    t = (x)** 2
    S += t
print(S)