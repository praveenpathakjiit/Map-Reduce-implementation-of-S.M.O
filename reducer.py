import sys

#Total = 0
mini= 10000000000.0
X = ""
for line in sys.stdin:
    data = line.strip().split()
    fun, x = data
    fun = float(fun)
    if mini > fun:
        mini = fun
        X = x
    
print (mini,X)
