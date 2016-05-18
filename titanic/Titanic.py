# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import print_function, division
import itertools
import operator

import pandas as pd
import numpy as np
import sklearn

from pylab import subplots

print("pandas  : {}".format(pd.__version__))
print("numpy   : {}".format(np.__version__))
print("sklearn : {}".format(sklearn.__version__))

# <codecell>

titanic = pd.read_csv("titanic/train.csv")
titanic.describe()

# <codecell>

alive = titanic[titanic["Survived"] == 1]
dead  = titanic[titanic["Survived"] == 0]

# <codecell>

colors = itertools.cycle(["red", "blue", "green"])

def boxplot_data(df, column, ax):
    ax.boxplot(df[column])
    x = np.random.normal(1, 0.08, size=len(df[column]))
    ax.plot(x, df[column], 'r.', alpha=0.2)
    ax.set_xlabel(column)

def histplot_data(df, column, ax, xmin=None, xmax=None, nbins=30):
    MIN, MAX = min(df[column]), max(df[column])
    if xmin is None:
        xmin = MIN
    if xmax is None:
        xmax = MAX
    width = (xmax - xmin)/nbins
    normal_ys, normal_bins = np.histogram(df[column], bins=nbins)
    normal_bin_centers = 0.5*(normal_bins[1:]+normal_bins[:-1])
    ax.bar(normal_bin_centers, normal_ys, width=width, color=next(colors))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(column)

#fig, ax = subplots(ncols=2, figsize=figsize(12, 5))
#boxplot_data(dead, "Fare", ax=ax[0])
#histplot_data(dead, "Fare", ax=ax[1])

# <codecell>

print(len(alive), len(alive[alive["Sex"]=="female"]))
print(len(dead), len(dead[dead["Sex"]=="female"]))

# <codecell>

def byFare(row):
    return row["Fare"]>10
def byGender(row):
    return row["Sex"]=="female"
def byFareAndGender(row):
    return row["Fare"]>10 and row["Sex"]=="female"

def getResultMatrix(func, df):
    def unfunc(row):
        return not func(row)
    withs = df[df.apply(func, axis=1)].Survived.value_counts().sort_index()
    wouts = df[df.apply(unfunc, axis=1)].Survived.value_counts().sort_index()
    return [[withs.iloc[1], withs.iloc[0]], [wouts.iloc[1], wouts.iloc[0]]]
    
def printResultMatrix(func, df, matrix=None):
    if matrix is None:
        matrix = getResultMatrix(func, df)
    withs, wouts = matrix
    print("\n" + func.func_name)
    print("       alive |  dead")
    print("       ----- | -----")
    print("with : {: >5} | {: >5} | {:.3}".format(withs[0], withs[1], withs[0]/sum(withs)))
    print("wout : {: >5} | {: >5} | {:.3}".format(wouts[0], wouts[1], wouts[0]/sum(wouts)))
    print("total: {: >5} | {: >5} | {:.3}".format(withs[0] + wouts[0], withs[1] + wouts[1],
                                                  (withs[0] + wouts[0]) / sum(withs + wouts)))
    print("right: {:.5}".format((withs[0] + wouts[1]) / sum(withs + wouts)))
    
printResultMatrix(byFareAndGender, titanic)

def byFareOrGender(row):
    return row["Sex"]=="female" or row["Fare"]>30
printResultMatrix(byFareOrGender, titanic)
printResultMatrix(byGender, titanic)

# <codecell>

def test_eval(func, df):
    survived = func(df) ## better return a pd.Series (np.array of booleans)
    df_results = df.copy(True) ## deep-copy
    df_results.loc[:,"Survived"] = survived.astype(int) ## coerce to 1/0, not True/False
    df_results.to_csv("results.csv", columns=["PassengerId", "Survived"], index=False)

testing = pd.read_csv("titanic/test.csv")
test_eval(lambda df:df.apply(byFareOrGender, axis=1), testing)

# <codecell>

def test_accuracy(func, df, matrix=None):
    if matrix is None:
        matrix = getResultMatrix(func, df)
    withs, wouts = matrix
    correct = (withs[0] + wouts[1]) / sum(withs + wouts)
    return correct

print("Gender: {}\nFare: {}\nFare and Gender{}\nFare or Gender{}"
      .format(test_accuracy(byGender, titanic),
      test_accuracy(byFare, titanic),
      test_accuracy(byFareAndGender, titanic),
      test_accuracy(byFareOrGender, titanic)))

# <markdowncell>

# #Function composition
# ---
# 
# Not the prettiest way to do it, but at some point, we will want to combine functions, perhaps in something like a grid-search. Therefore, it's helpful to have a way to generically combine functions programmatically rather than manually.

# <codecell>

def compose(oper, *funcs):
    def new_func(row):
        result = funcs[0](row)
        for func in funcs[1:]:
            result = oper(result, func(row))
        return result
    join_str = str(oper)
    if hasattr(oper, "__name__"):
        join_str = " {} ".format(oper.__name__)
    new_func.func_name = join_str.join(f.func_name for f in funcs)
    return new_func

# <codecell>

def youngerThan(row):
    return row["Age"] < 15

def olderThan(row):
    return row["Age"] > 60

for f in [lambda r: youngerThan(r) or olderThan(r),
          compose(operator.or_, youngerThan, olderThan)]:
    printResultMatrix(f, titanic)
fs = [youngerThan, olderThan, byGender]
printResultMatrix(compose(operator.or_, *fs), titanic)

# <markdowncell>

# The example above shows that the composition function works, not that it's particularly concise. The flexibility of it though will allow for more convoluted structures, and to compose compositions more easily, etc:

# <codecell>

def byPclass(row):
    return row["Pclass"] == 2

by_any = compose(operator.or_, byGender, byPclass, byFare)
gender_class = compose(operator.and_, byGender, byPclass)
gc_or_fare = compose(operator.or_, gender_class, byFare)

for f in (by_any, gender_class, gc_or_fare):
    printResultMatrix(f, titanic)

# <codecell>

pclasses = titanic["Pclass"].unique()
for pclass in pclasses:
    def f(row):
        return row["Pclass"] == pclass
    f.func_name = "Pclass == {}".format(pclass)
    printResultMatrix(f, titanic)

def var_search(variable, oper=operator.eq):
    funcs = []
    for val in titanic[variable].unique():
        if val is np.nan:
            continue
        def f(row):
            return oper(row[variable], val)
        oper_str = getattr(oper, "__name__", str(oper))
        f.func_name = "{} {} {}".format(variable, oper_str, val)
        funcs.append(f)
    return funcs

for f in var_search("Pclass"):
    printResultMatrix(f, titanic)

# <codecell>

def grid_search(var1, var2, op1=operator.eq, op2=operator.eq):
    best_accuracy, best_func = 0, "(none)"
    for f in var_search(var1, op1):
        for g in var_search(var2, op2):
            h_or = compose(operator.or_, f, g)
            h_or_acc = test_accuracy(h_or, titanic)
            if h_or_acc > best_accuracy:
                best_accuracy = h_or_acc
                best_func = h_or.__name__
            h_and = compose(operator.and_, f, g)
            h_and_acc = test_accuracy(h_and, titanic)
            if h_and_acc > best_accuracy:
                best_accuracy = h_and_acc
                best_func = h_and.__name__
    return best_func, best_accuracy

def grid_search2(*var_op_tuples):
    best_func, best_accuracy = "(none)", 0.0
    func_lists = [var_search(var, op) for var, op in var_op_tuples]
    for combos in itertools.product(*func_lists):
        h_or = compose(operator.or_, *combos)
        print(h_or.func_name)
        matrix=getResultMatrix(h_or, titanic)
        print(matrix)
        h_or_acc = test_accuracy(h_or, titanic)
        if h_or_acc > best_accuracy:
            best_accuracy = h_or_acc
            best_func = h_or.__name__
        h_and = compose(operator.and_, *combos)
        h_and_acc = test_accuracy(h_and, titanic)
        if h_and_acc > best_accuracy:
            best_accuracy = h_and_acc
            best_func = h_and.__name__
    return best_func, best_accuracy

print(grid_search("Pclass", "Sex"))
print(grid_search2(("Pclass", operator.eq), ("Sex", operator.eq)))

# <codecell>

titanic["Embarked"].unique()

for f in var_search("Embarked"):
    printResultMatrix(f, titanic)
    
grid_search("Embarked", "Sex")

# <codecell>

titanic

