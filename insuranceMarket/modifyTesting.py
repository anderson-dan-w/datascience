import pandas as pd
import collections

training = pd.read_csv("training.csv")
testing = pd.read_csv("testingCandidate.csv")

## pre-determined categorical columns
categorical = ["profession", "marital", "contact", "schooling", "month",
               "default", "day_of_week", "poutcome"]

## determine if the average return for marketing everyone in this dataframe
## is likely to yield a net gain (ie including the cost of marketing)
def yieldsPositive(df):
    counts = df.responded.value_counts()
    if "yes" not in counts:
        counts.yes = 0.0
    if "no" not in counts:
        counts.no = 0.0
    profit = df[df.responded == "yes"].profit.mean()
    net_gain = (counts.yes * profit) - (counts.sum() * 30.0)
    return net_gain > 0

## gather all the positive single indicators
## this is, in an abstract form, my model
positive_indicators = collections.defaultdict(set)
for category in categorical:
    for label, frame in training.groupby(category):
        if yieldsPositive(frame):
            positive_indicators[category].add(label)

## apply the model to the test set
mask = pd.Series([False] * len(testing))
mask.index = testing.index.copy()
for category, labels in positive_indicators.iteritems():
    for label in labels:
        mask |= (testing[category] == label)

## convert the true/false of the mask into 0/1 as required
mask = mask.astype(int)

## append these results to the dataframe, and write out to csv
## we could overwrite in place, but then we'll have may_respond column in
## our test data, and we may not want it to remain in there
testing["may_respond"] = mask
testing.to_csv("single-indicator-model")

