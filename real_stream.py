import ksienie as ks
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score as metric
from imblearn.over_sampling import SMOTE

# Gather all the datafiles and filter them by tags
files = ks.dir2files("datasets/")
tag_filter = ["binary"]
datasets = []
for file in files:
    X, y, dbname, tags = ks.csv2Xy(file)
    intersecting_tags = ks.intersection(tags, tag_filter)
    if len(intersecting_tags):
        datasets.append((X, y, dbname))

np.random.seed(1410)

for d in [2,3,4,5,6,7,8,9]:

    X_s = []
    y_s = []
    dbnames = []
    n_s = []
    scores = []
    counts = []

    for i, dataset in enumerate(datasets):
        X, y, dbname = dataset
        if X.shape[1] >= d:
            if X.shape[0] < 200:
                continue

            best = -2.
            bidx = 0
            columns = None

            for z in range(100):
                selcons = np.random.choice(list(range(X.shape[1])), size=d, replace=False)

                n = len(y)
                X_ = X[:,selcons]

                clf = GaussianNB().fit(X_, y)
                y_pred = clf.predict(X_)
                score = metric(y, y_pred)

                if score > best:
                    columns = selcons
                    best = score
                    bidx = z

            if best > .75:
                scores.append(np.copy(best))

                X_ = X[:,columns]

                sm = SMOTE(random_state=42,
                           sampling_strategy={
                               0: 3000,
                               1: 3000
                           })
                X_, y = sm.fit_resample(X_, y)

                p = np.random.permutation(len(y))

                X_s.append(X_[p])
                y_s.append(y[p])
                dbnames.append(dbname)
                n_s.append(X.shape[1])
                counts.append(X_.shape[0])

    X = np.concatenate(X_s, axis=0)
    y = np.concatenate(y_s, axis=0)

    db = np.concatenate((X, y[:,np.newaxis]), axis=1)

    if len(y)/250 >= 100:
        print("# D=%i" % d, len(dbnames), db.shape, np.unique(y, return_counts=True), len(y)/250, '%.3f:%.3f:%.3f' % (np.min(scores), np.max(scores), np.mean(scores)))

        print(counts)
        concepts = np.rint(np.cumsum(counts)/250)
        print(concepts)

        np.save('streams/all_%i_drifts' % d, concepts)
        np.save('streams/all_%i' % d, db)
