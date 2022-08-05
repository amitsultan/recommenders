import numpy as np

'''
metric_config:
{
'recall' : k:[1, 10, 20],
'precision: k:[50]
}
'''


def compute_metrics(alg, ds, metric_config):
    user_ratings = get_scores(alg, ds)
    for key in metric_config:
        k_lst = metric_config[key]
        if key == 'recall':
            for k in k_lst:
                score = recall(user_ratings, k)
                print(f'metric ${key}$  k ${k}$ score ${score}$')
        elif key == 'hitrate':
            for k in k_lst:
                score = hitrate(user_ratings, k)
                print(f'metric ${key}$  k ${k}$ score ${score}$')
        elif key == 'precision':
            for k in k_lst:
                score = precision(user_ratings, k)
                print(f'metric ${key}$  k ${k}$ score ${score}$')


def get_scores(alg, ds):
    user_ratings = {}
    for user in ds.USERS:
        ratings = {}
        true_items = user.items()
        for item in alg.dataset.ITEMS:
            if item.IID in alg.dataset.m_dUsers[user.UID].item_ratings:  # ignore items from training
                continue
            ratings[item.IID] = alg.predict(user.UID, item.IID)
        ratings = {k: v for k, v in sorted(ratings.items(), key=lambda item: -item[1])}
        user_ratings[user.UID] = {'predicted': [(k, v) for k, v in ratings.items()], 'true': list(true_items)}
    return user_ratings


def recall(user_ratings, k):
    recalls = []
    for user in user_ratings.keys():
        predicted, true = user_ratings[user]['predicted'], user_ratings[user]['true']
        if len(true) == 0:
            continue
        predicted = np.array(predicted[:k])
        predicted = predicted[:, 0]  # only the item ids
        hits = np.intersect1d(predicted, true)
        recalls.append(len(hits) / len(true))
    return np.mean(recalls)


def hitrate(user_ratings, k):
    hitrates = []
    for user in user_ratings.keys():
        predicted, true = user_ratings[user]['predicted'], user_ratings[user]['true']
        if len(true) == 0:
            continue
        predicted = np.array(predicted[:k])
        predicted = predicted[:, 0]  # only the item ids
        hits = np.intersect1d(predicted, true)
        hitrates.append(1 if len(hits) > 0 else 0)
    return np.mean(hitrates)


def precision(user_ratings, k):
    precisions = []
    for user in user_ratings.keys():
        predicted, true = user_ratings[user]['predicted'], user_ratings[user]['true']
        if len(true) == 0:
            continue
        predicted = np.array(predicted[:k])
        predicted = predicted[:, 0]  # only the item ids
        hits = np.intersect1d(predicted, true)
        precisions.append(len(hits) / k)
    return np.mean(precisions)



def lift(item_1, item_2, item_statistics):
    return NotImplementedError

def lift_scores(user_scores):
    return NotImplementedError

