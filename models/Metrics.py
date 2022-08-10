import numpy as np

'''
metric_config:
{
'recall' : k:[1, 10, 20],
'precision: k:[50]
}
'''


def compute_metrics(alg, ds, metric_config, with_lift=False):
    user_ratings = get_scores(alg, ds)
    results_summary = {}
    if with_lift:
        user_ratings = lift_scores(user_ratings, ds)
    for key in metric_config:
        k_lst = metric_config[key]
        results = []
        if key == 'recall':
            for k in k_lst:
                score = recall(user_ratings, k)
                results.append(score)
        elif key == 'hitrate':
            for k in k_lst:
                score = hitrate(user_ratings, k, ds)
                results.append(score)
        elif key == 'head_hitrate':
                for k in k_lst:
                    score = hitrate(user_ratings, k, ds, 'head')
                    results.append(score)
        elif key == 'tail_hitrate':
                for k in k_lst:
                    score = hitrate(user_ratings, k, ds, 'tail')
                    results.append(score)
        elif key == 'precision':
            for k in k_lst:
                score = precision(user_ratings, k)
                results.append(score)
        results_summary[key] = {'k_lst': k_lst, 'values': results}
    return results_summary


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


def hitrate(user_ratings, k, ds, use_head_tail=None):
    hitrates = []
    for user in user_ratings.keys():
        predicted, true = user_ratings[user]['predicted'], user_ratings[user]['true']
        if len(true) == 0:
            continue
        if use_head_tail == 'head':
            head = list(ds.head_items.keys())
            true = np.intersect1d(head, true)
        elif use_head_tail == 'tail':
            tail = list(ds.tail_items.keys())
            true = np.intersect1d(tail, true)
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


def lift(ds, iid1, iid2, item_statistics):
    co_count = item_statistics.get_co_count(iid1, iid2)
    iid1_count = item_statistics.get_item_count(iid1)
    iid2_count = item_statistics.get_item_count(iid2)
    if iid1_count is None:
        print(f'item not found at item_count: {iid1}')
    elif iid2_count is None:
        print(f'item not found at item_count: {iid2}')
    else:
        user_count = len(ds.USERS)
        if co_count > 0:
            return (user_count * co_count) / (1.0 * iid1_count * iid2_count)
    return 1.0


def lift_scores(user_scores, ds):
    for uid in user_scores.keys():
        user = ds.m_dUsers[uid]
        user_items = user.items()
        predicted, true = user_scores[uid]['predicted'], user_scores[uid]['true']
        new_predicted = []
        for item_id, item_score in predicted:
            new_item_score = []
            for item in ds.ITEMS:
                if item.IID in user_items:
                    new_item_score.append(lift(ds, item_id, item.IID, ds.item_statistics))
            new_predicted.append((item_id, item_score * np.median(new_item_score)))
            new_predicted.sort(key=lambda x: -x[1])
        user_scores[uid]['predicted'] = new_predicted
    return user_scores
