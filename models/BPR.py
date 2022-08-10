from models.Dataset import *
import numpy as np


class BPR:

    learningDecay = 0.99
    itemBiasStep = 0.5
    itemBiasReg = 0.1
    userBiasStep = 0.7
    userBiasReg = 0.1
    itemsTraitsStep = 0.2
    itemsTraitsReg = 0.01
    usersTraitsStep = 0.2
    usersTraitsReg = 0.01

    def __init__(self, dimensionality, ds, verbose=False):
        self.userBiases = {}
        self.userTraits = {}
        self.itemBiases = {}
        self.itemTraits = {}
        self.nUsers = len(ds.USERS)
        self.nItems = len(ds.ITEMS)
        self.dim = dimensionality
        self.mue = 0.0
        self.dataset = ds

        cRatings = 0
        for user in ds.USERS:
            for item_id, item_rating in user.item_ratings.items():
                self.mue += item_rating
                cRatings += 1
        self.mue /= cRatings
        # Init user biases and traits (can be done in the loop above but separated for conv)
        for user in ds.USERS:
            assert user.UID not in self.userBiases, 'User already has bias'
            assert user.UID not in self.userTraits, 'User already has traits'
            self.userBiases[user.UID] = 0
            self.userTraits[user.UID] = np.random.uniform(0, 1, self.dim) * 0.1
        # Init item biases and traits (can be done in the loop above but separated for conv)
        for item in ds.ITEMS:
            assert item.IID not in self.itemBiases, 'Item already has bias'
            assert item.IID not in self.itemTraits, 'Item already has traits'
            self.itemBiases[item.IID] = 0
            self.itemTraits[item.IID] = np.random.uniform(0, 1, self.dim) * 0.1

    def update(self, user, i_first, i_second, d_diff):
        d_exp = np.exp(-d_diff)
        d_coeff = d_exp / (1 + d_exp)
        # update traits
        v_first = self.itemTraits[i_first]
        v_second = self.itemTraits[i_second]
        v_user = self.userTraits[user]
        d_trait_diff = v_first - v_second
        self.itemTraits[i_first] = v_first + self.itemsTraitsStep * (d_coeff * v_user - self.itemsTraitsReg * v_first)
        self.itemTraits[i_second] = v_second + self.itemsTraitsStep * (d_coeff * (-v_user) - self.itemsTraitsReg * v_second)
        self.userTraits[user] = v_user + self.usersTraitsStep * (d_coeff * d_trait_diff - self.usersTraitsReg * v_user)
        # for i in range(self.dim):
        #     d_trait_diff = v_first[i] - v_second[i]
        #     v_first[i] = v_first[i] + self.itemsTraitsStep * (d_coeff * v_user[i] - self.itemsTraitsReg * v_first[i])
        #     v_second[i] = v_second[i] + self.itemsTraitsStep * (d_coeff * (-v_user[i]) - self.itemsTraitsReg * v_second[i])
        #     v_user[i] = v_user[i] + self.usersTraitsStep * (d_coeff * d_trait_diff - self.usersTraitsReg * v_user[i])

    def predict(self, user, item, turncate=False):
        res = self.mue
        p = 0
        if item in self.itemTraits and user in self.userTraits:
            item_vec = self.itemTraits[item]
            user_vec = self.userTraits[user]
            p = item_vec * user_vec
        res += np.sum(p)
        return res

    def recommend(self, user, k=10):
        if user not in self.dataset.m_dUsers:
            raise Exception('User not found at model dataset!')
        user_obj = self.dataset.m_dUsers[user]
        ratings = {}
        for item in self.dataset.ITEMS:
            if item.IID in user_obj.item_ratings:
                continue
            ratings[item.IID] = self.predict(user, item.IID)
        ratings = {k: v for k, v in sorted(ratings.items(), key=lambda item: -item[1])}
        ratings = dict(list(ratings.items())[0: k])
        return [(k, v) for k, v in ratings.items()]

    def SGD(self, train_ds, epochs):
        users = train_ds.USERS

        for epoch in range(epochs):
            training_lines, correct_train, incorrect_train = 0, 0, 0
            c_users = len(users)
            c_updates = c_users * 10

            c_failed_sample = 0

            i_user = 0

            for i_update in range(c_updates):
                i_user = (i_user + 1) % c_users
                user = users[i_user]
                items = user.item_ratings.keys()
                i_first, i_second = None, None
                dr1 = -1
                dr2 = -1
                if len(items) > 1:
                    if epoch < epochs / 2:
                        i_first = user.get_random_item()
                        i_second = i_first
                        while i_second in items:
                            i_second = train_ds.get_random_item()
                    else:
                        d_pr = np.random.uniform(0, 1)
                        if d_pr <= 1:
                            i_first = user.get_random_item()
                            i_second = i_first
                            while i_second in items:
                                i_second = train_ds.get_random_item()
                        dr1 = user.get_rating(i_first)
                        dr2 = user.get_rating(i_second)
                        if dr2 > dr1:
                            tmp = i_first
                            i_first = i_second
                            i_second = tmp
                            dr1 = user.get_rating(i_first)
                            dr2 = user.get_rating(i_second)

                    training_lines += 1
                    d_first_score = self.predict(user.UID, i_first)
                    d_second_score = self.predict(user.UID, i_second)
                    d_diff = d_first_score - d_second_score
                    if d_diff > 0:
                        correct_train += 1
                    else:
                        incorrect_train += 1
                    self.update(user.UID, i_first, i_second, d_diff)
                    if i_update % 10000 == 0:
                        print(f'{i_update}/{c_updates}: ')
            print(f'Finished round: {epoch}, Correct train: {correct_train}, Incorrect train: {incorrect_train}, Failed samples: {c_failed_sample}')
            # Exponential decay:
            self.itemBiasStep *= self.learningDecay
            self.userBiasStep *= self.learningDecay
            self.itemsTraitsStep *= self.learningDecay
            self.usersTraitsStep *= self.learningDecay

