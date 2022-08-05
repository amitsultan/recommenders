import pandas as pd
from models.DatasetLoader import *
import numpy as np


class Dataset:

    def __init__(self, dataset_dir_path):
        self.dataset_dir = dataset_dir_path
        self.m_dUsers = {}  # Dict - key: user_id(str) value: User obj
        self.m_dItems = {}  # Dict - key: item_id(str) value: Item obj
        self.popularity = {}
        self.m_iiStatistics = None
        self.RatingsCount = 0
        self.head_items = {}
        self.tail_items = {}
        self.item_probability = {}  # pr_i
        self.item_item_co_probability = {}  # pr_i ^ pr_j
        self.data = None
        self.USERS = []  # array of User objects
        self.ITEMS = []  # array of Item objects

    def add_user_item_rating(self, UID, IID, rating):
        if UID not in self.m_dUsers:
            self.m_dUsers[UID] = User(UID)
        u = self.m_dUsers[UID]
        if IID not in self.m_dItems:
            self.m_dItems[IID] = Item(IID)
        i = self.m_dItems[IID]
        u.add_item_rating(IID, rating)
        i.add_user_rating(UID, rating)

    def calculate_popularity(self, head_tail_ratio=0.2):
        for user_id in self.m_dUsers.keys():
            for item_id in self.m_dUsers[user_id].items():
                if item_id in self.popularity:
                    self.popularity[item_id] += 1
                else:
                    self.popularity[item_id] = 1
        sorted_dict = {k: v for k, v in sorted(self.popularity.items(), key=lambda item: item[1])}
        index = 0
        for item_id in sorted_dict:
            if index < head_tail_ratio * len(sorted_dict):
                self.head_items[item_id] = sorted_dict[item_id]
            else:
                self.tail_items[item_id] = sorted_dict[item_id]
            index += 1
            total_contributions = sum(sorted_dict.values())
            self.item_probability[item_id] = sorted_dict[item_id] / total_contributions

    def read_dataset_from_csv(self, dataset_name, filename=None):
        if dataset_name == 'ml-1m':
            self.data = MovieLens1M(self.dataset_dir)
        elif dataset_name == 'ml-20m':
            self.data = MovieLens20M(self.dataset_dir)
        elif dataset_name == 'scistarter':
            self.data = Scistarter(self.dataset_dir)
        elif dataset_name == 'zooniverse':
            self.data = Zooniverse(self.dataset_dir, filename)
        else:
            raise NotImplementedError
        for user, item, rating in self.data:
            self.add_user_item_rating(user, item, rating)
        self.USERS = list(set(self.m_dUsers.values()))
        self.ITEMS = list(set(self.m_dItems.values()))

    def get_random_item(self):
        index = np.random.randint(0, len(list(self.m_dItems.keys())))
        return list(self.m_dItems.keys())[index]


class User:

    def __init__(self, UID):
        self.UID = UID
        self.item_ratings = {}

    def get_rating(self, IID):
        if IID in self.item_ratings:
            return self.item_ratings[IID]
        return 0

    def add_item_rating(self, IID, rating):
        if IID in self.item_ratings:
            raise Exception('Item already in user items')
        self.item_ratings[IID] = rating

    def items(self):
        return self.item_ratings.keys()

    def get_random_item(self):
        index = np.random.randint(0, len(list(self.item_ratings.keys())))
        return list(self.item_ratings.keys())[index]


class Item:

    def __init__(self, IID):
        self.IID = IID
        self.user_ratings = {}

    def get_rating(self, UID):
        if UID in self.user_ratings:
            return self.user_ratings[UID]
        return None

    def add_user_rating(self, UID, rating):
        if UID in self.user_ratings:
            raise Exception('Item already in user items')
        self.user_ratings[UID] = rating

    def users(self):
        return self.user_ratings.values()
