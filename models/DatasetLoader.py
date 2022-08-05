import os
import pandas as pd


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.index = 0
        self.contain_time = True
        self.fpath = os.path.join(data_dir, 'ratings.dat')
        self.df = None

    def load(self):
        # Load data
        self.df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return self.df

    def __iter__(self):
        self.index = 0
        if self.df is None:
            self.load()
        return self

    def __next__(self):
        if self.index < self.df.shape[0]:
            user, item, rating = self.df.iloc[self.index][['user', 'item', 'rate']]
            self.index += 1
            return user, item, rating
        else:
            raise StopIteration

class MovieLens20M(DatasetLoader):
    def __init__(self, data_dir):
        self.df = None
        self.index = 0
        self.contain_time = True
        self.fpath = os.path.join(data_dir, 'ratings.csv')

    def load(self):
        self.df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user', 'item', 'rate', 'time'],
                         usecols=['user', 'item', 'time'],
                         skiprows=1)
        return self.df

    def __iter__(self):
        self.index = 0
        if self.df is None:
            self.load()
        return self

    def __next__(self):
        if self.index < self.df.shape[0]:
            user, item, rating = self.df.iloc[self.index][['user', 'item', 'rate']]
            self.index += 1
            return user, item, rating
        else:
            raise StopIteration


class Scistarter(DatasetLoader):
    def __init__(self, data_dir):
        self.df = None
        self.index = 0
        self.contain_time = False
        self.fpath = os.path.join(data_dir, 'ratings.csv')

    def load(self):
        self.df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user','item','rate'],
                         skiprows=1)
        return self.df

    def __iter__(self):
        self.index = 0
        if self.df is None:
            self.load()
        return self

    def __next__(self):
        if self.index < self.df.shape[0]:
            user, item, rating = self.df.iloc[self.index][['user', 'item', 'rate']]
            self.index += 1
            return user, item, rating
        else:
            raise StopIteration


class Zooniverse(DatasetLoader):
    def __init__(self, data_dir, filename):
        self.df = None
        self.index = 0
        self.contain_time = True
        self.fpath = os.path.join(data_dir, filename)

    def load(self):
        self.df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user','item','time','rate'],
                         skiprows=1)
        return self.df

    def __iter__(self):
        self.index = 0
        if self.df is None:
            self.load()
        return self

    def __next__(self):
        if self.index < self.df.shape[0]:
            user, item, rating = self.df.iloc[self.index][['user', 'item', 'rate']]
            self.index += 1
            return user, item, rating
        else:
            raise StopIteration
