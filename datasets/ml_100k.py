from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ML100KDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):  # as of Sep 2023
        return 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README.txt',
                'movies.csv',
                'ratings.csv',
                'tags.csv']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print("Downloaded successfully")

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        # To honor legacy code I duplicate enriched_meta_raw with the required cateogires field
        # But keep as much as possible previous meta_raw architecture
        meta_raw, enriched_meta_raw = self.load_meta_dict()
        df = df[df['sid'].isin(meta_raw)]  # filter items without meta info
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {
            smap[k]: v 
            for k, v in meta_raw.items() 
            if k in smap
        }
        # New field to avoid type change breakdowns 
        enriched_meta = {
            smap[k]: v 
            for k, v in enriched_meta_raw.items() 
            if k in smap
        }

        # TODO Use same structure, a single string not a dictionary
        new_dict = {
            key: f"{value['title']} ({', '.join(value['categories'][:3])})"
                for key, value in enriched_meta.items()
            }

        # print max length of the string
        max_length = max(len(valor) for valor in new_dict.values())
        print(f"Max length of the string: {max_length}")

        dataset = {
            'train': train,
            'val': val,
            'test': test,
            #'meta': meta, # Changed for title (categories)
            'meta': new_dict, # Changed for title (categories)
            'umap': umap,
            'smap': smap
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.csv')
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        meta_dict = {}
        enriched_meta_dict = {}
        for row in df.itertuples():
            title = row[2][:-7]  # remove year (optional)
            year = row[2][-7:] # TODO: añadir también las categorías
            categories = row[3].split("|")

            title = re.sub('\(.*?\)', '', title).strip()
            # the rest articles and parentheses are not considered here
            if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_pre = title.split(', ')[:-1]
                title_post = title.split(', ')[-1]
                title_pre = ', '.join(title_pre)
                title = title_post + ' ' + title_pre

            meta_dict[row[1]] = title + year

            enriched_meta_dict[row[1]] = {
                "title": title,
                "year": year,
                "categories": categories
            }
            
        return meta_dict, enriched_meta_dict
