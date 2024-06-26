from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class GamesDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'games'

    @classmethod
    def url(cls):
        # meta_Video_Games.json.gz from snap.stanford.edu does not contain full meta info
        return ['http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv',
                'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz']

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['games.csv', 'games_meta.json.gz']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        for idx, url in enumerate(self.url()):
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(url, tmpfile)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(tmpfile, folder_path.joinpath(self.all_raw_file_names()[idx]))
            print()

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
            smap[k]: v for k, v in meta_raw.items() 
            if k in smap
        }
        # New field to avoid type change breakdowns 
        enriched_meta = {
            smap[k]: v 
            for k, v in enriched_meta_raw.items() 
            if k in smap
        }

        import re
        def contains_html(text):
            return bool(re.search(r'<[^>]+>', text))

        def clean_html(diccionario):
            new_dict = {}
            for clave, valor in diccionario.items():
                if isinstance(valor, list):
                    new_dict[clave] = [item for item in valor if not contains_html(item)]
                else:
                    if not contains_html(valor):
                        new_dict[clave] = valor
            return new_dict

        # Clean all the entries of enriched_meta
        for key, value in enriched_meta.items():
            enriched_meta[key] = clean_html(value)

        # TODO Use same structure, a single string not a dictionary
        # select only the firts 3 categories or less
        new_dict = {
            key: f"{value['title']} ({', '.join(value['categories'][:3])})"
                for key, value in enriched_meta.items()
            }

        # print max length of the string
        max_length = max(len(valor) for valor in new_dict.values())
        print(f"The max length of the title is: {max_length}")

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
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])

        meta_dict = {}
        enriched_meta_dict = {}
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                item = eval(line)
                if 'title' in item and len(item['title']) > 0:
                    meta_dict[item['asin'].strip()] = item['title'].strip()
        
                    # New Dictionary with extended information, Categories is an array
                    # So it can be joined at any moment with required token
                    # categories = '|'.join(meta_dict['id']['categories'])
                    enriched_meta_dict[item['asin'].strip()] = {
                        'title': item['title'].strip(),
                        'categories': item['category']
                    }
                    
        return meta_dict, enriched_meta_dict
