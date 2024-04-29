import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request


from pathlib import Path
import zipfile
import tarfile
import sys


def download(url, savepath):
    try:
        urllib.request.urlretrieve(url, str(savepath))
        print()
    except Exception:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, str(savepath))



def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()
