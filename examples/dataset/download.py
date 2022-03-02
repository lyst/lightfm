import os
import zipfile
import csv

import requests


def _download(url: str, dest_path: str):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def get_data():

    ratings_url = (
        "http://www2.informatik.uni-freiburg.de/" "~cziegler/BX/BX-CSV-Dump.zip"
    )

    if not os.path.exists("data"):
        os.makedirs("data")

        _download(ratings_url, "data/data.zip")

    with zipfile.ZipFile("data/data.zip") as archive:
        return (
            csv.DictReader(
                (
                    x.decode("utf-8", "ignore")
                    for x in archive.open("BX-Book-Ratings.csv")
                ),
                delimiter=";",
            ),
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Books.csv")),
                delimiter=";",
            ),
        )


def get_ratings():

    return get_data()[0]


def get_book_features():

    return get_data()[1]
