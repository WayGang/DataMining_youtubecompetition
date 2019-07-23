import pandas as pd
import numpy as np
import sys
import csv
from copy import deepcopy


def read_file():
    """

    :param path:
    :return:
    """
    train = pd.read_csv('./bu-cs565-project-2-summer-2019/train_data.csv')
    columns = train.columns
    train_video_id = train['video_id']
    uploader = train['uploader']
    # print("uploader:",len(uploader))
    unique_uploader = uploader.drop_duplicates()
    # print(len(unique_uploader))
    category = train['category']
    unique_category = category.drop_duplicates()
    # print(len(unique_category))


    reco = [str(i) for i in range(1,21)]
    rec = train[reco]
    all_reco = deepcopy(rec.values)
    # print(rec.values)

    meta = pd.read_csv('./bu-cs565-project-2-summer-2019/videos_metadata.csv')
    meta_video_id = meta['video_id']

    count = 0
    test = pd.read_csv('./bu-cs565-project-2-summer-2019/test_data.csv')
    test_source = test['source']
    test_target = test['target']
    unique_target = test_target.drop_duplicates()
    # print(unique_target)
    not_in_any = []

    # for t in test_target:
    #     if t not in train_video_id.values:
    #         if t not in all_reco:
    #             if t not in meta_video_id.values:
    #                 count += 1
    #                 if count % 100 == 0:
    #                     print(count)
    # print(count)
    print(len(uploader))
    print(len(unique_uploader))


if __name__ == "__main__":

    read_file()
