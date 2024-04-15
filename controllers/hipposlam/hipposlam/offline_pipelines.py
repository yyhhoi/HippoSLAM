from .utils import read_pickle
from os.path import join
import pandas as pd
from torchvision.io import read_image
def concatenate_trajdata(load_trajdata_pth: str, save_trajdf_pth: str):
    """
    Concatenate a list of episode data and generate a dataframe.

    """

    keys = ['x', 'y', 'a', 't', 'fsigma', 'c']
    data_dict = {key: [] for key in keys}
    trajdata = read_pickle(load_trajdata_pth)  # list of dicts. The key in the dict contains a list of data.
    for i in range(len(trajdata)):
        episode = trajdata[i]
        for key in keys:
            data_dict[key].extend(episode[key])
    trajdf = pd.DataFrame(trajdata)

    # Add column: img_name c_t
    f = lambda x: str(x['c']) + '_' + str(x['t']) + '.png'
    trajdf['img_name'] = trajdf.apply(f, axis=1)
    trajdf['img_exist'] = trajdf['t'] % 5 == 0
    trajdf.save_pickle(save_trajdf_pth)
    return trajdf


def convert_images_to_mobilenet_embeddings(load_trajdf_pth, load_img_dir):

    trajdf = read_pickle(load_trajdf_pth)
    img_name_list = trajdf[trajdf['img_exist']]['img_name'].tolist()
    del trajdf

    for img_name in img_name_list:

        load_img_pth = join(load_img_dir, img_name)
        read_image(load_img_pth)

    pass