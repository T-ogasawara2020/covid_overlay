import pandas as pd
from glob import glob
import nibabel as nib 
import numpy as np
import matplotlib.pyplot as plt

def _get_df(base_path='dataset', folder=''):
    data_dict = pd.DataFrame({'FilePath': glob('{}/public-covid-data/{}/*'.format(base_path,folder)),
                         'FileName': [path.split('/')[-1] for path in glob('{}/public-covid-data/{}/*'.format(base_path,folder))]})    
    return data_dict
    

def get_df_all(base_path='dataset'):
    folder1_df = _get_df(base_path,folder='rp_im')
    folder2_df = _get_df(base_path,folder='rp_msk')
    df = folder1_df.merge(folder2_df, on='FileName', suffixes=('Image', 'Mask'))
    return df
    
    
def load_nifti(path):
    nifty = nib.load(path)
    data = nifty.get_fdata()
    data = np.rollaxis(data,axis=1)
    return data


def label_to_color(mask_volume, ggo_color=[255,0,0], consolidation_color=[0,255,0], effusion_color=[0,0,255]):
    shp = mask_volume.shape
    mask_color = np.zeros((shp[0],shp[1],shp[2],3),dtype=np.float32)
    mask_color[np.equal(mask_volume, 1)] = ggo_color
    mask_color[np.equal(mask_volume, 2)] = consolidation_color
    mask_color[np.equal(mask_volume, 3)] = effusion_color
    return mask_color


def hu_to_gray(volume):
    maxhu = np.max(volume)
    minhu = np.min(volume)
    volume_rerange = (volume-minhu) / max((maxhu-minhu), 1e-3)
    volume_rerange = volume_rerange * 255
    volume_rerange = np.stack([volume_rerange,volume_rerange,volume_rerange], axis=-1)
    
    return volume_rerange.astype(np.uint8)


def overlay(volume_gray,mask,mask_color,alpha=0.3):
    mask_filter = np.greater(mask,0)
    mask_filter = np.stack([mask_filter,mask_filter,mask_filter], axis=-1)
    overlayed = np.where(mask_filter,((1-alpha)*volume_gray+alpha*mask_color).astype(np.uint8), volume_gray)
    return overlayed   


def vis_overlay(overlayed,original_volume, mask_volume, cols=5, display_num=25,  figsize = (15, 15)):
    rows = (display_num - 1) // cols + 1
    total_num = overlayed.shape[-2]
    interval = total_num / display_num
    if interval < 1:
        interval = 1
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(display_num):
        idx = int(i * interval)
        row_i = i // cols
        col_i = i % cols
        if idx >= total_num:
            break
        stats = get_hu_stats(original_volume[:,:,idx], mask_volume[:,:,idx])
        title = 'slice #: {}'.format(idx)
        title += '\nggo mean: {:.0f}±{:.0f}'.format(stats['ggo_mean'], stats['ggo_std'])  
        title += '\nconsolidation mean: {:.0f}±{:.0f}'.format(stats['consolidation_mean'], stats['consolidation_std'])
        title += '\neffusion mean: {:.0f}±{:.0f}'.format(stats['effusion_mean'], stats['effusion_std'])
        ax[row_i, col_i].imshow(overlayed[:,:,idx])
        ax[row_i, col_i].set_title(title)
        ax[row_i, col_i].axis('off')        
    fig.tight_layout()
        
def get_hu_stats(volume, mask_volume,
                label_dict={1: 'ggo',
                           2: 'consolidation',
                           3: 'effusion'}):
    result = {}
    for key in label_dict.keys():
        label = label_dict[key]
        result[label+'_mean'] = volume[np.equal(mask_volume, key)].mean()
        result[label+'_std'] = volume[np.equal(mask_volume, key)].std()
    return result
    
    
    