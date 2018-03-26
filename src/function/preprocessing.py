#Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

TARGET_DIR = '../../Data/HCP_MRI'
TARGET_FNAME = 'T1w_restore_brain.nii.gz'
# SAVE_DIR = '../../Data/MRI/HCP_MRI_256.npy'
SAVE_DIR = '/DATA_1/HCP_MRI_256.npy'
COUNT = 0
MIN = 0

def make_path_list(dir, filename):

    pathlist = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if fname == filename:
                path = os.path.join(root,fname)
                pathlist.append(path)
    pathlist = np.asarray(pathlist)
    return pathlist

def normalize(img):
    max = np.max(img)
    min = np.min(img)
    normalized_img = (img-min)/(max-min)

    return normalized_img

def rescale(vol,scale):
    # The MRI dataset shape is w > h = d, so make rescaled mri isotropic
    h,w,d = vol.shape
    vol_rs = zoom(vol,zoom=(scale,scale*float(h)/w,scale),mode='nearest')
    return vol_rs

def get_mri(data_path,):
    first_flag = True
    proxy_img = nib.load(data_path)
    data_array = np.asarray(proxy_img.dataobj).astype(np.float32)
    data_array = data_array[2:-2,:,2:-2]
    global COUNT
    for s in range(data_array.shape[1]):
        _slice = data_array[:,s,:]
        if (np.count_nonzero(_slice)==0):
            continue
        _slice = _slice.T[::-1,:]
        _slice = normalize(_slice)
        if first_flag:
            concat = _slice[...,None]
            first_flag = False
        else:
            concat = np.concatenate((concat,_slice[...,None]),axis=2)
        if concat.shape[2] == 257:
            print(COUNT + 1)
            COUNT += 1
            return concat
    return concat

def print_aff(data_path):
    proxy_img = nib.load(data_path)
    print(proxy_img.affine)

    return

def get_aff(dir = TARGET_DIR, fname= TARGET_FNAME):
    f_list = make_path_list(dir, fname)
    proxy_img = nib.load(f_list[0])
    return proxy_img.affine

def preprocessing(data_dir=TARGET_DIR,save_dir=SAVE_DIR,fname=TARGET_FNAME):

    print("Preprocessing Start")
    f_list = make_path_list(data_dir,fname)
    concat_mri = [get_mri(path) for path in f_list]
    concat_mri = np.asarray(concat_mri).astype(np.float32)
    np.save(save_dir,concat_mri)
    print("Concatenation Done")
    return

def get_min_nonzero_slice(data_dir=TARGET_DIR,fname=TARGET_FNAME):
    f_list = make_path_list(data_dir, fname)
    m_count = 10000
    i = 1
    for path in f_list:
        data = np.asarray(nib.load(path).dataobj)
        data = np.transpose(data,(0,2,1))
        data = data.reshape((-1,data.shape[-1]))
        max = np.max(data,axis=0)
        count = np.count_nonzero(max)
        m_count = min(m_count,count)
        print(i,":")
        print(m_count)

        i += 1
    print(m_count)
    return m_count

if __name__=="__main__":
    preprocessing()
    # get_min_nonzero_slice()
