from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.engine.topology import Layer
import keras.backend as K

import scipy.io
import numpy as np
from glob import glob
import argparse
import json
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = 1024

def preprocess_image(x):

    # Substract Mean
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    # 'RGB'->'BGR'
    x = x[:, ::-1, :, :]

    return x

class RoiPooling(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            if self.dim_ordering == 'th':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], input_shape[1],
                                         y2 - y1, x2 - x1]
                            x_crop = img[:, :, y1:y2, x1:x2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3))
                            outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], y2 - y1,
                                         x2 - x1, input_shape[3]]
                            x_crop = img[:, y1:y2, x1:x2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output

def get_size_vgg_feat_map(input_W, input_H):
    output_W = input_W
    output_H = input_H
    for i in range(1,6):
        output_H = np.floor(output_H/2)
        output_W = np.floor(output_W/2)

    return output_W, output_H


def rmac_regions(W, H, L):

    ovr = 0.4 # desired overlap of neighboring regions
    steps = np.array([2, 3, 4, 5, 6, 7], dtype=np.float) # possible regions for the long dimension

    w = min(W,H)

    b = (max(H,W) - w)/(steps-1)
    idx = np.argmin(abs(((w ** 2 - w*b)/w ** 2)-ovr)) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []

    for l in range(1,L+1):

        wl = np.floor(2*w/(l+1))
        wl2 = np.floor(wl/2 - 1)

        b = (W - wl) / (l + Wd - 1)
        if np.isnan(b): # for the first level
            b = 0
        cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates

        b = (H-wl)/(l+Hd-1)
        if np.isnan(b): # for the first level
            b = 0
        cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                # R = np.array([i_, j_, wl, wl], dtype=np.int)
                R = np.array([j_, i_, wl, wl], dtype=np.int)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions


def generate_model(input_shape, num_rois, model_summary=False):

    # Load VGG16
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # The output size of VGG16 is 7x7x512
    
    # freeze the layers
    for layer in vgg16_model.layers[:-4]:
        layer.trainable = False

    print("Number of Region of Interests: ", num_rois)
    # Generate input layer for Region of Interests
    in_roi = Input(shape=(num_rois, 4), name='input_roi')
    
    # Use the RoiPooling layer which picks the regions and generate max over the regions
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-5].output, in_roi])
    model = Model([vgg16_model.input, in_roi], x)
    if model_summary:
        model.summary()

    return model


def generate_rep(img_path, img_size=224, model_summary=False):
    img = load_img(img_path, target_size=(img_size, img_size))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Mean substraction
    x = preprocess_image(x)

    Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1])
    regions = rmac_regions(Wmap, Hmap, 3)
    print('Loading Model...')
    model = generate_model((x.shape[1], x.shape[2], x.shape[3]),
                           len(regions), model_summary)

    # Compute vector
    print('Generating Vector...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    return np.sum(RMAC, axis=1)

def generate_rmac_vectors(image_dir, output_path):
    rmac_vec = {}

    img_files = glob(image_dir + "*")
    print("Number of images to generate for: ", len(img_files))
    with open(output_path, "w") as f:
        for idx, im in enumerate(img_files):
            imfile = im.split("/")[-1]
            print("Generating for im: {} file: {}".format(idx, imfile))
            rmac_vec[imfile] = generate_rep(im, 224).tolist()

        json.dump(rmac_vec, f)
        
    return rmac_vec

def get_similar_images(imgpath, image_dict, top_n=5):
    imfile = imgpath.split("/")[-1]
    # check if the image has pre-generated vector else generate it
    if imfile in image_dict.keys():
        img_vec = image_dict[imfile]
    else:
        img_vec = generate_rep(imgpath)

    similarities = {}
    for k, v in image_dict.items():
        similarities[k] = cosine_similarity(
            np.array(img_vec),
            np.array(v))

    # Return the top n similar images
    most_sim = dict(sorted(similarities.items(),
                key=lambda x:x[1],
                reverse=True)[:top_n]).keys()
    
    return most_sim

if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description="Feature Matching Image Retrieval"
        )
    parser.add_argument('--image_dir', dest="image_dir",
        default="", type=str)
    parser.add_argument('--trained_path', dest="trained",
        default="", type=str)
    parser.add_argument('--query_image', dest="query",
        default="", type=str)
    parser.add_argument('--mode', dest="mode",
        default="train", type=str)
    parser.add_argument('--top_n', dest="top_n",
        default=5, type=int)

    args =  vars(parser.parse_args())
    print(args)

    if args['mode'] == "retrieve":
        if not args["query"]:
            raise ValueError("Require query image..")
        elif not args['trained'] and not args['image_dir']:
            raise ValueError("Required either trained vectors or training data")

    if args['mode'] == "train" and not args["image_dir"]:
        raise ValueError("Require training data...")

    if args['mode'] == "retrieve" and args["trained"]:
        with open(os.path.join(args["trained"], "rmac_vectors.txt")) as f:
            rmac_vectors = json.loads(f.readlines()[0])
    else:
        t = int(time.time())
        dir_name = "deeplearning_{}".format(t)
        print("Saving vectors to directory: {}".format(dir_name))
        os.mkdir(dir_name)
        outfile = "{}/rmac_vectors.txt".format(dir_name)
        rmac_vectors = generate_rmac_vectors(args['image_dir'], outfile)


    if args['mode'] == "retrieve":
        sim_im = get_similar_images(args['query'], rmac_vectors, args['top_n'])
        print(sim_im)
