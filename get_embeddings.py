import tensorflow.compat.v1 as tf
import numpy as np
import sys
import os
import copy
import argparse
import imageio
from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import tqdm
from tensorflow.python.platform import gfile
import nbimporter
import align.detect_face
from keras.models import load_model

def get_model_filenames(model_dir):
    print(os.getcwd())
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file



def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    #print(os.path.isfile(model_exp))
    if (os.path.isfile(model_exp)):
        #print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y              


#def load_and_align_data(image_paths, image_size):
#    print(image_paths)
#    tmp_image_paths=copy.copy(image_paths)
#    img_list = []

 #   for image in glob.glob(os.path.expanduser(image_paths)): #(tmp_image_paths)
  #      img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
   #     img_size = np.asarray(img.shape)[0:2]
    #    aligned = np.array(Image.fromarray(img).resize((image_size, image_size)))
     #   prewhitened = prewhiten(aligned)
      #  img_list.append(prewhitened)
    #images = np.stack(img_list)
    #return images
    
#Gives images of dimension 160 for inference, and of 60 for explanation. 
def load_and_align_data(image_paths, image_size, margin = 44, gpu_memory_fraction = 1.0):
    #print(image_paths)
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in glob.glob(os.path.expanduser(image_paths)): #(tmp_image_paths)
        img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = np.array(Image.fromarray(cropped).resize((image_size, image_size)))
        #aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    
    img_list_60 = []
    for img in img_list:
        img_list_60.append(skimage.transform.resize(img, (60,60)))
    
    
    images = np.stack(img_list)
    images_60 = np.stack(img_list_60)
    return images, images_60


def get_embeddings(save = False, df = None, path = None, image_list = None):
    #READ FROM PICKLE
    #If not, return array of images of shape (60,60,3), and list of its associated embeddings
    
    if df is not None:
        df = pd.read_pickle('celeb60_identities_pickle' )# '\src\lfw_pickle')
        images_60 = list(df["images"])#Erase last in brackets #Doesn't support 50000 images
        images = []
        for i in range(len(images_60)):
            images.append(skimage.transform.resize(images_60[i], (160,160)) )

    
    
    if path is not None: #READ FROM FOLDER
        images, images_60 = load_and_align_data(path, 160)
        #WORK IN PROGRESS
    
    if image_list is not None:
        images_60 = image_list
        images = []
        for i in range(len(images_60)):
            images.append(skimage.transform.resize(images_60[i], (160,160)) )

        
        
            
    images = np.stack(images)
    batch_size = 100
    n_images = len(images)
    #print(n_images)




    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            load_model("20180402-114759.pb")

            # Get input and output tensors
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            emb = np.ndarray((n_images, 512))*0
            i = 0

            while i*batch_size + batch_size <= n_images:
                feed_dict = {images_placeholder: images[i* batch_size : i* batch_size + batch_size], phase_train_placeholder:False }#, batch_size_placeholder:batch_size}
                emb[i* batch_size : i*batch_size + batch_size] = sess.run(embeddings, feed_dict=feed_dict)
                i += 1



            feed_dict = {images_placeholder: images[i* batch_size : n_images], phase_train_placeholder:False }#, batch_size_placeholder:batch_size}
            emb[i* batch_size : n_images] = sess.run(embeddings, feed_dict=feed_dict)
            #emb[i* batch_size : n_images] = 
            #print(type(emb))
            #print(emb.shape)
            #print(emb)
            
            if save:
                #emb_df = df[["person", 'imagenum']]
                #emb_df["embeddings"] = list(emb)
                df = pd.DataFrame({"embeddings": list(emb)})#ERASE FOR LFW
                df.to_pickle("pickle_name")
            else:
                return images_60, emb
