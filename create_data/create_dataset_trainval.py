import tensorflow as tf
import numpy as np
import os
import scipy.io as spio
from matplotlib import pyplot as plt
#from matplotlib.pyplot import imread
from scipy.misc import imread

base_dataset_dir_voc = './VOC2011'
images_folder_name_voc = "JPEGImages/"
annotations_folder_name_voc = "SegmentationClass_1D/"
images_dir_voc = os.path.join(base_dataset_dir_voc, images_folder_name_voc)
annotations_dir_voc = os.path.join(base_dataset_dir_voc, annotations_folder_name_voc)

# define base paths for pascal augmented VOC images
# download: http://home.bharathh.info/pubs/codes/SBD/download.html
# base_dataset_dir_aug_voc = './benchmark_RELEASE/dataset'
# images_folder_name_aug_voc = "img"
# annotations_folder_name_aug_voc = "cls"
# images_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, images_folder_name_aug_voc)
# annotations_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, annotations_folder_name_aug_voc)
# fplist_images_dir_aug_voc = os.listdir(images_dir_aug_voc)
# fplist_annotations_dir_aug_voc = os.listdir(annotations_dir_aug_voc)


TRAIN_DATASET_DIR="./dataset_prj_VOC2011official_100"
sub_size = 1


def get_files_list(filename):
    file = open(filename, 'r')
    images_filename_list = [line for line in file]
    return images_filename_list

def write_filelist(filename, filelist):
    outF = open(filename, "w")
    for line in filelist:
        outF.write(line)
    outF.close()


#images_filename_list = get_files_list(base_dataset_dir_aug_voc, images_folder_name_aug_voc, annotations_folder_name_aug_voc, "custom_train.txt")
train_images_filename_list = get_files_list(TRAIN_DATASET_DIR + "/train.txt")
print("Total number of training images:", len(train_images_filename_list))

val_images_filename_list = get_files_list(TRAIN_DATASET_DIR + "/val.txt")
print("Total number of val images:", len(val_images_filename_list))


# shuffle array and separate 10% to validation
np.random.shuffle(train_images_filename_list)
sub_train_images_filename_list = train_images_filename_list[:int(sub_size*len(train_images_filename_list))]

# write_filelist(TRAIN_DATASET_DIR + '/train.txt', train_images_filename_list)
# write_filelist(TRAIN_DATASET_DIR + '/val.txt', val_images_filename_list)
# write_filelist(TRAIN_DATASET_DIR + '/test.txt', test_images_filename_list)

print("train set size:", len(sub_train_images_filename_list))
print("val set size:", len(val_images_filename_list))


TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
train_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE))
val_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_annotation_from_mat_file(annotations_path):
    mat = spio.loadmat(annotations_path)
    img = mat['im']
    return img


def create_tfrecord_dataset(filename_list, writer):

    # create training tfrecord
    for i, image_name in enumerate(filename_list):

        image_np = imread(os.path.join(images_dir_voc, image_name.strip() + ".jpg"))
        annotation_np = read_annotation_from_mat_file(os.path.join(annotations_dir_voc, (image_name.strip() + ".mat")))
        #annotation_np = imread(os.path.join(annotations_dir_voc, image_name.strip() + ".png"))

        #if i==1:
        #    print(annotation_np.shape)
        #    print(annotation_np[100:120,400:410])
        #    return 0

        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()
        annotation_raw = annotation_np.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_h),
                'width': _int64_feature(image_w),
                'image_raw': _bytes_feature(img_raw),
                'annotation_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())
    
        print("End of TfRecord. Total of image written:", i)

    writer.close()


# create training dataset
create_tfrecord_dataset(sub_train_images_filename_list, train_writer)

# create validation dataset
create_tfrecord_dataset(val_images_filename_list, val_writer)

# create testing dataset
#create_tfrecord_dataset(test_images_filename_list, test_writer)