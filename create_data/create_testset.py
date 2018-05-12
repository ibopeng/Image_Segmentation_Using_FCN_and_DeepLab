import tensorflow as tf
import numpy as np
import os
import scipy.io as spio
from matplotlib import pyplot as plt
#from matplotlib.pyplot import imread
from scipy.misc import imread

base_dataset_dir_voc = './test'
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


TEST_DATASET_DIR="./testset"


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
test_images_filename_list = get_files_list(TEST_DATASET_DIR + "/test.txt")
print("Total number of test images:", len(test_images_filename_list))


# shuffle array and separate 10% to validation
np.random.shuffle(test_images_filename_list)

# write_filelist(TRAIN_DATASET_DIR + '/train.txt', train_images_filename_list)
# write_filelist(TRAIN_DATASET_DIR + '/val.txt', val_images_filename_list)
# write_filelist(TRAIN_DATASET_DIR + '/test.txt', test_images_filename_list)

print("test set size:", len(test_images_filename_list))


Test_FILE = 'test.tfrecords'
test_writer = tf.python_io.TFRecordWriter(os.path.join(TEST_DATASET_DIR,Test_FILE))

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
    nerror = 0
    nerror_list = []
    for i, image_name in enumerate(filename_list):

        image_np = imread(os.path.join(images_dir_voc, image_name.strip() + ".jpg"))
        annotation_np = read_annotation_from_mat_file(os.path.join(annotations_dir_voc, (image_name.strip() + ".mat")))
        print(np.unique(annotation_np))
        #annotation_np = imread(os.path.join(annotations_dir_voc, image_name.strip() + ".png"))

        #if i==1:
        #    print(annotation_np.shape)
        #    print(annotation_np[100:120,400:410])
        #    return 0

        print(i)

        if annotation_np.shape != image_np.shape[:2]:
            nerror += 1
            nerror_list.append(image_name)
            print(annotation_np.shape)
            print(image_np.shape)

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
    print(nerror)
    print(nerror_list)

# create testing dataset
create_tfrecord_dataset(test_images_filename_list, test_writer)
print("test set size:", len(test_images_filename_list))