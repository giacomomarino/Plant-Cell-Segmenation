# https://stackoverflow.com/questions/29573126/image-boundary-finding-and-filling
from scipy import ndimage

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        #plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def partitioner(preds):
    # Read the image, convert the values to True or False;
    # discard all but the red channel (since it's a black and
    # white image, they're all the same)
    image = np.squeeze(np.where(np.array(preds) > 0.5, 1, 0))
    print(image.shape, image)

    # Compute connected regions in the image; we're going to use this
    # to find centroids for our watershed segmentation
    labels = skimage.measure.label(image)
    regions = skimage.measure.regionprops(labels)

    # Marker locations for the watershed segmentation; we choose these to
    # be the centroids of the different connected regions in the image
    markers = np.array([r.centroid for r in regions]).astype(np.uint16)
    marker_image = np.zeros_like(image, dtype=np.int64)
    marker_image[markers[:, 0], markers[:, 1]] = np.arange(len(markers)) + 1

    # Compute the distance map, which provides a "landscape" for our watershed
    # segmentation
    distance_map = ndimage.distance_transform_edt(1 - image)

    # Compute the watershed segmentation; it will over-segment the image
    filled = skimage.segmentation.watershed(1 - distance_map, markers=marker_image)

    # In the over-segmented image, combine touching regions
    filled_connected = skimage.measure.label(filled != 1, background=0) + 1

    # In this optional step, filter out all regions that are < 25% the size
    # of the mean region area found
    filled_regions = skimage.measure.regionprops(filled_connected)
    mean_area = np.mean([r.area for r in filled_regions])
    filled_filtered = filled_connected.copy()
    for r in filled_regions:
        if r.area < 0.25 * mean_area:
            coords = np.array(r.coords).astype(int)
            filled_filtered[coords[:, 0], coords[:, 1]] = 0

    print("FF", filled_filtered)

    # And display!
    #f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    #ax0.imshow(image, cmap='gray')
    #ax1.imshow(filled_filtered, cmap='Blues')
    #ax2.imshow(distance_map, cmap='gray')
    #plt.show()
    return filled_filtered


d = tf.random.uniform(shape=[100, 100])
# plt.imshow(d)
# plt.show()
# print(d)
# display([tf.expand_dims(d,axis=2)])
# partitioner(d)
