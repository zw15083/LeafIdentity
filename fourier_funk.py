import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def leaf_list(path, name):
    text_file = open(path, 'r')
   # file_direct = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/'
    text_file_list = text_file.readlines()
    the_line1 = next((s for s in text_file_list if name in s), None)  # returns 'str containing name
    index_number1 = text_file_list.index(the_line1)
    del text_file_list[0:index_number1]
    revearse_list = text_file_list[::-1]
    the_line2 = next((s for s in revearse_list if name in s), None)  # returns 'str containing name
    index_number2 = text_file_list.index(the_line2)
    del text_file_list[(index_number2+1):]
    # text_file_list = section of directories for a given tree, 'name'
    directory_list = []

    for index in range(len(text_file_list)):
        line = text_file_list[index]
        segmented_line1 = line.find('\t', 20)
        line = line[(segmented_line1+1):]
        segmented_line2 = line.find('\t', 10)
        line = line[0:segmented_line2] #store string in list
        directory_list.append(line)

    return directory_list

def read_image(fileid):
    im_norm = cv2.imread(fileid,0)
    """pixel_num = np.min(im.shape)
    pixel_ind = np.argmin(im.shape)
    if (pixel_ind == 0):
        im_norm = im[0:pixel_num][:, 0:pixel_num]
        return im_norm
    else:
        im_norm = im[0:pixel_num][0:pixel_num, :]"""
    return im_norm

def contour(im_norm):
    # Contour of leaf on plain background, thickness 10
    # ret, thresh = cv2.threshold(im_norm, 127, 255, 0) use when image is not in binary or use canny...
    image, contours, hierarchy = cv2.findContours(im_norm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_c = max(contours, key=cv2.contourArea) # draws largest contour
    # print('contoursbefore = ', np.shape(contours))
    i = 0
    while i < len(contours):
        contour_size = np.shape(contours[i])
        if contour_size[0] < 70:  # contours less than length 70 are ignored
            del contours[i]
        else:
            i += 1

    # leaf_contour = cv2.drawContours(np.zeros(im_norm.shape), [c], 0, (255, 255, 255), 8)
    leaf_contour = cv2.drawContours(np.zeros(im_norm.shape), contours, -1, (255, 255, 255), 8)

    # note, argument = -1 draws all contours

    return leaf_contour, max_c

def fft_im(leaf_contour):
    imfft = np.fft.fft2(leaf_contour)
    fshift = np.fft.fftshift(imfft)
    mag_spec = np.log(np.abs(fshift))
    phase_spec = np.angle(fshift, deg=True)

    return mag_spec, phase_spec

def sector_mask(shape,centre,radius,angle_range):
    """
    The start/stop angles in `angle_range` are in clockwise order.
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    mask = circmask * anglemask

    return mask


def bounding_box(leaf_contour): #make sure its only a single contour
    corner, w_h, angle = cv2.minAreaRect(leaf_contour)
    area_leaf = cv2.contourArea(leaf_contour)
    ratio = area_leaf/(w_h[0]*w_h[1])
    return ratio

def square_mask(shape,corner,length):
    """
    :param shape: shape of mask
    :param corner: top left
    :param length: length of side
    :return: mask
    """
    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = corner
    mask1 = x >= cx
    mask2 = y >= cy
    mask3 = x <= cx+length
    mask4 = y <= cx+length
    mask = mask1*mask2*mask3*mask4

    return mask


def feature_extract_sect(spec): #3D features
    spec[spec < 175] = 0  # filtering out lower values
    # maybe mag_spec is changing ever time hence the shape is different?

    imx = spec.shape[1]
    imy = spec.shape[0]

    f1 = sector_mask(spec.shape, ((imy * 1 / 3), (imx * 1 / 6)), 30, (0, 360))
    f2 = sector_mask(spec.shape, ((imy * 1 / 4), (imx * 3 / 4)), 30, (0, 360))

    f11 = sector_mask(spec.shape, (imy * 1 / 2, imx * 1 / 10), 30, (0, 360))
    f12 = sector_mask(spec.shape, (imy * 2 / 3, imx * 1 / 3), 30, (0, 360))

    center_imx = int(imx / 2)
    center_imy = int(imy / 2)
    fs1 = sector_mask(spec.shape, (center_imy, center_imx), 130, (251, 291))

    fs2 = sector_mask(spec.shape, (center_imy, center_imx), 130, (26, 66))

    mask1 = f1 + f2  # circles
    mask2 = f11 + f12
    mask3 = fs1  # sectors ... two circles = area of sector
    mask4 = fs2

    selections = [mask1, mask2, mask3, mask4]

    kernel = []

    for index in range(len(selections)):
        features = np.copy(spec)
        features[~(selections[index])] = 0
        value = (np.sum(features ** 2))
        if np.isinf(value):
            value = 0
        kernel.append(value)

    return features, kernel


def feature_extract_circle(spec, centers): #3D features
    #spec[spec < 170] = 0  # filtering out lower values
    # maybe mag_spec is changing ever time hence the shape is different?
    shape_m = spec.shape

    centers[0] *= shape_m[0]
    centers[1] *= shape_m[1]

    centers = np.round(centers + ((centers < 20)*20))
    centers[0] = centers[0]-(centers[0] > (shape_m[0] - 22))*20
    centers[1] = centers[1]-(centers[1] > (shape_m[1] - 22))*20

    #all_features = []
    kernel = []

    for index in range(centers.shape[1]):
        f1 = sector_mask(shape_m, centers[:, index], 30, (0, 360))
        features = np.copy(spec)
        features[~f1] = 0
        value = np.log(np.sum(features ** 2))
        if np.isinf(value):
            value = 0
        kernel.append(value)
      #  all_features.append(features)

    #all_features = sum(all_features)

    #plt.imshow(all_features)
    #plt.show()
    return kernel


def classify(kernel, folder_num):

    features = np.vstack(kernel)
    zero = np.zeros(folder_num[0])
    one = np.ones(folder_num[1])
    two = np.full(folder_num[2], 2)
    three = np.full(folder_num[3], 3)

    labels = np.concatenate((zero, one, two, three))
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

    gnb = GaussianNB()
    model = gnb.fit(train, train_labels)

    preds = model.predict(test)
    print(preds)
    print(accuracy_score(test_labels, preds))







