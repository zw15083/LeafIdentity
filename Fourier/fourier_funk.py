import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    im = cv2.imread(fileid,0)
    pixel_num = np.min(im.shape)
    pixel_ind = np.argmin(im.shape)
    if (pixel_ind == 0):
        im_norm = im[0:pixel_num][:, 0:pixel_num]
        return im_norm
    else:
        im_norm = im[0:pixel_num][0:pixel_num, :]
        return im_norm

def contour(im_norm):
    # Contour of leaf on plain background, thickness 10
    pixel_num = np.min(im_norm.shape)
    ret, thresh = cv2.threshold(im_norm, 127, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    leaf_contour = cv2.drawContours(np.zeros((pixel_num, pixel_num)), contours, 0, (255, 255, 255), 8)

    return leaf_contour

def fft_im(leaf_contour):
    imfft = np.fft.fft2(leaf_contour)
    fshift = np.fft.fftshift(imfft)
    mag_spec = 20 * np.log(np.abs(fshift))
    return mag_spec

def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

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

def feature_extract(mag_spec): #3D features
    mag_spec[mag_spec < 170] = 0 #filtering out lower values
# maybe mag_spec is changing ever time hence the shape is different?
    center_imx = int(mag_spec.shape[1]/2)
    center_imy = int(mag_spec.shape[0]/2)

    f1 = sector_mask(mag_spec.shape,(180,90),30,(0,360))
    f2 = sector_mask(mag_spec.shape,(150,350),30,(0,360))

    f11 = sector_mask(mag_spec.shape,(270,40),30,(0,360))
    f12 = sector_mask(mag_spec.shape,(450,450),30,(0,360))

    fs1 = sector_mask(mag_spec.shape,(center_imy,center_imx),130,(251,291))

    fs2 = sector_mask(mag_spec.shape,(center_imy,center_imx),130,(26,66))

    mask1 = f1 + f2 # circles
    mask2 = f11 + f12
    mask3 = fs1 # sectors ... two circles = area of sector
    mask4 = fs2

    selections = [mask1,mask2,mask3,mask4]

    kernel = []

    for index in range(len(selections)):
        features = np.copy(mag_spec)
        features[~(selections[index])] = 0
        value = np.log(np.sum(features**2))
        kernel.append(value)

    return kernel










