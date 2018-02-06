import numpy as np
import cv2

file_direct = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/dataset/segmented/lab/'
leaf = 'acer_pseudoplatanus/wb1561-07-4.png'
fileid = file_direct+leaf

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

def feature_extract(mag_spec):
    mag_spec[mag_spec < 170] = 0 #filtering out lower values

    f1 = sector_mask(mag_spec.shape,(150,300),30,(0,360))
    f2 = sector_mask(mag_spec.shape,(200,400),30,(0,360))
    f3 = sector_mask(mag_spec.shape,(300,440),30,(0,360))
    f4 = sector_mask(mag_spec.shape,(400,400),30,(0,360))
    f5 = sector_mask(mag_spec.shape,(430,300),30,(0,360))
    f6 = sector_mask(mag_spec.shape,(400,200),30,(0,360))
    f7 = sector_mask(mag_spec.shape,(300,150),30,(0,360))
    f8 = sector_mask(mag_spec.shape,(180,200),30,(0,360))
    mask = f1#+f2+f3+f4+f5+f6+f7+f8)

    features = mag_spec[~mask]
    value = np.log(np.sum(features**2))
    return value








