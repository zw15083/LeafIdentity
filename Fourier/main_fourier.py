from fourier_funk import *

def main(name):


    path = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/leafsnap-dataset-images.txt'
    file_direct = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/'
    directory_list = leaf_list(path,name)

    kernel = []
    for index in range(len(directory_list)):
        fileid = file_direct + directory_list[index]
        im_norm = read_image(fileid)
        leaf_contour = contour(im_norm)
        mag_spec = fft_im(leaf_contour)
        values = feature_extract(mag_spec)
        kernel.append(values)


    return kernel
