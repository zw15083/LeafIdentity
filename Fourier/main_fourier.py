from fourier_funk import *
from mpl_toolkits.mplot3d import Axes3D


def main(names_list):

    path = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/leafsnap-dataset-images.txt'
    file_direct = '/Users/arthurdodson/Documents/MDM/Leaves/leafsnap-dataset/'

    feature_values = []
    kernel = []

    for num in range(len(names_list)):
        name = names_list[num]
        directory_list = leaf_list(path, name)

        for index in range(len(directory_list)):
            fileid = file_direct + directory_list[index]
            im_norm = read_image(fileid)
            #plt.imshow(im_norm)
            #plt.show()
            leaf_contour = contour(im_norm)
            #plt.imshow(leaf_contour)
            #plt.show()
            mag_spec = fft_im(leaf_contour)
            values = feature_extract(mag_spec)
            feature_values.append(values)

        kernel.append(feature_values)
        print(np.shape(kernel))
        print(np.shape(directory_list))

    return kernel




