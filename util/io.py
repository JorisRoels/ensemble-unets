
import os
import tifffile as tiff

# reads tif formatted file and returns the data in it as a numpy array
def read_tif(file, dtype='uint8'):

    data = tiff.imread(file).astype(dtype)

    return data

# write a 3D data set to a directory (slice by slice)
def imwrite3D(x, dir, prefix='', rescale=False):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(0,x.shape[0]):
        if rescale:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i,:,:] * 255).astype('uint8'))
        else:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i, :, :]).astype('uint8'))