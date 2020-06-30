import SimpleITK as sitk
import numpy as np
import scipy
from scipy import ndimage as ndi

class Image3(object):
    def __init__(self, size = None, spacing = None, dtype = np.float32, origin = None):
        self.size = size
        self.spacing = spacing
        if origin != None:
            self.origin = origin
        else:
            self.origin = (0, 0, 0)

        if self.size != None:
            if self.spacing == None:
                self.spacing = (1, 1, 1)
            
            # Reverse size order to get a correct array structure
            self.data = np.zeros(self.size[::-1], dtype)

    def load_itk_image(self, file):
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        img = reader.Execute()

        self.size = img.GetSize()
        self.spacing = img.GetSpacing()
        self.origin = img.GetOrigin()

        self.data = sitk.GetArrayFromImage(img)

    def save_itk_image(self, file, dtype = None):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(file))

        if dtype != None:
            img = sitk.GetImageFromArray(self.data.astype(dtype))
        else:
            img = sitk.GetImageFromArray(self.data)

        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)

        writer.Execute(img)

    def load_rvf(self, file):
        with open(file, 'rb') as f:
            self.size = [int(i) for i in f.readline().split()]
            self.spacing = [float(i) for i in f.readline().split()]

            self.data = np.fromfile(f, dtype = [('x', 'float64'), ('y', 'float64'), ('z', 'float64')]).reshape(self.size[::-1])
            

    def save_rvf(self, file):
        with open(file, 'w') as f:
            f.write('%d %d %d\n' % (self.size[0], self.size[1], self.size[2]))
            f.write('%f %f %f\n' % (self.spacing[0], self.spacing[1], self.spacing[2]))
            self.data.tofile(f)


    def save_slice_images(self, file, r = None, axis = 2):
        if r == None:
            r = range(0, self.size[axis])
        assert(r[0] >= 0 and r[len(r)-1] < self.size[axis])

        if axis == 0:
            data = np.swapaxes(self.data, 2, 0)
        elif axis == 1:
            data = np.swapaxes(self.data, 0, 1)
        else:
            data = self.data

        for i in r:
            scipy.misc.imsave((file % i), data[i])

    def erode(self, structure, out = None):
        if out == None:
            out = Image3(self.size, self.spacing, dtype = np.bool)
        ndi.binary_erosion(self.data, structure, output=out.data)
        return out

    def greater_equal(self, x, out = None):
        if out == None:
            out = Image3(self.size, self.spacing, dtype = np.bool)
        np.greater_equal(self.data, x, out = out.data)
        return out
        
def create_itk_image(file):
    img = Image3()
    img.load_itk_image(file)
    return img

def create_rvf_image(file):
    img = Image3()
    img.load_rvf(file)
    return img




