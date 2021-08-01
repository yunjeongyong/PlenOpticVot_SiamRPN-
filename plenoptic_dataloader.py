import os
import glob
import numpy as np


class PlenopticDataLoader:
    def __init__(self, root, img2d_ref, focal_range=None):
        self.root = root
        self.img2d_ref = img2d_ref
        self.focal_range = focal_range

    def dataLoader_focal(self):
        self.img2d_files = []
        self.focal_files = []
        frames = os.listdir(self.root)
        for frame in frames:
            img2d_file = os.path.join(os.path.join(
                self.root, frame), self.img2d_ref)
            focal_path = os.path.join(os.path.join(self.root, frame), 'focal')
            focal_planes = glob.glob(glob.escape(focal_path) + "/*")
            if self.focal_range is None:
                pass
            else:
                focal_planes = focal_planes[self.focal_range[0]
                    :self.focal_range[1] + 1]
            self.img2d_files.append(img2d_file)
            self.focal_files.append(focal_planes)
        return self.img2d_files, self.focal_files

    def dataLoader_2d(self):
        self.img2d_files = []
        frames = os.listdir(self.root)
        for frame in frames:
            img2d_file = os.path.join(os.path.join(
                self.root, frame), self.img2d_ref)
            self.img2d_files.append(img2d_file)
        return self.img2d_files


if __name__ == "__main__":
    dataLoader_focal = PlenopticDataLoader(
        root='E:/NonVideo4', img2d_ref='images/005.png', focal_range=(20, 50))
    img2d_files, focal_files = dataLoader_focal.dataLoader_focal()
    print(np.asarray(img2d_files).shape)
    print(np.asarray(focal_files))
    print("end test")
