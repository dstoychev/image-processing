import skimage.morphology
import numpy

OUTPUT_DIR = "outputs"
INPUT_PATTERN = "inputs/*.tif"

CHANNEL_MCD8 = 0
CHANNEL_HRP = 1
CHANNEL_SHOT = 2

MORPHOLOGY_KERNEL = skimage.morphology.star(11)

# Region filter function should return False or True, depending on whether the
# region is to be kept or to be discarded, respectively.
def region_filter(region):
    if (
        0 not in region.bbox  # region doesn't touch any of the image sides
        and region.area > 1000
        and numpy.abs(numpy.degrees(region.orientation)) < 15
    ):
        return False
    else:
        return True


PSF_EX = 630
PSF_EM = 651
PSF_NA = 1.3
PSF_RI = 1.406
VOXEL_XY = 207
VOXEL_Z = 300