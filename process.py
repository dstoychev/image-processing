import pathlib
import glob
import tifffile
import numpy as np
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.measure
import skimage.feature
import bigfish.stack
import bigfish.detection
import csv

import process_config


def get_nmj_mask(image_mcd8, image_hrp, output_path):
    # Step 1: Thresholding
    # NOTE: No filtering is done before the thresholding. This is because the
    # images are already fairly clean and thus filtering will just destroy the
    # detail needlessly. With different datasets it may be a necessary step
    # though.
    threshold_mcd8 = skimage.filters.threshold_triangle(image_mcd8)
    threshold_hrp = skimage.filters.threshold_triangle(image_hrp)
    image_mcd8_thresholded = image_mcd8 > threshold_mcd8
    skimage.io.imsave(
        output_path.with_name(
            output_path.stem + "_p-01-mcd8-thresholded" + ".png"
        ),
        skimage.img_as_uint(image_mcd8_thresholded),
    )
    image_hrp_thresholded = image_hrp > threshold_hrp
    skimage.io.imsave(
        output_path.with_name(
            output_path.stem + "_p-01-hrp-thresholded" + ".png"
        ),
        skimage.img_as_uint(image_hrp_thresholded),
    )
    # Step 2: Subtraction (a bit redundant...)
    image_subtracted = image_mcd8_thresholded & image_hrp_thresholded
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_p-02-and" + ".png"),
        skimage.img_as_uint(image_subtracted),
    )
    # Step 3: Morphological processing
    # NOTE: The aim of this step is to close any gaps in the NMJ. The most
    # efficient structuring element for this task seems to be the star. It
    # dilates as little as possible while still able to establish a connection
    # between disjoin segments. The size needs to be that of the largest gap
    # and ideally not any larger, as this has the potential of connecting
    # unwanted regions. The second best structuring element is the disk, but it
    # requires a larger size than the star, which makes it less efficient and
    # more expansive.
    image_morphology = skimage.morphology.closing(
        image_subtracted, process_config.MORPHOLOGY_KERNEL
    )
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_p-03-morphology" + ".png"),
        skimage.img_as_uint(image_morphology),
    )
    # Step 4: Label and filter by various criteria
    image_labelled = skimage.measure.label(image_morphology)
    for region in skimage.measure.regionprops(image_labelled):
        if process_config.region_filter(region):
            image_labelled[image_labelled == region.label] = 0
    # Step 5: Create a mask and return it
    mask = image_labelled != 0
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_p-04-filtered" + ".png"),
        skimage.img_as_uint(mask),
    )
    return mask


def calculate_psf(w_ex, w_em, na, ri):
    psf_yx = 0.225 / na * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    psf_z = 0.78 * ri / na ** 2 * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    return (psf_z, psf_yx)


def fish_processing(zstack, output_path):
    # Step 0: Calculate PSF and sigma
    psf_z, psf_yx = calculate_psf(
        process_config.PSF_EX,
        process_config.PSF_EM,
        process_config.PSF_NA,
        process_config.PSF_RI,
    )
    psf_sigma = bigfish.detection.get_sigma(
        process_config.VOXEL_Z, process_config.VOXEL_XY, psf_z, psf_yx
    )
    # Step 2: Maximum Z projection
    stack_projection = bigfish.stack.maximum_projection(zstack)
    # Step 3: Do Laplacian of Gaussians filtering
    stack_filtered = bigfish.stack.log_filter(zstack, psf_sigma)
    # Step 4: Local maximum detection
    locmax = bigfish.detection.local_maximum_detection(
        stack_filtered, min_distance=psf_sigma
    )
    # Step 5: Spot detection
    threshold = bigfish.detection.automated_threshold_setting(
        stack_filtered, locmax
    )
    spots, _ = bigfish.detection.spots_thresholding(
        stack_filtered, locmax, threshold
    )
    # Step 6: Draw the contrasted Z projection and all the spots
    stack_projection_contrasted = bigfish.stack.rescale(
        stack_projection, channel_to_stretch=0
    )
    output_image = skimage.color.gray2rgb(
        skimage.img_as_ubyte(stack_projection_contrasted)
    )
    for spot in spots:
        rr, cc = skimage.draw.circle_perimeter(
            spot[1], spot[2], 3, shape=output_image.shape
        )
        output_image[rr, cc] = [255, 0, 0]
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_p-05-fish-spots" + ".png"),
        output_image,
    )
    # Return the spots; it's numpy array of shape (N, 3), where the three
    # coordinates are <Z, Y, X> or equivalently <Z, row, col>.
    return spots


def main():
    # Step 1: Ensure an output directory exists
    pathlib.Path(process_config.OUTPUT_DIR).mkdir(exist_ok=True)
    # Step 2: Iterate the images in the input directory, according to the
    # specified pattern
    image_paths = glob.glob(process_config.INPUT_PATTERN)
    print(f"Found {len(image_paths)} input images.")
    if len(image_paths) == 0:
        # No images detected => exit
        return
    count_data = []
    for image_index, image_path in enumerate(image_paths):
        print(
            f"Processing image {image_index + 1} out of {len(image_paths)}..."
        )
        # Step 2.1: Determine the name of the output image
        output_image_path = pathlib.Path(
            image_path.replace("inputs", "outputs")
        )
        # Step 2.2: Read the image
        image = tifffile.imread(image_path)
        # Step 2.3: Z projection (mean intensity) of each channels
        zstacks = np.mean(image, axis=0).astype(np.uint16)
        # Step 2.4: Get cell masks
        mask = get_nmj_mask(
            zstacks[process_config.CHANNEL_MCD8],
            zstacks[process_config.CHANNEL_HRP],
            output_image_path,
        )
        # Step 2.5: FISH processing
        spots = fish_processing(
            image[:, process_config.CHANNEL_SHOT, :, :], output_image_path
        )
        # Step 2.6: Count spots in the mask
        spot_count = np.count_nonzero(mask[spots[:, 1], spots[:, 2]])
        print(f"Found {spot_count} spots.")
        count_data.append(
            {
                "image": output_image_path.stem,
                "count": spot_count,
            }
        )
    # Step 3: Combine all the CSV files
    with open(
        pathlib.Path(process_config.OUTPUT_DIR).joinpath("metrics.csv"),
        "w",
        newline="",
    ) as fout:
        writer = csv.DictWriter(fout, fieldnames=list(count_data[0].keys()))
        writer.writeheader()
        writer.writerows(count_data)


if __name__ == "__main__":
    main()