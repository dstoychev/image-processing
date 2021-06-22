import pathlib
import glob
import tifffile
import numpy as np
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.measure
import skimage.feature
import scipy.ndimage
from PIL import Image, ImageDraw
import bigfish.stack
import bigfish.detection
import csv

# CONFIG PARAMETERS
OUTPUT_DIR = "outputs"
INPUT_PATTERN = "inputs/*.tif"
CHANNEL_RAB7 = 0
CHANNEL_FISH = 2
SEGMENTATION_FILTERING_KERNEL = skimage.morphology.disk(21)
SEGMENTATION_MIN_DISTANCE = 100
PSF_EX = 640
PSF_EM = 651
PSF_NA = 1.42
PSF_RI = 1.516
VOXEL_XY = 34.5
VOXEL_Z = 450


def get_cell_masks(image, output_path):
    # Step 1: Perform median filtering
    image_filtered = skimage.filters.median(
        image, SEGMENTATION_FILTERING_KERNEL
    )
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_rab7-01-filtered" + ".png"),
        skimage.img_as_uint(image_filtered),
    )
    # Step 2: Find threshold value using triangle algorithm
    threshold = skimage.filters.threshold_triangle(image_filtered)
    image_thresholded = image_filtered > threshold
    skimage.io.imsave(
        output_path.with_name(
            output_path.stem + "_rab7-02-threshold" + ".png"
        ),
        skimage.img_as_ubyte(image_thresholded),
    )
    # Step 3: Fill holes
    seed = np.copy(image_thresholded)
    seed[1:-1, 1:-1] = image_thresholded.max()
    image_filled = skimage.morphology.reconstruction(
        seed, image_thresholded, method="erosion"
    )
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_rab7-03-filled" + ".png"),
        skimage.img_as_ubyte(image_filled),
    )
    # Step 4: Perform watershed segmentation
    distance = scipy.ndimage.distance_transform_edt(image_filled)
    local_max_coords = skimage.feature.peak_local_max(
        distance, min_distance=SEGMENTATION_MIN_DISTANCE
    )
    local_max_mask = np.zeros_like(distance)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = skimage.measure.label(local_max_mask)
    image_segmented = skimage.segmentation.watershed(
        -distance, markers, mask=image_filled
    )
    image_segmented_colour = skimage.img_as_ubyte(
        skimage.color.label2rgb(image_segmented, bg_label=0)
    )
    skimage.io.imsave(
        output_path.with_name(
            output_path.stem + "_rab7-04-segmented" + ".png"
        ),
        image_segmented_colour,
    )
    # Step 5: Draw labels over the segmented image
    image_labelled = Image.fromarray(np.copy(image_segmented_colour))
    draw = ImageDraw.Draw(image_labelled)
    for region in skimage.measure.regionprops(image_segmented):
        # Drawing the labels in black will always work because the background
        # is guaranteed to be black and the cells to be coloured
        draw.text(region.centroid[::-1], f"{region.label}", (0,))
    image_labelled = np.array(image_labelled)
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_rab7-05-labelled" + ".png"),
        skimage.img_as_ubyte(image_labelled),
    )
    # Step 6: Return results
    return image_segmented


def calculate_psf(w_ex, w_em, na, ri):
    psf_yx = 0.225 / na * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    psf_z = 0.78 * ri / na ** 2 * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    return (psf_z, psf_yx)


def fish_processing(zstack, output_path):
    # Step 0: Calculate PSF and sigma
    psf_z, psf_yx = calculate_psf(PSF_EX, PSF_EM, PSF_NA, PSF_RI)
    psf_sigma = bigfish.detection.get_sigma(VOXEL_Z, VOXEL_XY, psf_z, psf_yx)
    # Step 2: Z projection
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
        output_path.with_name(output_path.stem + "_fish-spots" + ".png"),
        output_image,
    )
    # Return the spots
    return spots


def main():
    # Step 1: Ensure an output directory exists
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    # Step 2: Iterate the images in the input directory
    image_paths = glob.glob(INPUT_PATTERN)
    print(f"Found {len(image_paths)} images.")
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
        # Step 2.3: Z projection (mean intensity) of all colour channels
        zstacks = np.mean(image, axis=0).astype("uint16")
        # Step 2.4: Get cell masks
        cells = get_cell_masks(zstacks[CHANNEL_RAB7], output_image_path)
        # Step 2.5: FISH processing
        spots = fish_processing(
            image[:, CHANNEL_FISH, :, :], output_image_path
        )
        # Step 2.6: Count spots in each cell
        spots_per_cell = [0] * (np.max(cells) + 1)  # [label] = count
        for spot in spots:
            spots_per_cell[cells[spot[1], spot[2]]] += 1
        for index, count in enumerate(spots_per_cell):
            count_data.append(
                {
                    "image": output_image_path.stem,
                    "label": index,
                    "count": count,
                }
            )
    # Step 3: Combine all the CSV files
    with open(
        pathlib.Path(OUTPUT_DIR).joinpath("metrics.csv"),
        "w",
        newline="",
    ) as fout:
        writer = csv.DictWriter(fout, fieldnames=list(count_data[0].keys()))
        writer.writeheader()
        writer.writerows(count_data)


if __name__ == "__main__":
    main()