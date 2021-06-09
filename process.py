import pathlib
import glob
import tifffile
import numpy as np
import skimage
import bigfish.stack
import bigfish.detection

# CONFIG PARAMETERS
OUTPUT_DIR = "outputs"
INPUT_PATTERN = "inputs/*.tif"
CHANNEL_HRP = 2  # starting from 0
CHANNEL_SHOT = 3
THRESHOLD_LABEL_AREA = 1500
THRESHOLD_LABEL_DISTANCE = 150
MASK_CONTOUR_THICKNESS = 51


def get_nmj_mask_and_contour(image, output_path):
    # Step 1: Find threshold value using triangle algorithm
    # (See paper @ DOI:10.1177/25.7.70454)
    threshold = skimage.filters.threshold_triangle(image)
    image_thresholded = image > threshold
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_hrp-threshold" + ".png"),
        skimage.img_as_ubyte(image_thresholded),
    )
    # Step 2: Perform morphological opening to remove the tracheae, size 5
    # because that's their expected thickness
    image_morph = skimage.morphology.opening(
        image_thresholded, skimage.morphology.disk(5)
    )
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_hrp-morph" + ".png"),
        skimage.img_as_ubyte(image_morph),
    )
    # Step 3: Label image regions
    mask_labels = skimage.measure.label(image_morph)
    # Step 4: Filter out contours by area first, then include smaller (under threshold)
    # contours if they are within acceptable proximity
    mask = np.zeros_like(image_morph)
    regions = skimage.measure.regionprops(mask_labels)
    labels_to_keep = []
    large_region_centroids = []
    for region in regions:
        if region.area > THRESHOLD_LABEL_AREA:
            labels_to_keep.append(region.label)
            large_region_centroids.append(region.centroid)
    for region in skimage.measure.regionprops(mask_labels):
        for large_region_centroid in large_region_centroids:
            if (
                np.linalg.norm(
                    np.asarray(large_region_centroid)
                    - np.asarray(region.centroid)
                )
                < THRESHOLD_LABEL_DISTANCE
            ):
                labels_to_keep.append(region.label)
    for label in labels_to_keep:
        mask[mask_labels == label] = True
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_hrp-mask" + ".png"),
        skimage.img_as_ubyte(mask),
    )
    # Step 5: Derive a contour around the mask
    outer_contour = skimage.morphology.dilation(
        mask, skimage.morphology.disk(MASK_CONTOUR_THICKNESS)
    )
    outer_contour = np.bitwise_xor(outer_contour, mask)
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_hrp-contour" + ".png"),
        skimage.img_as_ubyte(outer_contour),
    )
    # Return mask and contour
    return mask, outer_contour


def calculate_psf(w_ex, w_em, na, ri):
    psf_yx = 0.225 / na * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    psf_z = 0.78 * ri / na ** 2 * w_ex * w_em / np.sqrt(w_ex ** 2 + w_em ** 2)
    return (psf_z, psf_yx)


def fish_processing(mask, contour, zstack, output_path):
    # Step 0: Calculate PSF and sigma
    psf_z, psf_yx = calculate_psf(640, 651, 1.42, 1.516)
    psf_sigma = bigfish.detection.get_sigma(200, 138.1, psf_z, psf_yx)
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
    # Step 7: Filter spots based on location
    spots_mask = []
    spots_contour = []
    for spot in spots:
        if mask[spot[1], spot[2]] == True:
            spots_mask.append(spot)
        if contour[spot[1], spot[2]] == True:
            spots_contour.append(spot)
    print(f"Total spots: {len(spots)}. Mask spots: {len(spots_mask)}. Contour spots: {len(spots_contour)}.")
    # Step 8: Generate contours of the mask and the contour for drawing purposes
    contour_mask = skimage.morphology.dilation(mask, skimage.morphology.disk(1))
    contour_mask = skimage.morphology.dilation(contour_mask, skimage.morphology.disk(1))
    contour_mask = np.bitwise_xor(contour_mask, mask)
    
    contour_contour = skimage.morphology.dilation(contour, skimage.morphology.disk(1))
    contour_contour = skimage.morphology.dilation(contour_contour, skimage.morphology.disk(1))
    contour_contour = np.bitwise_xor(contour_contour, contour)
    # Step 9: Draw the contrasted Z projection, a contour of the mask, and all
    # the spots within the mask
    stack_projection_contrasted = bigfish.stack.rescale(
        stack_projection, channel_to_stretch=0
    )
    output_image = skimage.color.gray2rgb(
        skimage.img_as_ubyte(stack_projection_contrasted)
    )
    output_image[contour_mask] = [255, 0, 255]
    for spot in spots_mask:
        rr, cc = skimage.draw.circle_perimeter(spot[1], spot[2], 3, shape=output_image.shape)
        output_image[rr, cc] = [255, 0, 0]
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_shot-bigfish-mask" + ".png"),
        output_image,
    )
    # Step 10: Draw the contrasted Z projection (already derived), a contour
    # of the contour, and all spots within the contour
    output_image = skimage.color.gray2rgb(
        skimage.img_as_ubyte(stack_projection_contrasted)
    )
    output_image[contour_contour] = [255, 255, 0]
    for spot in spots_contour:
        rr, cc = skimage.draw.circle_perimeter(spot[1], spot[2], 3, shape=output_image.shape)
        output_image[rr, cc] = [0, 127, 255]
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_shot-bigfish-contour" + ".png"),
        output_image,
    )
    # Step 11: Draw the contrasted Z projection (already derived), a contour
    # of both the mask and the contour, and all spots withing the mask and
    # the contour
    output_image = skimage.color.gray2rgb(
        skimage.img_as_ubyte(stack_projection_contrasted)
    )
    output_image[contour_mask] = [255, 0, 255]
    output_image[contour_contour] = [255, 255, 0]
    for spot in spots_mask:
        rr, cc = skimage.draw.circle_perimeter(spot[1], spot[2], 3, shape=output_image.shape)
        output_image[rr, cc] = [255, 0, 0]
    for spot in spots_contour:
        rr, cc = skimage.draw.circle_perimeter(spot[1], spot[2], 3, shape=output_image.shape)
        output_image[rr, cc] = [0, 127, 255]
    skimage.io.imsave(
        output_path.with_name(output_path.stem + "_shot-bigfish-combined" + ".png"),
        output_image,
    )
    # Return the number of spots
    return len(spots_mask), len(spots_contour)


def main():
    # Step 1: Ensure an output directory exists
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    # Step 2: Iterate the images in the input directory
    image_paths = glob.glob(INPUT_PATTERN)
    print(f"Found {len(image_paths)} images.")
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
        # Step 2.4: Get mask and contour
        mask, contour = get_nmj_mask_and_contour(
            zstacks[CHANNEL_HRP], output_image_path
        )
        # Step 2.5: Calculate means of mask and outer contour of channel 1
        mean_mask = np.mean(zstacks[0][mask])
        mean_contour = np.mean(zstacks[0][contour])
        print("Mean value of mask:", mean_mask)
        print("Mean value of contour:", mean_contour)
        # Step 2.6: FISH processing
        spots_mask, spots_contour = fish_processing(
            mask, contour, image[:, CHANNEL_SHOT, :, :], output_image_path
        )
        print("Number of FISH spots within the mask:", spots_mask)
        print("Number of FISH spots within the mask's contour:", spots_contour)
        # Write data to metrics file
        with open(
            output_image_path.with_name(
                output_image_path.stem + "_metrics.txt"
            ),
            "w",
        ) as fout:
            fout.write(f"Mean value of whole mask: {mean_mask:.3f}.\n")
            fout.write(f"Mean value of mask's contour: {mean_contour:.3f}.\n")
            fout.write(f"Number of FISH spots within the mask: {spots_mask}.\n")
            fout.write(f"Number of FISH spots within the contour: {spots_contour}.\n")


if __name__ == "__main__":
    main()
