import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, misc, signal
from PIL import Image

'''split rgb image to its channels'''


def split_rgb_channels(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return blue, green, red


'''combine image channels to its rgb form'''


def combine_all_channels(r_chan, g_chan, b_chan):
    image = np.zeros((r_chan.shape[0], r_chan.shape[1], 3)).astype(np.uint8)
    image[:, :, 0] = r_chan
    image[:, :, 1] = g_chan
    image[:, :, 2] = b_chan
    return image


'''reduce image by 1/2'''


def reduce(image):

    out = image[::2, ::2]
    return out


'''expand image by factor of 2'''


def expand(image):
    w_1d = np.array([0.25 - 0.4 / 2.0, 0.25, 0.4, 0.25, 0.25 - 0.4 / 2.0])
    W = np.outer(w_1d, w_1d)
    outimage = np.zeros(
        (image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4 * signal.convolve2d(outimage, W, 'same')
    return out


'''create a gaussain pyramid of a given image'''


def apply_reduce(image, levels):
    output = []
    output.append(image)
    tmp = image
    for i in range(0, levels):
        tmp = reduce(tmp)
        output.append(tmp)
    return output


'''build a laplacian pyramid'''


def apply_laplacian(reduced_list):
    output = []
    k = len(reduced_list)
    for i in range(0, k - 1):
        gu = reduced_list[i]
        egu = expand(reduced_list[i + 1])
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(reduced_list.pop())
    return output


'''Blend the two laplacian pyramids by weighting them according to the mask.'''


def blend(first_image, second_image, reduced_list_mask):
    blended_pyramid = []
    k = len(reduced_list_mask)
    for i in range(0, k):
        p1 = reduced_list_mask[i] * first_image[i]
        p2 = (1 - reduced_list_mask[i]) * second_image[i]
        blended_pyramid.append(p1 + p2)
    return blended_pyramid


'''Reconstruct the image based on its laplacian pyramid.'''


def reconstruct_lablacian(laplacian_pyramid):
    output = np.zeros(
        (laplacian_pyramid[0].shape[0], laplacian_pyramid[0].shape[1]), dtype=np.float64)
    for i in range(len(laplacian_pyramid) - 1, 0, -1):
        expanded_lap = expand(laplacian_pyramid[i])
        next_lap = laplacian_pyramid[i - 1]
        if expanded_lap.shape[0] > next_lap.shape[0]:
            expanded_lap = np.delete(expanded_lap, (-1), axis=0)
        if expanded_lap.shape[1] > next_lap.shape[1]:
            expanded_lap = np.delete(expanded_lap, (-1), axis=1)
        tmp = expanded_lap + next_lap
        laplacian_pyramid.pop()
        laplacian_pyramid.pop()
        laplacian_pyramid.append(tmp)
        output = tmp
    return output


def apply_pyramid_blending(image_1, image_2, mask, level=3, what_display="org"):
    r1, g1, b1 = split_rgb_channels(image_1)
    r2, g2, b2 = split_rgb_channels(image_2)
    rm, gm, bm = split_rgb_channels(mask)

    # display the Original images
    if what_display == "org":
        plt.figure("The original images and mask")
        plt.subplot(1, 3, 1)
        plt.imshow(image_1)
        plt.title("Image one")
        plt.subplot(1, 3, 2)
        plt.imshow(image_2)
        plt.title("Image two")
        plt.subplot(1, 3, 3)
        plt.imshow(mask)
        plt.title("The mask")

    # convert the type to deal with float number and in the future will convert it again

    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)

    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)

    rm = rm.astype(float) / 255
    gm = gm.astype(float) / 255
    bm = bm.astype(float) / 255

    # first reduce image one channels

    reduced_list_image1r = apply_reduce(r1, level)
    reduced_list_image1g = apply_reduce(g1, level)
    reduced_list_image1b = apply_reduce(b1, level)

    # display the reduced image 1
    if what_display == "reduce1":
        display_reduced_list_image = [
            reduced_list_image1b, reduced_list_image1g, reduced_list_image1r]
        for x in range(level):
            l = [display_reduced_list_image[0][x], display_reduced_list_image[1]
                 [x], display_reduced_list_image[2][x]]
            reduced_image = combine_all_channels(l[0], l[1], [2])
            plt.figure("Reduced image 1 Level " + str(x))
            plt.imshow(reduced_image)

    reduced_list_image2r = apply_reduce(r2, level)
    reduced_list_image2g = apply_reduce(g2, level)
    reduced_list_image2b = apply_reduce(b2, level)

    # display the reduced image 2
    if what_display == "reduce2":
        display_reduced_list_image = [
            reduced_list_image2b, reduced_list_image2g, reduced_list_image2r]
        for x in range(level):
            l = [display_reduced_list_image[0][x], display_reduced_list_image[1]
                 [x], display_reduced_list_image[2][x]]
            reduced_image = combine_all_channels(l[0], l[1], [2])
            plt.figure("Reduced image 2 Level " + str(x))
            plt.imshow(reduced_image)

    reduced_list_maskr = apply_reduce(rm, level)
    reduced_list_maskg = apply_reduce(gm, level)
    reduced_list_maskb = apply_reduce(bm, level)

    laplacian_pyramid_image1r = apply_laplacian(reduced_list_image1r)
    laplacian_pyramid_image1g = apply_laplacian(reduced_list_image1g)
    laplacian_pyramid_image1b = apply_laplacian(reduced_list_image1b)

    # display the reduced image 2
    if what_display == "lap1":
        display_reduced_list_image = [
            laplacian_pyramid_image1b, laplacian_pyramid_image1g, laplacian_pyramid_image1r]
        for x in range(level):
            l = [display_reduced_list_image[0][x], display_reduced_list_image[1]
                 [x], display_reduced_list_image[2][x]]
            reduced_image = l[0] + l[1] + [2]
            plt.figure("Laplacian for image 1 Level " + str(x))
            plt.imshow(reduced_image, "gray")

    laplacian_pyramid_image2r = apply_laplacian(reduced_list_image2r)
    laplacian_pyramid_image2g = apply_laplacian(reduced_list_image2g)
    laplacian_pyramid_image2b = apply_laplacian(reduced_list_image2b)

    # display the reduced image 2
    if what_display == "lap2":
        display_reduced_list_image = [
            laplacian_pyramid_image2b, laplacian_pyramid_image2g, laplacian_pyramid_image2r]
        for x in range(level):
            l = [display_reduced_list_image[0][x], display_reduced_list_image[1]
                 [x], display_reduced_list_image[2][x]]
            reduced_image = l[0] + l[1] + [2]
            plt.figure("Laplacian for image 2 Level " + str(x))
            plt.imshow(reduced_image, "gray")

    blend_red = blend(laplacian_pyramid_image2r,
                      laplacian_pyramid_image1r, reduced_list_maskr)
    blend_green = blend(laplacian_pyramid_image2g,
                        laplacian_pyramid_image1g, reduced_list_maskg)
    blend_blue = blend(laplacian_pyramid_image2b,
                       laplacian_pyramid_image1b, reduced_list_maskb)

    if what_display == "blend":
        display_reduced_list_image = [blend_blue, blend_green, blend_red]
        for x in range(level):
            l = [display_reduced_list_image[0][x], display_reduced_list_image[1]
                 [x], display_reduced_list_image[2][x]]
            reduced_image = l[0] + l[1] + l[2]
            plt.figure("Blended laplacian Level " + str(x))
            plt.imshow(reduced_image, "gray")

    recon_red = reconstruct_lablacian(blend_red).astype(np.uint8)
    recon_green = reconstruct_lablacian(blend_green).astype(np.uint8)
    recon_blue = reconstruct_lablacian(blend_blue).astype(np.uint8)

    # display the blended result
    if what_display == "result":
        result = np.zeros(image1.shape, dtype=image1.dtype)
        tmp = [recon_blue, recon_green, recon_red]
        result = combine_all_channels(recon_blue, recon_green, recon_red)
        plt.figure("Result with level "+str(level))
        plt.imshow(result)

    plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# image1 = np.array(Image.open("test_images/cat.jpg").convert('RGB').resize((227, 224)))
# image2 = np.array(Image.open("test_images/dog.jpg").convert('RGB').resize((227, 224)))
# 
image1 = ndimage.imread('test_images/orange.jpg')
image2 = ndimage.imread('test_images/apple.jpg')
mask = misc.imresize(ndimage.imread('test_images/mask.jpg'), image1.shape)

''' displaying parameters '''
#-----------------------------
# org     = > for the original images.
# lap1    = > for the laplacian pyramid for the first image.
# lap2    = > for the laplacian pyramid for the second image.
# reduce1 = > for the gaussian pyramid for the first image.
# reduce2 = > for the gaussian pyramid for the second image.
# blend   = > for the blending pyramid.
# result  = > for the last result .

apply_pyramid_blending(image2, image1, mask, level=5, what_display="result")
