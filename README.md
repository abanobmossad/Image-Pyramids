# Image Pyramids
----
## Goal
>We will use Image pyramids to blending two images
## Theory
>Normally, we used to work with an image of constant size. But in some occassions, we need to work with images of different resolution of the same image. For example, while searching for something in an image, like face, we are not sure at what size the object will be present in the image. In that case, we will need to create a set of images with different resolution and search for object in all the images. These set of images with different resolution are called Image Pyramids (because when they are kept in a stack with biggest image at bottom and smallest image at top look like a pyramid).


![image](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/messipyr.jpg)

**Laplacian Pyramid**

Image Blending using Pyramids
One application of Pyramids is Image Blending. For example, in image stitching, you will need to stack two images together, but it may not look good due to discontinuities between images. In that case, image blending with Pyramids gives you seamless blending without leaving much data in the images. One classical example of this is the blending of two fruits, Orange and Apple.

![image](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/lap.jpg)

**Pyramid Blending**

Load the two images of apple and orange
Find the Gaussian Pyramids for apple and orange From Gaussian Pyramids, find their Laplacian Pyramids
Now join the left half of apple and right half of orange in each levels of Laplacian Pyramids
Finally from this joint image pyramids, reconstruct the original image.

![image](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/orapple.jpg)


## Built with
- matplotlib
- PIL
- scipy
- numpy

## Run
Change the images path and run the ```Pyramid_task.py``` file