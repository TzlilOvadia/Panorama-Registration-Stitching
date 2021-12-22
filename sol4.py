import time

import numpy as np
import scipy
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.color
import scipy.ndimage as scimage
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite
import os as os
import sol4_utils

# ################ #
# ####CONSTANTS### #
# ################ #

K = 0.04
RGB = 2
GRAY_SCALE = 1
RGB_FORMAT = 3
EVEN_PIXELS = 2
D_X = np.array([1, 0, -1]).reshape(1,3)
D_Y = D_X.T


def genMatrix(im_x,im_y):
    """
    helper function to generate the matrix which is used to generate the response image R.
    :param im_x: image derivative with respect to x
    :param im_y: image derivative with respect to y
    :return: matrix of the form [[i_x_blurred, i_yx_blurred],[i_yx_blurred, i_y_blurred]]
    """
    i_x_blurred = sol4_utils.blur_spatial(im_x*im_x, 3)
    i_y_blurred = sol4_utils.blur_spatial(im_y*im_y, 3)
    i_yx_blurred = sol4_utils.blur_spatial(im_x*im_y, 3)
    m = [[i_x_blurred, i_yx_blurred],[i_yx_blurred, i_y_blurred]]
    return m

def getResponse(mat):
    """
    helper function to calculate the response of the given matrix, with the formula we saw in class
    :param mat:
    :return:
    """
    det = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]
    trace = mat[0][0] + mat[1][1]
    return det - K*(trace**2)


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # Step 1: Get the Ix and Iy derivatives of the image using the filters [1, 0, −1], [1, 0, −1]T respectively.
    i_x = scimage.convolve(im, D_X)
    i_y = scimage.convolve(im, D_Y)
    # Step 2: Blur the images: Ix2 , Iy2 , IxIy. You may use blur_spatial function from sol4_utils.py with kernel_size=3.
    m = genMatrix(i_x, i_y)
    # Step 4: Build the response image R using the formula we saw in class
    responseIm = getResponse(m)
    # Step 5: Finding R for every pixel results in a response image R. The corners are the local maximum points of R
    res = non_maximum_suppression(responseIm)
    pos = np.argwhere(res)
    return np.flip(pos, axis=1)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    # Step 1: Create a patch for each point in pos parameter in radius of desc_rad*2+1
    descriptors = np.zeros((len(pos),desc_rad*2+1,desc_rad*2+1))
    for i in range(len(pos)):
        # fetch the corresponding (x,y) for the ith iteration:
        x = pos[i][0]
        y = pos[i][1]
        # Create the size of the patch and the corresponding indices for the patch, relative to the image:
        r1 = np.arange(desc_rad * 2 + 1)
        x_pos = (np.ones((desc_rad * 2 + 1, desc_rad * 2 + 1))[r1,:]*np.array(np.arange(x-desc_rad,x+desc_rad+1))).flatten()
        y_pos = (np.ones((desc_rad * 2 + 1, desc_rad * 2 + 1))[r1,:]*np.array(np.arange(y-desc_rad,y+desc_rad+1))).T.flatten()
        indices = np.array([ y_pos,x_pos])
        # Using map_coordinates, fetch the patch, when out of boundaries are zeros:
        descriptors[i,:,:] = standartizeDescriptors(map_coordinates(im, indices, order=1,prefilter=False)
                                                    .reshape(desc_rad*2+1, desc_rad*2+1))
    return descriptors


def standartizeDescriptors(descriptors):
    """
    given a descriptor, this function will return a normalized version of the descriptor.
    :param descriptors:
    :return:
    """
    enumerator = (descriptors - np.mean(descriptors))
    denominator = (np.linalg.norm(enumerator))
    if denominator:
        return enumerator/denominator
    return descriptors


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    pos = spread_out_corners(pyr[0], 7, 7, 7)
    sampleDescriptor = sample_descriptor(pyr[2],pos.astype(np.float64)/4,3)
    return [pos,sampleDescriptor]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # Generate the product of all matrices in desc1 and desc2:
    s_matrix = np.tensordot(desc1,desc2, axes=((1,2),(1,2)))
    # At this point, s_matrix is N1 by N2 matrix, where each k,j coordinate in it represents the dot product between
    # the D_i,k and D_(i+1),j descriptors. We will now use it to find the 2nd maximal value on each row and column
    # inorder to find the matched features:
    max_cols = (np.partition(s_matrix,kth=-2,axis=0)[-2,:])[None, :]
    max_rows = (np.partition(s_matrix,kth=-2, axis=1)[:,-2])[:, None]
    matched = np.flip(np.argwhere((s_matrix >= max_rows) & (s_matrix >= max_cols) & (s_matrix > min_score)))
    matched = [matched[:,1], matched[:,0]]
    return matched




def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    points = np.arange(len(pos1))
    # Step 1: modify the dims of pos1 from (x,y) to (x,y,1):

    pos1_3d = np.ones((pos1.shape[0],3))
    pos1_3d[ : ,:-1] = pos1

    # Step 2: prepare the output for this function:
    pos2_temp = np.zeros(pos1_3d.shape)

    # Step 3: calculate the trasformation for each (x,y,1) from pos1 to pos2:
    pos2_temp[points] = (H12@pos1_3d[points,:].T).T

    # Step 4: recalculate the result of pos2_temp to be (x,y) by dividing the each 3-d point by the z component:
    pos2 = np.array([pos2_temp[:,0]/pos2_temp[:,2], pos2_temp[:,1]/pos2_temp[:,2]]).T

    return pos2



def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inliers_ind = []
    point1,point2 = None, None

    for iter in range(num_iter):
        # Step 1:Pick a random set of 2 point matches from the supplied N point matches. Let’s denote their indices by J.
        rand_indices = np.random.choice(points1.shape[0], size=2)
        # We call these two sets of 2 points in the two images P1,J and P2,J:
        p1j1 = points1[rand_indices[0]]
        p1j2 = points1[rand_indices[1]]

        p2j1 = points2[rand_indices[0]]
        p2j2 = points2[rand_indices[1]]
        p1j = np.array([p1j1,p1j2])
        p2j = np.array([p2j1, p2j2])

        # Step 2: Compute the homography H1,2 that transforms the 2 points P1,J to the 2 points P2,J . We will use
        # estimate_rigid_transform function to performs this step. This function returns a 3x3 homography matrix which
        # performs the rigid transformation:

        H12 = estimate_rigid_transform(p1j, p2j, translation_only)
        est_point2 = apply_homography(points1, H12)

        # Step 3: Estimate the distance between the original points2 to the estimated set of points - est_point2:
        euclideanDist = euclideanDistance(points2,est_point2)

        # Step 4: filter out the points which is smaller than the inlier_tol THRESHOLD, and assign it as the inliers:
        euclideanDist[euclideanDist < inlier_tol] = 0
        inliers = np.count_nonzero(euclideanDist == 0)
        # Step 5: check if the current iteration improved the result for the inliers group:
        if len(max_inliers_ind) < inliers:
            # Should update the set of points which maximise the inliers count:
            point1 = p1j
            point2 = p2j
            index = np.argwhere(euclideanDist == 0)
            max_inliers_ind = index.reshape(index.shape[0], )

    H12 = estimate_rigid_transform(point1,point2,translation_only)
    return [H12,max_inliers_ind]


def euclideanDistance(points2, points2Tag):
    """
    using this function in order to calculate the the euclidean distance between the original points (points2) and the
    estimated points (points2Tag).
    :param points2: the original set of points
    :param points2Tag: the estimated set of points, got after applying apply_homography() on points1 and H12.
    :return: an array where the i'th element represents the absolute distance between points2[i] and points2Tag[i]
    """
    euclideanDist = np.linalg.norm(points2 - points2Tag,axis=1)**2
    return euclideanDist

def plotAssistant(vec_x1, vec_x2, vec_y1, vec_y2, n, is_inlier=True):
    """Helper function for plotting the layers of matches between two consecutive frames"""
    if is_inlier:
        for iter in range(n):
            plt.plot([vec_x1[iter:iter+2],vec_x2[iter:iter+2]],[vec_y1[iter:iter+2],vec_y2[iter:iter+2]],
                     mfc='r', c='y',lw= .4, ms=10,
                     marker='o')
    else:
        for iter in range(n):
            plt.plot([vec_x1[iter:iter+2],vec_x2[iter:iter+2]],[vec_y1[iter:iter+2],vec_y2[iter:iter+2]],
                     mfc='r', c='b',lw= .4, ms=10,
                     marker='o')
    return

def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    stacked_image = np.hstack((im1, im2))
    plt.imshow(stacked_image,cmap='gray')
    # get the inlier matches from each image:
    inliers1,inliers2 = points1[inliers],points2[inliers]
    inliers1_x, inliers1_y, inliers2_x, inliers2_y = _get_Inliers(inliers1, inliers2)
    outliers1,outliers2 = points1[np.setdiff1d(np.arange(len(points1)), inliers)],\
                          points2[np.setdiff1d(np.arange(len(points1)), inliers)]
    # Draw the inliers:
    plotAssistant(inliers1_x,inliers2_x,inliers1_y,inliers2_y, len(inliers))
    # Draw the outliers:
    plotAssistant(outliers1[:,0], outliers2[:,0], outliers1[:,1], outliers2[:,1], len(outliers1), is_inlier=False)
    plt.show()


def _get_Inliers(inliers1, inliers2):
    inliers1_x = inliers1[:, 0]
    inliers1_y = inliers1[:, 1]
    inliers2_x = inliers2[:, 0]
    inliers2_y = inliers2[:, 1]
    return inliers1_x, inliers1_y, inliers2_x, inliers2_y

def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """

    H2m = [0 for _ in range(len(H_succesive)+1)]
    H2m[m] = np.eye(3)
    toStartMat,toEndMat = np.eye(3),np.eye(3)
    # Compute the H matrices from m-1 down to 0:
    for i in range(m-1,-1,-1):
        toStartMat = toStartMat @ H_succesive[i]
        # Normalize the matrix such that the z-coord whould be equal to 1:
        H2m[i] = toStartMat / toStartMat[2,2]

    # Compute the H matrices from m+1 up to M-1:
    for i in range(m,len(H_succesive)):
        toEndMat = toEndMat @ np.linalg.inv(H_succesive[i])
        H2m[i+1] = toEndMat / toEndMat[2,2]

    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """

    # Given an image's width and height, compute its corners:
    top_r,top_l, bottom_r,bottom_l = [w-1,h-1],[0,h-1],[w-1,0],[0,0]
    image_boundries = np.array([top_r,top_l,bottom_r,bottom_l])

    # Calculate the transformation for the image's boundries using the homography:
    xy_homography = apply_homography(image_boundries, homography)
    x_val,y_vals = xy_homography[:,0], xy_homography[:,1]

    # Find the minimal and maximal values on each axis after applying the homography on it:
    max_x, max_y, min_x, min_y = np.max(x_val),np.max(y_vals), np.min(x_val), np.min(y_vals)

    # Return the values in the described format:
    box_corners = [[min_x, min_y],[max_x,max_y]]
    return np.array(box_corners).astype(np.int)

def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # Set the boundaries for the warped image:
    bounding_box = compute_bounding_box(homography,image.shape[1],image.shape[0])
    min_x,max_y,max_x,min_y = bounding_box[0][0], bounding_box[1][1], bounding_box[1][0],bounding_box[0][1]
    x,y = np.meshgrid(np.arange(min_x,max_x+1), np.arange(min_y,max_y+1))
    inv_homography = np.linalg.inv(homography)
    # get the x,y arrays to be stacked as (x_i, y_i) in a sequence called pos:
    pos = np.stack([x,y], axis=2)
    # Apply homography with the inv_homography matrix inorder to get the indices with respect to frame i:
    frame_i_coords = apply_homography(pos.reshape(pos.shape[0]*pos.shape[1],2), inv_homography).T
    # Rehape it back to x coordinates and y coordinates to pass onto the map_coordinates function:
    frame_i_coords = np.flip(frame_i_coords, axis=0)
    # interpolate the coordinates we got to the new coordinates
    new_coords = map_coordinates(image,frame_i_coords,order=1,prefilter=False).reshape(x.shape)
    return new_coords


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


################################
### Helper functions section ###
################################
def _buildGaussianVec(sizeOfVector):
    """
    Helper function to generate the gaussian vector with the size of the sizeOfVector
    """
    if sizeOfVector <= 2:
        return np.array([np.ones(sizeOfVector)])
    unitVec = np.ones(2)
    resultVec = np.ones(2)
    for i in range(sizeOfVector - 2):
        resultVec = scipy.signal.convolve(resultVec, unitVec)
    return np.array(resultVec/np.sum(resultVec)).reshape(1, sizeOfVector)



def _reduceImage(image, filter_vec):
    """
    a simple function to reduce the image size after blurring it
    :param image: image to reduce
    :return: reduced image
    """
    # Step 1: Blur the image:
    blurredImage = _blurImage(filter_vec, image)

    # Step 2: Sub-sample every 2nd pixel of the image, every 2nd row, from the blurred image:
    reducedImage = blurredImage[::EVEN_PIXELS,::EVEN_PIXELS]
    return reducedImage


def _blurImage(filter_vec, image):
    #Step 1: blur the rows:
    blurredImage = scipy.ndimage.filters.convolve(image,filter_vec)

    # Step 2: complete the blurred image:
    blurredImage = scipy.ndimage.filters.convolve(blurredImage, filter_vec.T)

    return blurredImage


def _expandImage(image, filter_vec):
    """

    :param image:  image to expand
    :param filter_vec:
    :return:
    """
    # Step 1: Expand the image using zeros on odd pixels:
    expandedImage = np.zeros((image.shape[0]*2,image.shape[1]*2))
    expandedImage[::EVEN_PIXELS,::EVEN_PIXELS] = image

    # Step 2: Blur the expanded image:
    blurredExpandedImage = _blurImage(filter_vec*2,expandedImage)
    return blurredExpandedImage


def _max_levels_calc(max_levels, im):
    """
    Simple helper function to calculate the maximum levels of the pyramid, given the
    restrictions on the assignment's pdf
    :return: the correct maximum levels of the pyramid we will calculate
    """
    widthLayers = np.log2(im.shape[1]/16)
    heightLayers = np.log2(im.shape[0]/16)
    max_levels = min(max_levels, int(widthLayers) + 1,
        int(heightLayers) + 1)
    return max_levels

def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).

    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
                        to be used in constructing the pyramid filter
                        (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume the filter size will be >=2.
    :return:
    """

    pyr = []
    gaussian_vec = _buildGaussianVec(filter_size)
    max_levels = _max_levels_calc(max_levels, im)
    tmp = im

    for i in range(max_levels):
        pyr.append(tmp)
        tmp = _reduceImage(np.copy(tmp), gaussian_vec)

    return pyr, gaussian_vec



def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size:
    :return:
    """
    #Step 1: calculate the gaussian pyramid so we can use it for next calculations:
    gauPyramid,filter_vec = build_gaussian_pyramid(im, max_levels, filter_size )

    #Step 2: Initialize the pyramid and add to each of its levels the correct value (using the formula we learnd):
    pyr = []
    for i in range(len(gauPyramid)-1):
        pyr.append(gauPyramid[i] - _expandImage(gauPyramid[i+1], filter_vec))

    #Step 3: add the last level of the pyramid:
    pyr.append(gauPyramid[-1])

    return pyr, filter_vec



def read_image(filename, representation):
    """
    filename - the filename of an image on disk (could be grayscale or RGB).
    representation - representation code, either 1 or 2 defining whether the output should be a:
    grayscale image (1)
    or an RGB image (2).
    NOTE: If the input image is grayscale, we won’t call it with represen- tation = 2.
    :param filename: String - the address of the image we want to read
    :param representation: Int - as described above
    :return: an image in the correct representation
    """
    if representation != RGB and representation != GRAY_SCALE:
        return "Invalid Input. You may use representation <- {1, 2}"
    tempImage = plt.imread(filename)[:, :, :3]
    resultImage = np.array(tempImage)

    if representation == GRAY_SCALE:
        resultImage = skimage.color.rgb2gray(tempImage)
    elif representation == RGB:
        resultImage = tempImage
    if resultImage.max() > 1:
        resultImage = resultImage/256

    return resultImage.astype(np.float64)



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: laplacian pyramid
    :param filter_vec: filter vector
    :param coeff: python array of coefficients
    :return: reconstructed image from the lpyr
    """
    #Step 1: initialize the image we want to return:
    resultImage = lpyr[-1]*coeff[-1]

    #Step 2: iterate through all the levels of the pyramid and reconstruct an image out of it:
    for i in range(2, len(lpyr) + 1):
        resultImage = coeff[-i]*lpyr[-i] + _expandImage(resultImage,filter_vec)

    return resultImage

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1,im2: are two input grayscale images to be blended.
    :param mask:is a boolean (i.e. dtype == np.bool) mask containing
                True and False representing which parts of im1 and im2 should appear in the resulting im_blend.
                Note that a value of True corresponds to 1, and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter)
                             which defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: Blended image
    """
    resImg = np.zeros(im1.shape)
    mask = mask.astype(np.float64)
    # Build Gaussian pyramid for the mask:
    G, filter_vec = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    if len(im1.shape) == RGB_FORMAT:
        for channel in range(len(im1.shape)):
            # Build laplacian pyramid for im1 and im2:
            L_a = build_laplacian_pyramid(im1[:,:,channel],max_levels, filter_size_im)[0]
            L_b = build_laplacian_pyramid(im2[:,:,channel], max_levels, filter_size_im)[0]


            # Build laplacian pyramid using the formula we saw in class:
            L_c = np.multiply(G,L_a) + np.multiply(np.subtract(1, G),L_b)
            resImg[:,:,channel] = laplacian_to_image(L_c,filter_vec,np.ones(max_levels))
    else:
        # Build laplacian pyramid for im1 and im2:
        L_a = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
        L_b = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

        # Build laplacian pyramid using the formula we saw in class:
        L_c = np.multiply(G, L_a) + np.multiply(np.subtract(1, G), L_b)
        resImg = laplacian_to_image(L_c, filter_vec, np.ones(max_levels))

    return np.clip(resImg,0,1)

if __name__ == '__main__':
    #
    # # x = np.array([[1,1,1,1],[2,2,2,2]])
    # x = np.eye(3)
    # y = np.concatenate((x,np.ones((3,1)).T))
    # print(y)
    # # ## Testing the functions:
    is_bonus = False
    experiments = ['iguazu.mp4', 'boat.mp4']

    for experiment in experiments:
        exp_no_ext = experiment.split('.')[0]
        os.system('mkdir dump')
        os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
        os.system(
            ('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (
            experiment, exp_no_ext, exp_no_ext))

        s = time.time()
        panorama_generator = PanoramicVideoGenerator(os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100,
                                                          bonus=is_bonus)
        panorama_generator.align_images(translation_only='boat' in experiment)
        panorama_generator.generate_panoramic_images(9)
        print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

        panorama_generator.save_panoramas_to_video()
