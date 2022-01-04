import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from PIL import Image
import glob
import os


# code taken from:
# https://stackoverflow.com/questions/40592327/deskewing-scanned-image-to-match-original-image-using-opencv-and-sift-surf/40640482#40640482
def warpFinalImage(final_Rec_path, SI_path):
    orig_image = cv2.imread(final_Rec_path, 0)
    skewed_image = cv2.imread(SI_path, 0)
    surf = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    kp2, des2 = surf.detectAndCompute(skewed_image, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))

        ## Perform the calculated Homography on SI image
        im_si = cv2.imread(final_Rec_path.replace('recon-original', 'SI'))
        #im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
        im_out = cv2.warpPerspective(im_si, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
        im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(im_out)
        save_path = final_Rec_path.replace('recon-original', 'SI_warped')
        im.save(save_path)

        #plt.imshow(im_out)
        #plt.show()

    else:
        print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None



path_list_x_down = glob.glob(os.path.join('/media/access/SDB500GB/dev/data_sets/kitti/Ablation/recon-original', '*.png'))
for i in range(len(path_list_x_down)):
    path_recon = path_list_x_down[i]
    path_SI = path_recon.replace('original', 'SI')
    warpFinalImage(path_recon, path_SI)

