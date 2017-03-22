# # import cv2
# # from PIL import Image
# # image_file = '/home/chentao/PycharmProjects/table_tennis/train/data/raw_data/f0/h0/left/left-0.jpg'
# # # image = Image.open(image_file)
# #
# # image = cv2.imread(image_file)
# # # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # r = 0.6
# # intps = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
# # print('cv2.INTER_LINEAR', cv2.INTER_LINEAR)
# # print('cv2.INTER_AREA', cv2.INTER_AREA)
# # print('cv2.INTER_CUBIC', cv2.INTER_CUBIC)
# # print('cv2.INTER_NEAREST', cv2.INTER_NEAREST)
# # print('cv2.INTER_LANCZOS4', cv2.INTER_LANCZOS4)
# # # cv2.imshow("Ori", image)
# # for intp in intps:
# #     resized_image = cv2.resize(image, None, fx=r, fy=r, interpolation=intp)
# #     # cv2.imshow("Scaled", resized_image)
# #     cv2.imwrite('/home/chentao/Pictures/%d.jpg'%intp, resized_image)
# #     cv2.imwrite('/home/chentao/Pictures/%d.png'%intp, resized_image)
# # # cv2.waitKey(0)
#
# import numpy as np
# import cv2
# import os
# image_path = '/home/chentao/PycharmProjects/table_tennis/train/data/raw_data/f51/h16/right/'
# image_list = os.listdir(image_path)
# num_images = len(image_list)
#
# fgbg = cv2.BackgroundSubtractorMOG()
# for idx in xrange(num_images):
#     image_name = 'right-%d.jpg'%idx
#     image = cv2.imread(os.path.join(image_path, image_name))
#     fgmask = fgbg.apply(image)
#     cv2.imshow("mask", fgmask)
#     contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.imshow('frame%d'%idx, bin)
#     if contours:
#         # find the largest contour in the mask, then use
#         # it to compute the minimum enclosing circle and
#         # centroid
#         xy_radius = []
#         for contour in contours:
#             if contour.shape[0] < 3:
#                 continue
#             M = cv2.moments(contour)
#             # cx = int(M['m10'] / M['m00'])
#             # cy = int(M['m01'] / M['m00'])
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             xy_radius.append((int(x), int(y), radius))
#             # xy_radius.append((cx, cy, radius))
#         if xy_radius:
#             xy_radius = sorted(xy_radius, key=lambda tup: tup[0])
#             x, y, radius = xy_radius[-1]
#             radius = int(radius)
#             cv2.circle(image, (x, y), radius, (0, 255, 0), 5)
#     cv2.imshow("ori", image)
#     # if idx == 3:
#     #     cv2.waitKey(0)
#     k = cv2.waitKey(300) & 0xff
#     if k == 27:
#         break
#
import rospy
import time
import numpy as np
from pingpang_control.srv import *
if __name__ == "__main__":
    rospy.init_node('request_hitting_pos')
    a = np.random.randint(0, 6, (6, 4))
    rospy.wait_for_service('prediction_interface')
    start = time.time()
    try:
        prediction = rospy.ServiceProxy('prediction_interface', Table_Tennis)
        b = Table_TennisRequest()
        b.inputs = a.reshape(-1)
        resp = prediction(b)
        print(resp)
        end = time.time()
        print('elapsed time:', end-start)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e



