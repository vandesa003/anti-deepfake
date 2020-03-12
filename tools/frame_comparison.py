import cv2
import numpy as np

real_name = "kqlvggiqee"
fake_name = "cerwqpmouj"

for i in range(10):
    real_rgb = cv2.imread("/Users/bohang.li/Desktop/comparison/{0}_00{1}.jpg".format(real_name, i))
    mask = np.zeros_like(real_rgb)
    fake = cv2.imread("/Users/bohang.li/Desktop/comparison/{0}_00{1}.jpg".format(fake_name, i))
    real = cv2.cvtColor(real_rgb, cv2.COLOR_BGR2GRAY)
    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    flow = cv2.calcOpticalFlowFarneback(
        real, fake, None, pyr_scale=0.5,
        levels=5, winsize=11, iterations=5,
        poly_n=5, poly_sigma=1.1, flags=0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    dense_flow = cv2.addWeighted(real_rgb, 1, rgb, 2, 0)
    cv2.imwrite("/Users/bohang.li/Desktop/comparison/diff_00{0}.jpg".format(i), dense_flow)
