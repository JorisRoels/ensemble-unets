import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

POS_W = 25
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67

def dense_crf(img, output_probs, iterations=10):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    # d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(iterations)
    Q = np.array(Q).reshape((c, h, w))
    return Q
