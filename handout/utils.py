import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(im_path):
    """ 
    Given path to an image, load as NumPy array.
    
    Note:
        OpenCV loads images in BGR format, ie first channel is blue, second is green, and third is red.
        matplotlib (which we use to display images) expects RGB format. Hence we perform a color conversion.
    """
    im = cv2.imread(str(im_path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_segmentation(seg_path):
    """ Given path to a segmentation, load as grayscale image (FG: 255 | BG: 0) """
    return cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)


def resize_image(img, fac=0.5, rule=cv2.INTER_AREA):
    """ """
    new_w = int(img.shape[1] * fac)
    new_h = int(img.shape[0] * fac)
    dim = (new_w, new_h)
    
    # resize image
    return cv2.resize(img, dim, interpolation=rule)


def noise_img(img, std):
    return np.clip(img + std * np.random.randn(*img.shape), 0., 1.)


def load_sample(sample_path, resize_fac=None, noise_std=None):
    """
    Args:
        sample_path ... Expected to be Path() object to a sample directory
        resize_fac  ...
        noise_std   ...
        
    Returns:
        A dictionary with following keys, values:
        'img'         ... The RGB image as dtype float and in range [0, 1]
        'scribble_fg' ... 
        'scribble_bg' ... 
        'mask_true'   ... The ground truth segmentation mask - if available
        
    """
    fname = sample_path / 'mask.jpg'
    mask_true = load_segmentation(sample_path / 'mask.jpg')
    
    img = load_image(sample_path / 'im_rgb.jpg')
    if resize_fac is not None:
        img = resize_image(img, resize_fac)
    img = img / 255.
    if noise_std is not None:
        img = noise_img(img, noise_std)
    
    scribble_fg = load_segmentation(sample_path / 'mask_fg.jpg')
    scribble_bg = load_segmentation(sample_path / 'mask_bg.jpg')
    
    if resize_fac is not None: 
        scribble_fg = resize_image(scribble_fg, resize_fac, cv2.INTER_NEAREST)
        scribble_bg = resize_image(scribble_bg, resize_fac, cv2.INTER_NEAREST)
        mask_true = resize_image(mask_true, resize_fac, cv2.INTER_NEAREST)
            
    return {
        'img': img,
        'scribble_fg': scribble_fg,
        'scribble_bg': scribble_bg,
        'mask_true': mask_true
    }


def show_image(im, title=None):
    plt.imshow(im)
    plt.title(title)
    plt.show()


def show_sample(sample_dd, mask_pred=None):
    """
    Args:
        sample_dd ... A dictionary as returned by load_sample()
        mask_pred ... Binary NumPy array with predicted segmentation 
    """
    num_images = 1 + (sample_dd['mask_true'] is not None) + (mask_pred is not None)

    fig, ax = plt.subplots(1, num_images, figsize=(10, 10))

    ax[0].imshow(sample_dd['img'])
    ax[0].set_title('RGB Image')

    idx = 1
    if sample_dd['mask_true'] is not None:
        ax[idx].imshow(sample_dd['mask_true'], interpolation='nearest')
        ax[idx].set_title('True Mask')
        idx += 1 

    if mask_pred is not None:
        ax[idx].imshow(mask_pred, interpolation='nearest')
        ax[idx].set_title('Pred Mask')
    plt.show()


def compute_iou(true, pred):
    """
    Calculate intersection over union metric for binary segmentations
    
    Args:
        true ... Ground truth segmentation mask. Supposed to be boolean array
        pred ... Predicted segmentation mask. Supposed to be boolean array
    """
    assert true.dtype == bool, "Expected boolean arrays"
    assert pred.dtype == bool, "Expected boolean arrays"
    
    intersection = (true * pred).sum()
    union = true.sum() + pred.sum() - intersection
    return intersection / union


def evaluate_segmentation(segmenter, sample_dirs, display=False):
    """ Evaluate segmentation algorithm on images in sample_dirs
    
    Args:
        segmenter   ... Instance of ImageSegmenter
        sample_dirs ... List of paths to sample images
        display     ... If True show results
        
    Returns:
        mean_iou  ... IoU metric averaged over images
        mean_time ... Computation time averaged over images
    """
    iou_score_all = []
    time_all = []
    for sample_path in sample_dirs:

        # Load validation sample
        sample_dd = load_sample(sample_path)

        # Run your segmentation algorithm
        t1 = time.time()
        mask_pred = segmenter.segment_image(sample_dd)
        t2 = time.time()

        iou_score = compute_iou(
            sample_dd['mask_true'].astype(bool),
            mask_pred.astype(bool)
        )
        iou_score_all.append(iou_score)
        time_all.append(t2 - t1)

        if display:
            # Visualize your prediction
            show_sample(sample_dd, mask_pred)
            print(f'IoU score is: {iou_score:0.3f}')

    mean_iou = sum(iou_score_all) / len(iou_score_all)
    mean_time = sum(time_all) / len(time_all)
    
    return mean_iou, mean_time