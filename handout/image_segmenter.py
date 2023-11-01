import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx
)

from extract_patches import extract_patches
from advanced_methods import perform_min_cut


class ImageSegmenter:
    def __init__(self, k_fg=7, k_bg=19, mode='kmeans', avg_sizes=[40, 60, 70, 75, 80, 90, 100]):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        
        # Number of clusters in FG/BG
        self.k_fg = k_fg
        self.k_bg = k_bg
        
        self.mode= mode
        self.avg_sizes = avg_sizes

    def avg_convolve(self, image, n_size):
        # Transform the image into intensity values
        #image = np.linalg.norm(image, axis=-1)

        #image /= image.max()
        #image *= 255
        
        # Define an averaging kernel over an n_sizeXn_size window. The kernel
        # must have the same size as the image so I padded it with zeros to get
        # the filter.
        kernel = np.ones((n_size, n_size)) * 1/(n_size**2)
        filter = np.zeros(image.shape[:2])
        filter[:kernel.shape[0], :kernel.shape[1]] += kernel

        # convolution step
        img_fft = np.fft.rfft2(image, axes=(0,1))
        filter_fft = np.fft.rfft2(filter)
        # Somtimes the result has a small imaginary part left. We discard it 
        # here.
        for col in range(img_fft.shape[-1]):
            img_fft[:, :, col] *= filter_fft
        res = np.fft.irfft2(img_fft, axes=(0,1), s=image.shape[:2]).real
        '''print(res.shape)
        print(res.max())
        from matplotlib.pyplot import imshow, show
        imshow(res, cmap="Greys")
        show()
        quit()'''
        return res

    def extract_features_(self, sample_dd):
        """ Extract features, e.g. p x p neighborhood of pixel, from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape
        # Make a container for all the averages
        ng_ints = np.zeros((H*W, len(self.avg_sizes)*3))

        for i, size in enumerate(self.avg_sizes):
            reshaped = np.reshape(self.avg_convolve(img, size), (H*W, 3))
            ng_ints[:, i:i+3] = reshaped
        feat = img.reshape((H*W, C))
        
        temp_feat = np.zeros((feat.shape[0], feat.shape[1]+len(self.avg_sizes)*3))
        temp_feat[:,:feat.shape[1]] = feat
        temp_feat[:, feat.shape[1]:] = ng_ints
        feat = temp_feat

        return feat
    

    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        
        # Select the pixel values for the foreground
        # and the background.
        fg = features[(sample_dd["scribble_fg"]==255).flat]
        bg = features[(sample_dd["scribble_bg"]==255).flat]

        # Create the intensity clusters
        fg_clusters=kmeans_fit(fg, self.k_fg)
        bg_clusters=kmeans_fit(bg, self.k_bg)

        # Fuse both groups of clusters so we can compute on all
        # of them with one command.
        clusters = np.concatenate((fg_clusters, bg_clusters))
        # The index of the last foreground cluster in clusters.
        delineate = fg_clusters.shape[0]-1

        # Predict index of closest cluster.
        idxs = kmeans_predict_idx(features, clusters)

        # If the index is smaller than delineate it is a foreground cluster
        # otherwise it is a background cluster.
        return np.reshape(idxs <= delineate, (H, W))

    def segment_image_grabcut(self, sample_dd):
        """ Segment via an energy minimisation """

        # Foreground potential set to 1 inside box, 0 otherwise
        unary_fg = sample_dd['scribble_fg'].astype(np.float32) / 255

        # Background potential set to 0 inside box, 1 everywhere else
        unary_bg = 1 - unary_fg

        # Pairwise potential set to 1 everywhere
        pairwise = np.ones_like(unary_fg)

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_fg, unary_bg, pairwise)
        
        return im_mask

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        elif self.mode == 'grabcut':
            return self.segment_image_grabcut(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")