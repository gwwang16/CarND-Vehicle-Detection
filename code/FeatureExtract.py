import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog


class FeatureExtract:
    def __init__(self, feature_params):
        self.color_space = feature_params['color_space']
        self.orient = feature_params['orient']
        self.pix_per_cell = feature_params['pix_per_cell']
        self.cell_per_block = feature_params['cell_per_block']
        self.spatial_size = feature_params['spatial_size']
        self.hist_bins = feature_params['hist_bins']
        self.hog_channel = feature_params['hog_channel']
        self.spatial_feat = feature_params['spatial_feat']
        self.hist_feat = feature_params['hist_feat']
        self.hog_feat = feature_params['hog_feat']

    def bin_spatial(self, img, size=(32, 32)):
        """Compute binned color features"""
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        """Compute color histogram features"""
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            # Use skimage.hog() to get both features and a visualization
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell = (pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      visualise=vis, feature_vector=feature_vec,
                                      block_norm="L2-Hys")
            return features, hog_image
        else:      
            # Use skimage.hog() to get features only
            features = hog(img, orientations=orient, 
                           pixels_per_cell = (pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           visualise=vis, feature_vector=feature_vec,
                           block_norm="L2-Hys")
            return features


    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      

            if spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features