import numpy as np
import cv2 as cv


class SiftMemory:
    def __init__(self, matchthresh=0.6, contrastThreshold=0.04, edgeThreshold=0.01):
        self.obs_list = []
        self.sift = cv.SIFT_create(contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
        self.matcher = cv.BFMatcher()
        self.matchthresh = matchthresh


    def observe(self, img):
        """

        Parameters
        ----------
        img: ndarray
            image of an 8-bit integer 2D array.

        Returns
        -------
        ID of the memory (observation)

        """

        _, des2 = self.sift.detectAndCompute(img, None)
        if des2 is None:
            return -1, 0, False

        if len(self.obs_list) < 1:
            self.obs_list.append(des2)
            return len(self.obs_list)-1, 0, True
        else:
            all_matchscores = []
            for des1 in self.obs_list:
                matchscore = self.feature_matching(des1, des2)
                all_matchscores.append(matchscore)

            maxid = np.argmax(all_matchscores)
            maxscore = all_matchscores[maxid]

            # Novel scene, not matching with any existing observation
            if maxscore < self.matchthresh:
                self.obs_list.append(des2)
                return len(self.obs_list)-1, maxscore, True

            else:  # Familiar scene
                return maxid, maxscore, False


    def feature_matching(self, des1, des2):
        n1, n2 = des1.shape[0], des2.shape[0]
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        # TODO: n2 could be 1. Need to exempt this case. Error unsolved.
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
        except:
            breakpoint()
        matchscore = len(good) / max(n1, n2)
        return matchscore

