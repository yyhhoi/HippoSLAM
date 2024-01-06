import numpy as np
import cv2 as cv


class SiftMemory:
    def __init__(self, newMemoryThresh=0.6, memoryActivationThresh=0.6, contrastThreshold=0.04, edgeThreshold=0.01):
        assert memoryActivationThresh >= newMemoryThresh
        self.obs_list = []
        self.sift = cv.SIFT_create(nfeatures=50, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
        self.matcher = cv.BFMatcher()
        self.newMemoryThresh = newMemoryThresh
        self.memoryActivationThresh = memoryActivationThresh
        self.min_Ndes = 50


    def observe(self, img):
        """

        Parameters
        ----------
        img: ndarray
            image of an 8-bit integer 2D array.

        Returns
        -------
        ID of the memory (observation)
        Matching Score
        Novel boolean tag

        """

        _, des2 = self.sift.detectAndCompute(img, None)

        # No descriptors
        if des2 is None:
            return [], 0, False

        # Too few descriptors
        if des2.shape[0] < self.min_Ndes:
            return [], 0, False

        # Initialize with the first scene memory
        if len(self.obs_list) < 1:
            self.obs_list.append(des2)
            return [len(self.obs_list)-1], 0, True
        else:
            all_matchscores = np.zeros(len(self.obs_list))
            for i in range(len(self.obs_list)):
                des1 = self.obs_list[i]
                matchscore = self.feature_matching(des1, des2)
                all_matchscores[i] = matchscore

            max_id = np.argmax(all_matchscores)
            max_score = all_matchscores[max_id]


            if max_score <= self.newMemoryThresh:   # Add new memory
                self.obs_list.append(des2)

                if max_score > self.memoryActivationThresh:
                    return [len(self.obs_list) - 1], max_score, True
                else:
                    return [], max_score, True
            else:

                if max_score > self.memoryActivationThresh:
                    return [max_id], max_score, False
                else:
                    return [], max_score, False



            # # The scene is novel
            # familiar_ids = np.where(all_matchscores > self.newMemoryThresh)[0]
            # if familiar_ids.shape[0] < 1:
            #     self.obs_list.append(des2)
            #     return [len(self.obs_list) - 1], 0, True
            # else:
            #     return familiar_ids, all_matchscores[familiar_ids], False



    def feature_matching(self, des1, des2):
        n1, n2 = des1.shape[0], des2.shape[0]
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
        except:
            breakpoint()
        matchscore = len(good) / max(n1, n2)
        return matchscore

