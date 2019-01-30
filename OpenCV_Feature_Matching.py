import cv2 as cv
import os
from collections import defaultdict

def feature_matcher(query_image, image_folder, method="surf", top_n=5):

	matches_scores = defaultdict()

	img1 = cv.imread(query_image, 0)
	if method == "sift":
		cv_descriptor = cv.xfeatures2d.SIFT_create(nfeatures=800)
	else:
		cv_descriptor = cv.xfeatures2d.SURF_create(800)
	kp1, des1 = cv_descriptor.detectAndCompute(img1, None)

	bf = cv.BFMatcher(cv.NORM_L2)

	images = next(os.walk(image_folder))[2]

	count = 0

	for img in images:
		if count == 10:
			break
		try:
			train_image = image_folder + '/' + img
			img2 = cv.imread(train_image, 0)
			surf = cv.xfeatures2d.SURF_create(800)
			kp2, des2 = surf.detectAndCompute(img2, None)

			matches = bf.knnMatch(des1, des2, k=2)

			good = []
			for m, n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)

			matches_scores[img] = len(good)

		except:
			pass
		count += 1

	return dict(sorted(matches_scores.items(), key=lambda x:x[1], reverse=True)[:top_n]).keys()


if __name__ == '__main__':

	# parse cmd args
	parser = argparse.ArgumentParser(
			description="Feature Matching Image Retrieval"
		)
	parser.add_argument('--query_image', dest="query", required=True)
	parser.add_argument('--image_dir', dest="image_dir", required=True)

	args =  vars(parser.parse_args())
	print(args)

	top_match = feature_matcher(args['query'], args['image_dir'])
	print("Top matches are:\n")
	print(top_match)

