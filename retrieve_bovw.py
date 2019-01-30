import argparse
from collections import defaultdict
import numpy as np
import json
from functools import reduce
import os
import operator
from sklearn.metrics.pairwise import cosine_similarity


def cos_similarity(v1, v2):
	return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

def retrieve_similar_image(img, top_n=5):
	# retrieve the vocabulary for the image in question
	vocab = image_voc[img][0]
	img_to_check = []

	# For each vocabulary, retrieve the list of relevant images
	# from inverted index. This will reduce the number of images to be
	# processed and hence speeding it up. 
	for ix, v in enumerate(vocab):
		if int(v) > 0:
			img_to_check.append(inverted_index[str(ix)])
	img_to_check = np.unique(reduce(operator.add, img_to_check))
	similarities = defaultdict()

	# For each image that is relevant, compute the similarity score
	for image in img_to_check:
		similarities[image] = cos_similarity(
			np.array(computed_tfidf[img]),
			np.array(computed_tfidf[image]))[0][0]

	# Return the top n similar images
	return dict(sorted(similarities.items(),
				key=lambda x:x[1],
				reverse=True)[:top_n]).keys()


if __name__ == '__main__':

	# parse cmd args
	parser = argparse.ArgumentParser(
			description="Retrieve Similar Images"
		)
	parser.add_argument('--pretrained_path', dest="path", required=True)
	parser.add_argument('--image', dest="image", required=True)
	parser.add_argument('--top_n', dest="top_n",
					default=5, type=int)

	args = vars(parser.parse_args())
	print(args)

	with open(os.path.join(args['path'], "inverted_index.txt")) as f:
		inverted_index = json.loads(f.readlines()[0])[0]

	with open(os.path.join(args['path'], "tfidf.txt")) as f:
		computed_tfidf = f.readlines()
		computed_tfidf = json.loads(computed_tfidf[0])[0]

	with open(os.path.join(args['path'], "image_vocabs.txt")) as f:
		image_voc = f.readlines()
		image_voc = json.loads(image_voc[0])[0]

	print(retrieve_similar_image(args['image']))
