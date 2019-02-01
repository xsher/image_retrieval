import cv2
import os
import numpy as np
import json
import time
import math
import redis
import argparse

import caffeine
caffeine.on(display=True)

from glob import glob
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from collections import defaultdict


class KMeans:
	"""
	Class that performs the KMeans related tasks such as fitting and predicting
	"""

	def __init__(self, n_clusters):
		self.n_clusters = n_clusters
		self.compactness = None
		self.label = None
		self.center = None
		
	def fit(self, data):
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		self.compactness, self.label, self.center = cv2.kmeans(
			data, self.n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	def predict(self, vectors):
		"""
		Vector may need to be reshaped with (1, -1)
		Takes in a list of vectors
		"""
		cluster_labels = []
		for vect in vectors:
			dist = []
			for idx, c in enumerate(self.center):
				dist.append(euclidean_distances(vect.reshape(1, -1), c.reshape(1, -1)))
			cluster_labels.append(dist.index(min(dist)))

		return cluster_labels


class DescriptorGenerator:
	"""
	Class to help with image processing and retrieval of descriptors
	"""

	def __init__(self, method='sift', nfeatures=1000):
		self.des_obj = None
		if method == "sift":
			self.des_obj = cv2.xfeatures2d.SIFT_create(
				nfeatures=nfeatures)
		elif method == "surf":
			self.des_obj = cv2.xfeatures2d.SURF_create()

	def gray(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return gray

	def features(self, image):
		"""
		Retrieve the keypoints and descriptors for an image
		"""
		keypoints, descriptors = self.des_obj.detectAndCompute(image, None)
		return [keypoints, descriptors]


class BOVW:
	"""
	BOVW class performs the relevant tasks to generate the data required for
	image retrieval.
	"""
	def __init__(self, data_path, method='sift', n_clusters=10,
			nfeatures=1000):
		self.t = int(time.time())
		self.n_clusters = n_clusters
		self.path = data_path

		self.opencv = DescriptorGenerator(method=method, 
			nfeatures=nfeatures)
		self.kmeans = KMeans(n_clusters=n_clusters)
		self.descriptor_vstack = None

		self.images = None
		self.image_count = 0
		self.descriptor_list = []
		self.image_vocab = {}
		self.tfidf_corpus = None

	def cluster(self):
		"""    
		Cluster using KMeans algorithm implemented with opencv 
		"""
		self.kmeans.fit(self.descriptor_vstack)
		with open("{}/kmeans_results.csv".format(self.t), "w") as f:
			json.dump([self.kmeans.compactness], f)
			f.write("\n")
			json.dump([self.kmeans.label.tolist()], f)
			f.write("\n")
			json.dump([self.kmeans.center.tolist()], f)
  
	def format_data(self, list_items):
		"""    
		Convert list into vstack array of shape M samples x N features
		for kMeans training
		"""
		v_stack = np.array(list_items[0])
		print(len(list_items))
		count = 0
		for r in list_items[1:]:
			print("stacking: ", count)
			v_stack = np.vstack((v_stack, r))
			count += 1
		self.descriptor_vstack = v_stack.copy()
		return v_stack

	def plot_hist(self):
		"""
		Plot the histogram for the distribution of the vocabularies i.e. clusters
		"""
		print("Plotting histogram")
		counts = Counter(self.kmeans.label.flatten())

		x = list(counts.keys())
		y = list(counts.values())

		plt.bar(x, y)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(np.array(x) + 0.4, x)
		plt.savefig("visual_word_histogram.jpg")

	def load_images(self):
		imlist = {}
		count = 0
		print("Loading images from: ", self.path)
		for f in glob(self.path + "*"):
			imfile = f.split("/")[-1]
			print("Loading image file {} ==========".format(imfile))
			im = cv2.imread(f, 0)
			imlist[imfile] = im
			count +=1

		self.images = imlist
		self.image_count = count

	def train_vocabulary(self):
		"""
		This function loads the images, generates the descriptors
		and performs the clustering
		"""

		if not self.images:
			self.load_images()

		des_list = []
		for im, imlist in self.images.items():
			print("Computing Features for image: ", im)
			lkp, des = self.opencv.features(imlist)
			print("descriptor length", len(des))
			self.descriptor_list.append(des)
			des_list.append(list(des.tolist()))

		print("Saving descriptor list")
		with open("{}/descriptor_list.txt".format(self.t), "w") as f:
			json.dump(des_list, f)

		print("Formatting data")
		self.format_data(self.descriptor_list)
		print("Performing Clustering")
		self.cluster()
		self.plot_hist()

	def generate_vocabulary(self):
		"""
		Generates vocabulary for each image
		"""
		print("Generating Vocabulary for each image")
		self.image_vocab = {}
		for im, imlist in self.images.items():
			print("Image: ", im)
			vocab = self.generate_vocab(imlist)
			self.image_vocab[im] = list(vocab)

		# save a copy first
		with open("{}/image_vocabs.txt".format(self.t), "w") as f:
			try:
				json.dump([self.image_vocab], f)
			except:
				pass
		return self.image_vocab

	def generate_redis(self, data_dict, redis_path):
		print("Generating file for redis")
		with open(redis_path, "w") as redis_f:
			for key, value in data_dict.items():
				redis_f.write("SET {} {}".format(key, value))
				redis_f.write("\n")

	def generate_vocab(self, img):
		""" 
		This method generates the vocabulary for an image
		It returns a vector of length n_clusters with its frequency count
		"""
		kp, des = self.opencv.features(img)

		vocab = [[0 for i in range(self.n_clusters)]]

		test_ret = self.kmeans.predict(des)

		for each in test_ret:
			vocab[0][each] += 1

		return vocab


class TFIDF:

	def __init__(self, corpus, n_clusters):
		self.corpus = corpus
		self.idf = None
		self.n_clusters = n_clusters

	def compute_TF(self, bow):
		"""
		Term Frequency = Number of occurences of word j in document d /
						 Number of words in document d
		The parameter bow refers to the bag of words which represents
		the frequencey of each word in the bag
		"""
		n_words = len(bow[0])
		tf_bow = np.array(bow[0]) / n_words
		return tf_bow

	def generate_IDF_dict(self):
		"""
		IDF = log(total no. of docs / number of docs which has word j)
		"""
		n_docs = len(self.corpus)
		idf_dict = dict.fromkeys(range(self.n_clusters), 0)

		for img, corp in self.corpus.items():
			for idx, count in enumerate(corp[0]):
				if int(count) > 0:
					idf_dict[idx] += 1
		for idx, count in idf_dict.items():
			idf_dict[idx] = math.log10(n_docs / float(count))

		self.idf = idf_dict
		return idf_dict

	def compute_TFIDF(self, bow):
		"""
		TFIDF = TF * IDF
		"""
		tf_bow = self.compute_TF(bow)
		tfidf = np.zeros(self.n_clusters)
		for idx, tf in enumerate(tf_bow):
			tfidf[idx] = tf * self.idf[idx]
		return tfidf

	@staticmethod
	def cosine_similarity(v1, v2):
		return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))


class Redis:

	def __init__(self):
		self.redis = redis.Redis(host='localhost', port=6379, db=0)

	def mset(self, dict_vals):
		"""
		dict_vals format: k, list(v)
		"""
		redis_f = defaultdict()
		for k, v in dict_vals.items():
			redis_f[k] = json.dumps(v)

		self.redis.mset(redis_f)

	def get(self, key):
		return json.loads(self.redis.get(key))


def generate_inverted_index(image_vocab):
	inv_indx = defaultdict(list)
	for imgidx, vocab in image_vocab.items():
		for idx, count in enumerate(vocab[0]):
			if int(count) > 0:
				inv_indx[idx].append(imgidx)
	return inv_indx


if __name__ == '__main__':

	# parse cmd args
	parser = argparse.ArgumentParser(
			description="Bag of Visual Words"
		)
	parser.add_argument('--path', dest="path", required=True)
	parser.add_argument('--n_clusters', dest="n_clusters",
						default=100, type=int)
	parser.add_argument('--nfeatures', dest="nfeatures",
						default=1000, type=int)

	args = vars(parser.parse_args())
	print(args)

	bov = BOVW(args['path'], n_clusters=args['n_clusters'])

	os.mkdir(str(bov.t))
	print("Saving to directory: {}".format(bov.t))

	print("Training for vocabularies")
	bov.train_vocabulary()

	print("Generating vocabulary for each image")
	bov.generate_vocabulary()

	print("Generating TFIDF")
	tfidf = TFIDF(bov.image_vocab, args['n_clusters'])

	tfidf.generate_IDF_dict()
	# Dump out the IDF dict so that it can be extended for unknown images if 
	# required. - currently not implemented yet
	with open("{}/idf_dict.txt".format(bov.t), "w") as f:
		try:
			json.dump([bov.image_vocab], f)
		except:
			pass

	print("Generating TFIDF for each image")
	img_tfidf = defaultdict(list)
	for k, img_hist in bov.image_vocab.items():
		img_tfidf[k] = tfidf.compute_TFIDF(img_hist).tolist()

	with open("{}/tfidf.txt".format(bov.t), "w") as f:
		json.dump([img_tfidf], f)

	print("Generating inverted index")
	invt_idx = generate_inverted_index(bov.image_vocab)
	with open("{}/inverted_index.txt".format(bov.t), "w") as f:
		json.dump([invt_idx], f)

	caffeine.off()
