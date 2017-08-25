import os
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from recommender_django_app.settings import BASE_DIR
from image_upload.models import Image
from recommender_django_app.settings import MODEL, PCA, KMEANS, SCALER, graph
from scipy.stats import pearsonr


class ImageProcessing:

	def get_features(self, img_path):
		x = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(x)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		with graph.as_default():
			prediction = MODEL.predict(x)
			feature_array = prediction.reshape((1000,))
			return feature_array


	def pca_transform(self, feature_array):
		# method to find features after PCA transformation
		pca_feature_array = PCA.transform(feature_array)
		sc_feature_array = SCALER.transform(pca_feature_array)
		return sc_feature_array


	def define_cluster(self, pca_feature_array):
		assignment = KMEANS.predict(pca_feature_array)
		return assignment


	def find_similar(self, cluster, feature_vector, num_nearest=15):
		# find the list of similar images (db id's)
		data = Image.objects.all().filter(cluster=cluster)
		vectors = [np.frombuffer(item.pca_features, dtype=np.float64) for item in data]
		distances = []
		top_k_distances = []
		for vector in vectors:
			distances.append(pearsonr(feature_vector[0], vector)[0])
		distances = np.array(distances)
		top_k_indices = distances.argsort()[(-1)*num_nearest:]
		for i in top_k_indices:
			top_k_distances.append(distances[i])
		return top_k_indices, top_k_distances

	def main(self, img_name):
		"""
		:param
		img_path    :  path to image to find similar of (in media folder)

		:return:
		 sim_images :  list of similar image names (from 'static' folder)

		"""
		img_path = os.path.join(BASE_DIR, 'media', img_name)
		f_vec = self.get_features(img_path)
		f_vec_pca = self.pca_transform(f_vec)
		cluster = self.define_cluster(f_vec_pca)
		sim_inds, sim_dists = self.find_similar(cluster, f_vec_pca)
		sim_images = []
		for ind in sim_inds:
			img_name = Image.objects.get(id=ind+1).name
			sim_images.append(img_name)
		return sim_images


if __name__ == '__main__':
	im_proc = ImageProcessing()
	print(im_proc.main('artdeco_01_014.jpg'))	
	print('Image processing called')