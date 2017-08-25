# -- coding: utf-8 --

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import pickle
import os
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


current_dir = os.path.dirname(os.path.abspath(__file__))
data_set_dir = os.path.join(current_dir, '..', 'static')


warnings.filterwarnings('ignore')


class NewModel:

	def __init__(self, image_set_dir, database, n_clusters, n_features=1000,
				 n_features_pca=50, batch_size=50):
		"""
		Initiate parameters for modeling data set
		@params:
			image_set_dir  - Required  : image data directory path (str)
			database       - Required  : database stored data set information (DataBase)
			n_clusters     - Required  : the number of clusters (int)
			n_features     - Optional  : the number of VGG16 FC-layer features (int)
			n_features_pca - Optional  : the number of principal components (int)
			batch_size     - Optional  : the size of batch to fit PCA and k-means (int)
			
		"""

		self.base_model = VGG16(include_top=True, weights='imagenet', input_tensor=None,
								input_shape=(224, 224, 3))
		self.model = Model(outputs=self.base_model.layers[-1].output, inputs=self.base_model.input)
		self.n_features = n_features
		self.image_set_dir = image_set_dir
		self.batch_size = batch_size

		self.db = database
		self.number_of_images = len([f for f in os.listdir(image_set_dir)
									 if os.path.isfile(os.path.join(image_set_dir, f))])
		self.n_features_pca = n_features_pca
		self.n_clusters = n_clusters
		self.centroids = None
		self.axis = None
		self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=100, batch_size=500, verbose=0, compute_labels=True, 
			random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
		self.scaler = StandardScaler()
		self.pca = IncrementalPCA(n_components=n_features_pca)


	def predict(self, img):
		"""
		Calculate feature vector from the model
		@params:
			img   - Required  : image (224x224 matrix)
			
		@:return:
			prediction : feature vector (numpy.array)
			
		"""
		try:
			x = image.img_to_array(img)
		except Exception as exception:
			print("exception.args")
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		prediction = self.model.predict(x)
		prediction = prediction.reshape([self.n_features])
		std_scale = StandardScaler().fit(prediction)
		prediction = std_scale.transform(prediction)
		return prediction

	def def_pca(self):
		"""
		Define principal components for data set
		
		"""
		partial_features_matrix = np.zeros((self.batch_size, self.n_features))
		for i in tqdm(range(int(self.number_of_images / self.batch_size))):
			for j, key in enumerate(self.db.c.execute('SELECT name FROM image_upload_image WHERE ID BETWEEN ? AND ?',
										 (i*self.batch_size+1, (i+1)*self.batch_size))):
				img = image.load_img(os.path.join(self.image_set_dir, key[0]), target_size=(224, 224))
				prediction = self.predict(img)
				partial_features_matrix[j, :] = prediction
			self.pca.partial_fit(partial_features_matrix)

		# writing pca data into 'pca.pickle' file
		write_pca_file = open(os.path.join('..', 'pickle','pca.pickle'), 'wb')
		pickle.dump(self.pca, write_pca_file)
		write_pca_file.close()


	def def_kmeans(self):
		"""
		Create matrix to feed k-means

		"""
		
		features_matrix = np.zeros((self.batch_size, self.n_features_pca))
		number_of_images = len(os.listdir(self.image_set_dir))
		for i in tqdm(range(int(number_of_images / self.batch_size))):
			for j, data in enumerate(self.db.c.execute('SELECT pca_features FROM image_upload_image WHERE ID BETWEEN ? AND ?', (i * self.batch_size + 1, (i + 1) * self.batch_size))):
				feature_vector = np.frombuffer(data[0], dtype=np.float64)
				features_matrix[j, :] = feature_vector
			self.kmeans.partial_fit(features_matrix)

		self.centroids = self.kmeans.cluster_centers_,

		# writing k-means data into 'kmeans.pickle' file
		pickle.dump( self.kmeans, open( os.path.join('..', 'pickle', 'kmeans.pickle'), "wb" ) )
		file_kmeans = os.path.join('..', 'pickle','kmeans_centroids.pickle')
		write_kmeans_file = open(file_kmeans, 'wb')
		pickle.dump(self.centroids, write_kmeans_file)
		write_kmeans_file.close()

	def load_images(self):
		"""
		Calculate feature vector for all the images 
		in data set and save the results in database
		
		"""

		self.pca = np.load(os.path.join('..', 'pickle', 'pca.pickle'))

		partial_features_matrix = np.zeros((self.batch_size, self.n_features))
		for i in tqdm(range(int(self.number_of_images / self.batch_size))):
			self.db.c.execute('SELECT name FROM image_upload_image WHERE ID BETWEEN ? AND ?',
							  (i * self.batch_size + 1, (i + 1) * self.batch_size))
			for j, key in enumerate(self.db.c.fetchall()):
				img = image.load_img(os.path.join(self.image_set_dir, key[0]), target_size=(224, 224))
				prediction = self.predict(img)
				partial_features_matrix[j, :] = prediction

			prediction_matrix = self.pca.transform(partial_features_matrix)
			prediction_matrix = prediction_matrix.reshape([self.batch_size, self.n_features_pca])
			self.scaler.partial_fit(prediction_matrix)

		for i in tqdm(range(int(self.number_of_images / self.batch_size) * self.batch_size, self.number_of_images + 1)):
			self.db.c.execute('SELECT name FROM image_upload_image WHERE ID=?', (i,))
			data = self.db.c.fetchall()
			img = image.load_img(os.path.join(self.image_set_dir, data[0][0]), target_size=(224, 224))
			prediction = self.predict(img)
			prediction = prediction.reshape((1, self.n_features))
			prediction = self.pca.transform(prediction)
			prediction = prediction.reshape([self.n_features_pca])
			self.scaler.partial_fit(prediction)
		write_scaler_file = open(os.path.join('..', 'pickle', 'scaler.pickle'), 'wb')
		pickle.dump(self.scaler, write_scaler_file)
		write_scaler_file.close()

		partial_features_matrix = np.zeros((self.batch_size, self.n_features))
		for i in tqdm(range(int(self.number_of_images / self.batch_size))):
			self.db.c.execute('SELECT name FROM image_upload_image WHERE ID BETWEEN ? AND ?',
							  (i * self.batch_size + 1, (i + 1) * self.batch_size))
			for j, key in enumerate(self.db.c.fetchall()):
				img = image.load_img(os.path.join(self.image_set_dir, key[0]), target_size=(224, 224))
				prediction = self.predict(img)
				partial_features_matrix[j, :] = prediction
			prediction_matrix = self.pca.transform(partial_features_matrix)
			prediction_matrix = prediction_matrix.reshape([self.batch_size, self.n_features_pca])
			prediction_matrix = self.scaler.transform(prediction_matrix)

			self.db.c.execute('SELECT name FROM image_upload_image WHERE ID BETWEEN ? AND ?',
							  (i * self.batch_size + 1, (i + 1) * self.batch_size))
			for j, key in enumerate(self.db.c.fetchall()):
				self.db.c.execute('UPDATE image_upload_image SET pca_features=? WHERE name=?', (prediction_matrix[j, :], key[0]))
				self.db.conn.commit()

		for i in tqdm(range(int(self.number_of_images / self.batch_size) * self.batch_size, self.number_of_images + 1)):
			self.db.c.execute('SELECT name FROM image_upload_image WHERE ID=?', (i,))
			data = self.db.c.fetchall()
			img = image.load_img(os.path.join(self.image_set_dir, data[0][0]), target_size=(224, 224))
			prediction = self.predict(img)
			prediction = prediction.reshape((1, self.n_features))
			prediction = self.pca.transform(prediction)
			prediction = prediction.reshape([self.n_features_pca])
			prediction = self.scaler.transform(prediction)
			self.db.c.execute('UPDATE image_upload_image SET pca_features=? WHERE name=?', (prediction, data[0][0]))
			self.db.conn.commit()

	def cluster(self):
		"""
		Define cluster for all the images in 
		data set and save the results in database
		
		"""
		file_kmeans = os.path.join('..', 'pickle', 'kmeans.pickle')
		self.kmeans = np.load(file_kmeans)

		features_matrix = np.zeros((self.batch_size, self.n_features_pca))
		number_of_images = len(os.listdir(self.image_set_dir))
		for i in tqdm(range(int(number_of_images / self.batch_size))):
			names = []
			for j, data in enumerate(self.db.c.execute('SELECT name, pca_features FROM image_upload_image WHERE ID BETWEEN ? AND ?',
									 (i * self.batch_size + 1, (i + 1) * self.batch_size))):
				names.append(data[0])
				feature_vector = np.frombuffer(data[1], dtype=np.float64)
				features_matrix[j, :] = feature_vector

			clusters = self.kmeans.predict(features_matrix)
			for (name, cluster) in zip(names, clusters):
				self.db.c.execute('UPDATE image_upload_image SET cluster=? WHERE name=?', (int(cluster), name))
				self.db.conn.commit()
		features_matrix = np.zeros((1, self.n_features_pca))
		for k in tqdm(range(int(number_of_images / self.batch_size)*self.batch_size, number_of_images+1)):
			self.db.c.execute('SELECT name, pca_features FROM image_upload_image WHERE ID=?', (k,))
			data = self.db.c.fetchall()
			feature_vector = np.frombuffer(data[0][1], dtype=np.float64)
			features_matrix[0, :] = feature_vector
			name = data[0][0]
			cluster = self.kmeans.predict(features_matrix)
			self.db.c.execute('UPDATE image_upload_image SET cluster=? WHERE name=?', (int(cluster), name))
			self.db.conn.commit()

	def build_model_pca(self):
		print('Fitting principal components...')
		# self.def_pca()
		print('Transforming initial feature vectors...')
		self.load_images()

	def build_model_cluster(self):
		print('Fitting clusters...')
		self.def_kmeans()
		print('Defining cluster for each item...')
		self.cluster()


if __name__ == '__main__':
	print('vgg16_modelling...')
