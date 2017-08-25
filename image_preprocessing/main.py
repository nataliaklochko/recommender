# -- coding: utf-8 --

import os
import data_base
import vgg16_modeling
from tqdm import tqdm


class Main:

	def __init__(self):
		self.current_dir = os.path.dirname(os.path.abspath(__file__))
		self.data_set_dir = os.path.join(self.current_dir, '..', 'static')
		self.n_clusters = 10
		self.db = data_base.DataBase(number=self.n_clusters)

	def main(self):
		print('Database creating...')
		db = data_base.DataBase(number=self.n_clusters)
		# db.create_image_table()
		# for img in tqdm(os.listdir(self.data_set_dir)):
		# 	self.db.insert_image_entry(image_name=img)
		# print('Image table was created')

		print('Model creating...')
		new_model = vgg16_modeling.NewModel(image_set_dir=self.data_set_dir,
											database=db, n_clusters=self.n_clusters)
		new_model.build_model_pca()
		# new_model.build_model_cluster()
		print('Model was created')


if __name__ == '__main__':
	app = Main()
	app.main()
