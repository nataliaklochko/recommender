import sqlite3
from tqdm import tqdm

init_db = 'image_db_100.db'
new_db = 'db.sqlite3'

init_conn = sqlite3.connect(init_db)
init_c = init_conn.cursor()

new_conn = sqlite3.connect(new_db)
new_c = new_conn.cursor()

# imageshow_image table
# 3 clusters, cos metric

init_c.execute('SELECT name, pca_features, cluster_cos FROM image_set')
data = init_c.fetchall()


for i, row in tqdm(enumerate(data)):
	new_link = "http://inhome360.ru/catalog/show/" + row[0].split('_')[2]
	# print(new_link)
	new_c.execute('INSERT INTO image_upload_image VALUES (?,?,?,?,?)', (i, row[0], row[1], row[2], new_link))
	new_conn.commit()

init_conn.close()
new_conn.close()