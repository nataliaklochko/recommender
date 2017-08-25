import sqlite3
import os


class DataBase:

    def __init__(self, number):
        self.db_name = os.path.join('..', 'db.sqlite3')
        self.conn = sqlite3.connect(self.db_name)
        self.c = self.conn.cursor()

    def create_image_table(self):
        self.c.execute('CREATE TABLE IF NOT EXISTS image_upload_image\
         (ID INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, pca_features BLOB, cluster INTEGER)')


    def insert_image_entry(self, image_name):
        self.c.execute('INSERT INTO image_upload_image (name) VALUES (?)', (image_name,))
        self.conn.commit()

    def close_conn(self):
        self.c.close()
        self.conn.close()


if __name__ == '__main__':
    print('Database')

