from django.db import models


class Image(models.Model):
	name = models.CharField(max_length=128, unique=True)
	pca_features = models.BinaryField()
	cluster = models.IntegerField()

	def __unicode__(self):
		return self.name

