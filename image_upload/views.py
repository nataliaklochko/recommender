from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from image_upload.new_image_processing import ImageProcessing
from image_upload.models import Image
FORMATS = ['jpg', 'png', 'jpeg']


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        im_proc = ImageProcessing()
        sim_images = im_proc.main(filename)
        sim_image_objects = []
        for name in sim_images:
            sim_image_objects.append(Image.objects.get(name=name))
        message_1 = 'Выбранное изображение'
        message_2 = 'Похожие товары'
        return render(request, 'image_upload/index.html', {
            'uploaded_file_url': uploaded_file_url,

            'myfile': myfile,
            'sim_image_objects': sim_image_objects,
            'message_1': message_1,
            'message_2': message_2,
        })
    return render(request, 'image_upload/index.html')


