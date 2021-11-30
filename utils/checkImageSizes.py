


from PIL import Image
import os





path = '/home/access/dev/Holopix50k/train/left'

files = os.listdir(path)
min_h = 20000
min_w = 20000
for i in range(len(files)):

    file_name = os.path.join(path, files[i])
    image = Image.open(file_name)
    h, w = image.size
    if h < min_h:
        print('h',h)
        min_h = h
    if w < min_w:
        print('w', w)
        min_w = w

print(min_h, min_w)



print('done')