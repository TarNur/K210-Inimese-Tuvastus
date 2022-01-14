import xml.etree.ElementTree as ET
import os


path = '/Users/tarvonurmetu/Downloads/pascal_20_detection_proov/anns_validation'
images_path = '/Users/tarvonurmetu/Downloads/pascal_20_detection_proov/imgs_validation'


for filename in os.listdir(path):
   noPeople = 0
   if not filename.endswith('.xml'): continue
   fullname = os.path.join(path, filename)
   tree = ET.parse(fullname)
   root = tree.getroot()
   for objects in root.findall('object'):
      objectname = objects.find('name').text
      if objectname == 'person':
         noPeople = 1
      else:
         root.remove(objects)
   if noPeople == 0:
      print(filename, ' pole inimesi')
      os.remove(fullname)
      os.remove(images_path+"/"+filename[:-4]+".jpg")
   else:
      tree.write(fullname)