import os, tarfile
 
import os
from google.colab import files

def make_targz_one_by_one(output_filename, source_dir):
  tar = tarfile.open(output_filename,"w")
  for root,dir_name,files_list in os.walk(source_dir):
    for file in files_list:
      pathfile = os.path.join(root, file)
      tar.add(pathfile)
  tar.close()
 
  files.download(output_filename)
 
 
make_targz_one_by_one('neg_tar', '/content/drive/MyDrive/pytorch_resnet18_places365')
