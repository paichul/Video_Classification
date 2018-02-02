import csv
import glob
import os
import numpy as np
import os.path
from subprocess import call
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

classes = ['2', '29', '26', '43', '30', '36', '37', '44', '23', '15', '10', '39', '22',
           '24', '28', '17', '27', '1', '42', '41', '34', '35', '20', '25', '33', '19']

FRAME_SAMPLE = 5
FRAME_SAMPLE1 = FRAME_SAMPLE-1

def extract_frames_uniform(clip, duration, output_path):
     clip.save_frame(output_path+"_0.png", t=10)
     df = duration/FRAME_SAMPLE1
     for i in range(1, FRAME_SAMPLE1):
          clip.save_frame(output_path+"_"+str(i)+".png", t=i*df)
     clip.save_frame(output_path+"_"+str(FRAME_SAMPLE1)+".png", t=(duration-5))
         
def get_uniform_frames_by_group(group_dir, output_dir, samplePerClass):
     count = {}
     category_ids = os.listdir(group_dir)
     
     for category_id in classes:
          input_folder = group_dir+str(category_id)+"/"
          output_folder = output_dir+str(category_id)+"/"

          if not os.path.exists(output_folder):
               os.makedirs(output_folder)
          
          video_names = os.listdir(input_folder)
          image_count = len(glob.glob(output_folder + '*.png'))
          
          if image_count >= samplePerClass:
               print("skipping " + input_folder, image_count)
               count[category_id] = image_count
               continue

          count[category_id] = 0
          print("processing " + input_folder)

          for video_name in video_names:
               cid = count[category_id]
               if cid >= samplePerClass:
                    break
               
               video_path = input_folder + video_name
               clip = VideoFileClip(video_path)
               duration = clip.duration
               output_path = output_folder + video_name
               extract_frames_uniform(clip, duration, output_path)

               count[category_id] += FRAME_SAMPLE

     for key, val in count.items():
          print(key, val)

sampleN = 1000
classN = 27
samplePerClassN = sampleN*FRAME_SAMPLE

get_targeted_frames_by_group("./train/", "./uniform_data/train/", samplePerClassN)

sampleN = 100
classN = 27
samplePerClassN = sampleN*FRAME_SAMPLE

get_targeted_frames_by_group("./test/", "./uniform_data/test/", samplePerClassN)
