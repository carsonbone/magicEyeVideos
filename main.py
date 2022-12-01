import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import rgba2rgb
from PIL import Image
plt.rcParams['figure.dpi'] = 150

#testing function that just displays an image
def display(img):
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 2:
        i = skimage.io.imshow(img, cmap='gray')
    else:
        i = skimage.io.imshow(img)
    plt.tight_layout()
    plt.show()

#generates a pattern for our magic eye image
def make_pattern(shape=(200, 200), levels=64):
    return np.random.randint(0, levels - 1, shape) / levels


# generates single magic eye image
def make_autostereogram(depthmap, pattern, shift_amplitude=0.1,):

    if depthmap.max() > depthmap.min():
        depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())

    autostereogram = np.zeros_like(depthmap, dtype=pattern.dtype)
    temp = np.arange(autostereogram.shape[0])
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram



capture = cv2.VideoCapture('files/input.mp4')

cur_Frame = 0
collection = []
vidInfo = [int(capture.get(3)),int(capture.get(4)),int(capture.get(5))]
#the above variable holds the width, height, and fps for our output
while (True):
    success,frame = capture.read()

    if success:
        collection.append(frame)
    else:
        break
        #program reaches here once all frames have been read
    cur_Frame = cur_Frame+1

capture.release()

i = 0
magiceyes = []

#generates a gif for our output
# def make_gif(frame_folder):
#     frames = [Image.open(image) for image in glob.glob(f"files/{frame_folder}/*.jpg")]
#     frame_one = frames[0]
#     frame_one.save("output_gif.gif", format="GIF", append_images=frames,
#                    save_all=True, duration=33, loop=0)

#generates a video for our output
def make_vid(frame_folder):
    #grabs our images, sort by creation time
    temparr = sorted(glob.glob(f"files/{frame_folder}/*.jpg"), key=os.path.getmtime)
    frames = [Image.open(image) for image in temparr]
    #using a motion jpg codec
    writer = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"MJPG"), vidInfo[2],
                             (vidInfo[0],vidInfo[1]), isColor=0)
    for frame in frames:
        tempframe = np.asarray(frame)
        writer.write(tempframe)
    writer.release()
    #deletes the images created, as they have now served their purpose
    files = glob.glob(f"files/{frame_folder}/*.jpg")
    for f in files:
        os.remove(f)

#iterates over each frame, generating a magic eye image
while(i < len(collection)):
    current = skimage.color.rgb2gray(collection[i])
    new_tuple = (int(round(current.shape[0] / 8)), int(round(current.shape[1] / 8)))
    temppattern = make_pattern(shape=new_tuple)
    temp = make_autostereogram(current, temppattern)
    magiceyes.append(temp)
    temp2 = temp*255
    cv2.imwrite(f'files/output1/frame_' + str(i) + '.jpg',temp2)
    print(i)

    i = i+1

#make_gif("output1")
make_vid("output1")
cv2.destroyAllWindows()



