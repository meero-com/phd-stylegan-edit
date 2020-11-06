import numpy as np
from PIL import Image
import cv2

# normalizing

# image loading

def load_images(filenames):
    # Images must all be of same shape.
    images = []
    for filename in filenames:
        temp_image = np.asarray(Image.open(filename).convert('RGB'))
        
        # Adjust channel dimension to work with torch.
        temp_image = np.transpose(temp_image, (2,0,1))
        images.append(temp_image)

    return np.array(images)




def save_image(image, save_path):
    image = np.transpose(image, (1,2,0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path)


    
'''
VIDEO TOOLS
'''

    
def images_to_video(images, save_path, image_size=1024, nb_pause=0):
    '''
    nb_pause: number of frames to pause first and last frames
    '''
    size = (image_size, image_size)
    fps = 15
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i, image in enumerate(images):
        # Channel, width, height -> width, height, channel, then RGB to BGR
        image = np.transpose(image, (1,2,0))
        image = image[:,:,::-1]
        
        video.write(image.astype(np.uint8))
        #cv2.imwrite( save_path.split('.')[0] + '.png', image.astype(np.uint8))
        
        # if we're on the first or last frame then add an extra pause if required
        if i == 0 or i == len(images) - 1:
            for j in range(nb_pause):
                video.write(image.astype(np.uint8))
                #cv2.imwrite( save_path.split('.')[0] + '.png', image.astype(np.uint8))
         
    video.release() 