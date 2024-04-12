import numpy as np
import cv2
import keras
import albumentations as albu

class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, label_filenames, batch_size, black_white = True, augmentations = True):
        self.image_filenames = image_filenames    
        self.label_filenames = label_filenames
        self.batch_size = batch_size
        self.bw = black_white
        self.aug = augmentations

        # Defining the augmentations
        self.aug_train = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.Affine(scale=(0.5, 1.5), translate_percent=(-.125,.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True)
        ])

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]


        x = []
        y = []
        for x_file, y_file in zip(batch_x, batch_y):
            temp1 = cv2.imread(x_file)
            temp2 = cv2.imread(y_file)
            temp2 = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)
            if self.bw:
                temp1 = cv2.cvtColor(temp1, cv2.COLOR_BGR2GRAY)
            if self.aug:
                ug = self.aug_train(image=temp1, mask=temp2)
                temp1 = ug['image'] / 255
                temp2 = ug['mask'] / 255
            x.append(temp1)
            y.append(temp2)

        return np.array(x), np.array(y)

def generate_gt_img(img, nPct = 5, cCount = 100):
    """Returns a mask from a black and white image
    
    img     -- image to process
    nPct    -- n percent of top bright pixels to be highlighted(5% default)
    cCount  -- size of minimum group taggged as noise for removal(100 default)
    """
    blurred = cv2.GaussianBlur(img[:,:], (5,5), 1)

    size = blurred.shape[0] * blurred.shape[1] # length * width
    brightness_pct = nPct / 100.0

    flat_blur = blurred.flatten()

    perc_idx = int(size * brightness_pct)

    brightness_thresh = np.argpartition(flat_blur, -perc_idx)[-perc_idx:]   # picking out the top pct
    brightness_thresh = brightness_thresh[np.argmin(flat_blur[brightness_thresh])]
    brightness_thresh = flat_blur[brightness_thresh]
    
    blurred[blurred < brightness_thresh] = 0
    blurred[blurred > 0] = 255  # Non-Maximal Suppression
    
    # Small noise removal technique
    noise_removal_threshold = 100
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(blurred)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)
    
    return mask

def shatter_img(in_img, size=352):
    
    img_len = len(in_img)
    img_wid = len(in_img[0])
    # pad our image beforehand to make it easier for overlaying
    bottom_pad = size - img_len % size
    right_pad = size - img_wid % size
    img = cv2.copyMakeBorder(in_img, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    img_len = len(img)
    img_wid = len(img[0])
    image_grid = []
    for i in range(size, img_len + 1, size):
        temp = []
        for j in range(size, img_wid + 1, size):
            temp.append(img[i-size:i,j-size:j])
        image_grid.append(temp)
    return image_grid

def stitch_img_grid(img_grid):
    image_rows = []
    for row in range(len(img_grid)):
        image_rows.append(np.concatenate(img_grid[row], axis=1))
    
    out_img = np.concatenate(image_rows, axis=0)
    
    return out_img

def align_datum(image_path = '', mask_path = '', color = False, size=512, ):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    if not color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    images = shatter_img(image, size=size)
    masks = shatter_img(mask, size=size)

    aligned_lists = {'path':[], 'i':[], 'j':[], 'data':[], 'mask':[]}
    for i in range(len(images)):
        for j in range(len(images[0])):
            aligned_lists['path'].append(image_path)
            aligned_lists['i'].append(i)
            aligned_lists['j'].append(j)
            aligned_lists['data'].append(images[i][j] / 255)
            aligned_lists['mask'].append(masks[i][j] / 255)

    return aligned_lists