import numpy as np 
import pandas as pd 
import os 
from tqdm import tqdm 
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import cv2 

df = pd.read_csv('../data/train_folds.csv')
df['vid'] = df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
df['step'] = df['contact_id'].apply(lambda x: int(x.split('_')[2]))
df['pair'] = df['vid'] + '_' + df['nfl_player_id_1'].astype(str) + '_' + df['nfl_player_id_2'].astype(str)

df['frame'] = df['frame'] #+6

det_dict = np.load('../data/det_dict.npy', allow_pickle=True).item()

df = df[df.nfl_player_id_2 != 'G']
df = df.reset_index(drop=True)         
print(df.shape)

df = df[df.distance<1.7]
print(df.shape)

vids = df.vid.unique()
print(len(vids))


def get_sample(vid, window_size = 20, out_size = 128, stride=4):
    ws = []
    hs = []
    for fr in range(fr_id + window_size[0], fr_id + window_size[-1] + 1):
        if fr in det_dict[vid]:
            if idx1 in det_dict[vid][fr] and idx2 in det_dict[vid][fr]: 
                x, y, w, h = det_dict[vid][fr][idx1]['box']
                ws.append(w)
                hs.append(h)
                x, y, w, h = det_dict[vid][fr][idx2]['box']
                ws.append(w)
                hs.append(h)

    if len(ws)>0:
        crop_size = int(5*max(np.mean(ws), np.mean(hs)))
    else:
        crop_size = out_size

    bboxes = []
    for fr in range(fr_id + window_size[0], fr_id + window_size[-1] + 1):
        if fr in det_dict[vid]:
            if idx1 in det_dict[vid][fr] and idx2 in det_dict[vid][fr]: 
                x, y, w, h = det_dict[vid][fr][idx1]['box']
                x1 = x + w/2
                y1 = y + h/2

                x, y, w, h = det_dict[vid][fr][idx2]['box']
                x2 = x + w/2
                y2 = y + h/2

                xc = 0.5*x1 + 0.5*x2
                yc = 0.5*y1 + 0.5*y2

                bboxes.append([xc-crop_size, yc-crop_size, xc+crop_size, yc+crop_size])
            else:
                bboxes.append([np.nan, np.nan, np.nan, np.nan])
        else:
            bboxes.append([np.nan, np.nan, np.nan, np.nan])


    bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values

    images = []
    masks1 = []
    masks2 = []
    empty_count = 0
    for i, ii in enumerate(window_size):
        if bboxes.sum() > 0:
            fr = ii + fr_id
            path = f'{vid}_{fr}'

            if path in image_dict:
                image = image_dict[path]
            else:
                image = np.zeros((720, 1280,3), dtype = np.uint8)
                empty_count +=1


            mask1 = np.zeros((720, 1280), dtype = np.uint8)
            mask2 = np.zeros((720, 1280), dtype = np.uint8)

            x1, y1, x2, y2 = list(map(int, bboxes[ii-window_size[0]]))

            y1 = y1 + int(0.2*crop_size)
            x2 = x1 + crop_size*2
            y2 = y1 + crop_size*2

            if fr in det_dict[vid]:
                if idx1 in det_dict[vid][fr]: 
                    x, y, w, h = det_dict[vid][fr][idx1]['box']
                    # cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
                    # mask1[y:y+h, x:x+w] = 255
                    cv2.circle(mask1, (x+w//2, y+h//2), int(0.25*h+0.25*w), 255, thickness=-1)

                if idx2 in det_dict[vid][fr]:
                    x, y, w, h = det_dict[vid][fr][idx2]['box']
                    # cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,255), 2)
                    # mask2[y:y+h, x:x+w] = 255
                    cv2.circle(mask2, (x+w//2, y+h//2), int(0.25*h+0.25*w), 255, thickness=-1)

            crop = image[y1:y2, x1:x2]
            crop_mask1 = mask1[y1:y2, x1:x2]
            crop_mask2 = mask2[y1:y2, x1:x2]

            cr_y, cr_x = crop.shape[:2]
            if cr_x == crop_size*2 and cr_y == crop_size*2:
                crop = cv2.resize(crop, (out_size*2,out_size*2))
                crop_mask1 = cv2.resize(crop_mask1, (out_size*2,out_size*2))
                crop_mask2 = cv2.resize(crop_mask2, (out_size*2,out_size*2))
                images.append(crop)
                masks1.append(crop_mask1)
                masks2.append(crop_mask2)
            else:
                tmp_crop =  np.zeros((crop_size*2, crop_size*2,3), dtype = np.uint8)
                tmp_mask1 =  np.zeros((crop_size*2, crop_size*2), dtype = np.uint8)
                tmp_mask2 =  np.zeros((crop_size*2, crop_size*2), dtype = np.uint8)
                if x1 < 0:
                    if y2>=720:
                        tmp_crop[crop_size*2-cr_y:,:cr_x] = crop
                        tmp_mask1[crop_size*2-cr_y:,:cr_x] = crop_mask1
                        tmp_mask2[crop_size*2-cr_y:,:cr_x] = crop_mask2
                    else:
                        tmp_crop[:cr_y,:cr_x] = crop
                        tmp_mask1[:cr_y,:cr_x] = crop_mask1
                        tmp_mask2[:cr_y,:cr_x] = crop_mask2

                elif x2> 1280:
                    if y2>=720:
                        tmp_crop[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop
                        tmp_mask1[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop_mask1
                        tmp_mask2[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop_mask2
                    else:
                        tmp_crop[:cr_y,crop_size*2-cr_x:] = crop
                        tmp_mask1[:cr_y,crop_size*2-cr_x:] = crop_mask1
                        tmp_mask2[:cr_y,crop_size*2-cr_x:] = crop_mask2

                tmp_crop = cv2.resize(tmp_crop, (out_size*2,out_size*2))
                tmp_mask1 = cv2.resize(tmp_mask1, (out_size*2,out_size*2))
                tmp_mask2 = cv2.resize(tmp_mask2, (out_size*2,out_size*2))
                images.append(tmp_crop)
                masks1.append(tmp_mask1)
                masks2.append(tmp_mask2)
        else:
            empty_count +=1
            # crop =  np.zeros((crop_size*2, crop_size*2,3))
            crop =  np.zeros((out_size*2, out_size*2,3), dtype = np.uint8)
            crop_mask1 =  np.zeros((out_size*2, out_size*2), dtype = np.uint8)
            crop_mask2 =  np.zeros((out_size*2, out_size*2), dtype = np.uint8)
            images.append(crop)
            masks1.append(crop_mask1)
            masks2.append(crop_mask2)
    return images, masks1, masks2, empty_count

os.makedirs('cache/cache15/', exist_ok=True)
results = []

window_size = [-44, -37, -30, -24, -18, -13, -8, -4, -2, 0, 2, 4, 8, 13, 18, 24, 30, 37]
window_size = [x+6 for x in window_size]

vids = df.vid.unique()
for vid in tqdm(vids):
    e_vid = vid + f'_Endzone'
    s_vid = vid + f'_Sideline'

    e_vid_path = f'../data/train/{e_vid}.mp4'
    s_vid_path = f'../data/train/{s_vid}.mp4'
    
    image_dict = {}
    
    cap = cv2.VideoCapture(e_vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,250)
    frame_count = 250
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        kk = f'{e_vid}_{frame_count}'
        image_dict[kk] = frame
        frame_count +=1

    cap = cv2.VideoCapture(s_vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
    frame_count = 250
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        kk = f'{s_vid}_{frame_count}'
        image_dict[kk] = frame
        frame_count +=1


    df1 = df[df['vid']==vid].reset_index(drop=True)
    for i, row in tqdm(df1.iterrows()):
        idx1 = int(row['nfl_player_id_1'])
        idx2 = int(row['nfl_player_id_2'])
        fr_id = int(row['frame'])
        step = int(row['step'])
        e_images, e_masks1, e_masks2, e_empty_count = get_sample(e_vid, window_size = window_size, stride=4)
        s_images, s_masks1, s_masks2, s_empty_count = get_sample(s_vid, window_size = window_size, stride=4, out_size = 128)

        #visualize
        os.makedirs(f'draw/cache15/{idx1}_{idx2}_{fr_id}', exist_ok=True)
        # for ii, (e_img, s_img, e_m1, e_m2, s_m1, s_m2) in enumerate(zip(e_images, s_images, e_masks1, e_masks2, s_masks1, s_masks2)):
        for ii in range(len(e_images)):
            e_img = e_images[ii]
            e_img[e_masks1[ii]>100] = 255
            e_img[e_masks2[ii]>100] = 0

            s_img = s_images[ii]
            s_img[s_masks1[ii]>100] = 255
            s_img[s_masks2[ii]>100] = 0

            e_img = cv2.resize(e_img, (256,256))
            s_img = cv2.resize(s_img, (256,256))
            e_img = np.hstack([e_img, s_img])
            cv2.imwrite(f'draw/cache15/{idx1}_{idx2}_{fr_id}/{ii}.jpg', e_img)


#         path = f'cache/cache15/{vid}_{idx1}_{idx2}_{fr_id:04d}_{step}'
#         item = {'path': path, 'fold':row['fold'], 'contact': row['contact'], 'distance': row['distance'], 'step': step, 'e_empty': e_empty_count, 's_empty': s_empty_count}

#         results.append(item)

#         images_e = []
#         images_s = []
#         for ii in range(len(e_images)):
#             e_img = e_images[ii]
#             e_img[e_masks1[ii]>100] = 255
#             e_img[e_masks2[ii]>100] = 0

#             s_img = s_images[ii]
#             s_img[s_masks1[ii]>100] = 255
#             s_img[s_masks2[ii]>100] = 0

#             images_e.append(e_img)
#             images_s.append(s_img)


#         np.save(f'{path}_e.npy', np.array(images_e))
#         np.save(f'{path}_s.npy', np.array(images_s))

# #     # break

# df = pd.DataFrame(results)
# print(df.shape)
# print(df.head())

# df.to_csv('train_cache15.csv', index=False)
