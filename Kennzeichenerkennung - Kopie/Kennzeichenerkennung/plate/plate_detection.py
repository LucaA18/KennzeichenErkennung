from ultralytics import YOLO
import cv2

# import util
from plate.sort.sort import *
import easyocr
import re
# from util import get_car, read_license_plate, write_csv

from plate.country_eu_detection import eu_detection


def inference(search_plate, video_path):
    results = {}

    mot_tracker = Sort()

    # load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('plate/best_plate.pt')

    # load video
    cap = cv2.VideoCapture(video_path)

    vehicles = [2, 3, 5, 7]

    # read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                    
                    # print(license_plate_text)
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        
                    if license_plate_text and str(search_plate) in license_plate_text:
                        print("found " + search_plate)
                        # write_csv(results, './test.csv')
                        resulting_dict = eu_detection(license_plate_crop)

                        temp = search_plate + ' ' + resulting_dict
                        return True, frame, frame_nmr, search_plate, resulting_dict

    return False, None, None, None, None
    # write results
    # write_csv(results, './test.csv')

def read_license_plate(license_plate_crop):
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        image=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image, bg, scale=255)
        out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
        # _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)

        # text = pytess(license_plate_crop)
        # cv2.imwrite('license_plate_gray.jpg', out_binary)
        result = reader.readtext(out_binary)
        for res in result:
            if len(result) == 1:
                text = res[1]
            
            if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
                text = res[1]
        text = re.sub(r'[^A-Za-z0-9]+', '', text)
        print(text)
        return text, None
    except Exception as e:
        print(e)
        return None, None
    
def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def write_csv(results, output_path):

    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

# inference("V96")