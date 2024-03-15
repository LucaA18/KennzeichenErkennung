import cv2
import numpy as np
import easyocr
import re
import torchvision
import torch
from torchvision import transforms

from PIL import Image
from ultralytics import YOLO

class LicenceCountryDetection:
    def __init__(self, search_text=None, vid=None, search_text_img=None, img=None, mode_anpr='easyocr', device='cpu'):
        self.vid = vid
        self.img = img
        self.search_text_img = search_text_img
        self.search_text = search_text
        self.mode_anpr = mode_anpr
       
        
        print(torch.cuda.is_available())

        self.model_yolo_ocr = YOLO('plate/best.pt')
        self.model_yolo_ocr.to(device)

        self.model_yolo_plate = YOLO('plate/best_plate.pt')
        self.model_yolo_plate.to(device)

        
        self.device = device

        # model for easyocr        
        if self.device=='cuda:0':
            gpu = True
        else:
            gpu = False

        self.reader = easyocr.Reader(['en'], gpu=gpu)


        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT 
        model2 = torchvision.models.mobilenet_v2(weights=weights).to(device)
        
        output_shape = 11


        model2.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, 
                        bias=True))
        model2.load_state_dict(torch.load('plate/licence_class_v3.pth', map_location=torch.device(device)))
        self.transform_eu = transforms.Compose([
        transforms.Resize(232),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        ])
        model2.to(device)
        self.model2 = model2


        self.model2.eval()
        
        if self.vid!=None:
            if not vid.endswith('.mp4'):
                return 'Your given video is not a video format'
                
        if self.img!=None:
            if not (self.img.endswith('.jpg') or self.img.endswith('.png')):
                return 'Your given image is not a image format'
            
                
    
    def easyocr_license_plate(self, license_plate_crop, search_text):
            gray_image = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

            background = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, structuring_element)

            out_gray = cv2.divide(gray_image, background, scale=255) 
            
            result = self.reader.readtext(np.asarray(out_gray))

            text_final = ''

            if len(result) == 1:
                text = result[0][1]
                text = re.sub(r'[^A-Za-z0-9]+', '', text)
                text = text.strip()

                if text==search_text:
                    return True
                
            if len(result) > 1:
                for res in result: 
                    if res[2] > 0.3:
                        text = res[1]
                        text = re.sub(r'[^A-Za-z0-9]+', '', text)
                        text_final += text
                        text_final = text_final.strip()

                        if text_final==search_text or text==search_text:
                            return True
                        text_final += ''
            return False


    def anpr_licence_plate(self, image, len_text):
        image = [image]
        with torch.inference_mode():
            output = self.model_yolo_ocr(image)[0]
        xmins = []
        labels = []
        scores = []
        for license_plate in output.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            labels.append(self.model_yolo_ocr.names[class_id])
            xmins.append(x1)
            scores.append(score)

        predictions = list(zip(xmins, labels, scores))
        sorted_predictions = sorted(predictions, key=lambda x:x[0])
        
        final_str = ''
        for i in sorted_predictions:
            if i[1]=='EUR':
                continue
            final_str += i[1]
        print(final_str)
        return final_str



    def detect_objects(self, frame):
        
        frame=[frame]
        with torch.inference_mode():
            license_plates = self.model_yolo_plate(frame)[0]

        bboxs = []
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            bboxs.append([int(x1), int(y1), int(x2), int(y2)])
        
        return bboxs


    def eu_detection(self, img, model, transform):
        height, width = img.shape[:2]
        img = img[0:height, 0:width//4]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        
        img = transform(img).to(self.device)

        with torch.inference_mode():
            output = model(img.unsqueeze(0))
    
        class_map = {
            0: 'B',
            1: 'BG',
            2: 'D',
            3: 'E',
            4: 'F',
            5: 'GR',
            6: 'HR',
            7: 'I',
            8: 'NL',
            9: 'P',
            10: 'PL'
        }

        output_idx = np.argmax(output.cpu()).item()
        country = class_map[output_idx]
        
        return country


    def plot_enlarged_object(self, frame, box, i=0):
        
        enlarged_box = np.copy(box)
        enlarged_box[0] -= (0 + i*5)
        enlarged_box[1] -= (0 + i*5)  
        enlarged_box[2] += (0 + i*5)  
        enlarged_box[3] += (0 + i*5) 

        enlarged_box[0] = max(enlarged_box[0], 0)
        enlarged_box[1] = max(enlarged_box[1], 0)
        enlarged_box[2] = min(enlarged_box[2], frame.shape[1])
        enlarged_box[3] = min(enlarged_box[3], frame.shape[0])

        enlarged_object = frame[enlarged_box[1]:enlarged_box[3], enlarged_box[0]:enlarged_box[2]]
        
        interpolation = cv2.INTER_LINEAR if 640 > enlarged_object.shape[1] else cv2.INTER_AREA

        enlarged_object = cv2.resize(enlarged_object, (640,320), interpolation=interpolation)

        return enlarged_object

    def main(self):
        if self.vid!=None:
            
            if self.mode_anpr == 'both':
                print('We wont recommend you this. This would result in immense speed drops even more using a cpu.')
                answer_continue = input('Would you like to continue?')
                if answer_continue.lower() == 'yes' or answer_continue.lower() == 'y':
                    pass
                else:
                    return print('Process stopped')
            
            video_capture = cv2.VideoCapture(self.vid)

            counter = 0

            enlarged_frames = []
            found = False
            search_licence = False
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                bboxes_preds = self.detect_objects(frame)

                for i in range(1):
                    if len(bboxes_preds) > 0:
                        for boxes in bboxes_preds:
                            enlarged_object = self.plot_enlarged_object(frame, boxes, i)
                            cv2.imshow('en', enlarged_object)

                            if not found:
                                if self.mode_anpr == 'easyocr' or self.mode_anpr == 'both':
                                    search_licence = self.easyocr_license_plate(enlarged_object, self.search_text)
                                    if search_licence:
                                        found=True

                                if self.mode_anpr != 'easyocr':
                                    final_string = self.anpr_licence_plate(enlarged_object, len(self.search_text))
                                    print(final_string)
                                    if final_string==self.search_text:
                                                found=True
                                        
                            if found:
                                counter+=1
                                enlarged_frames.append(enlarged_object)
                                if counter % 4 == 0:
                                    country_list = []
                                    for frame in enlarged_frames:
                                        country = self.eu_detection(frame, self.model2, self.transform_eu)
                                        country_list.append(country)
                                        
                                    unique_elements, counts = np.unique(country_list, return_counts=True)
                                    label_eu = unique_elements[np.argmax(counts)]

                                    final_string = ''
                                    final_string += f'The country of Licence: {label_eu}<br>'
                                    final_string += f'The Licence plate:{self.search_text}<br>'
                                    position_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)

                                    current_time_sec = position_ms / 1000

                                    minutes = int(current_time_sec // 60)
                                    seconds = int(current_time_sec % 60)
                                    final_string += f"Aktuelle Zeit: {minutes:02d}:{seconds:02d}"

                                    break
                                                        
                        for box in bboxes_preds:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                        

                cv2.imshow('Object Detection', frame)

                if (cv2.waitKey(1) & 0xFF == ord('q')) or (counter==4):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
            return final_string, frame
        
        if self.img!=None:
            img = cv2.imread(self.img)

            bboxes_preds = self.detect_objects(img)

            search_licence = False

            final_string = ''
            countries=[]
            for i in range(3):
                for boxes in bboxes_preds:
                    enlarged_object = self.plot_enlarged_object(img, boxes, i)
                
                    if self.mode_anpr == 'easyocr' or self.mode_anpr == 'both':
                        search_licence = self.easyocr_license_plate(enlarged_object, self.search_text_img)

                    if self.mode_anpr != 'easyocr':
                        licence_str = self.anpr_licence_plate(enlarged_object, 0)
                        if licence_str == self.search_text_img:
                            search_licence = True
                            country = self.eu_detection(enlarged_object, self.model2, self.transform_eu)
                            countries.append(country)
            if search_licence:
            
                final_string += 'Positive<br>'
                final_string += f'The given Licence Plate {self.search_text_img} was found in the image.<br>'
                final_string += f'The country of the Licence Plate is: {country}'
           
            else:
                final_string += 'Negative<br>'
                final_string += "The model couldn't find the licence plate In the image<br>"
                if self.mode_anpr!='easyocr':
                    final_string += f"The model found following licence plate: {licence_str}"
            
            return final_string, img

if __name__ == '__main__':
    vid = "..\Sportcars leaving a Carshow WILD _ SupercarMadness 2023.mp4"
    img = r'c:\Users\moakg\OneDrive\Documents\Kennzeichenerkennung\P05786.jpg' 

    search_text_vid = 'K455GG'
    search_text_img = ''
    
    mode_anpr = 'anpr' # either use 'easyocr'  or 'multi' else anpr model will be used
    conf_tresh_detect = 0.25

    device='cuda:0'
    lcd = LicenceCountryDetection(search_text=search_text_vid, img=img, vid=vid, search_text_img=search_text_img, mode_anpr=mode_anpr, device=device)
    lcd.main()