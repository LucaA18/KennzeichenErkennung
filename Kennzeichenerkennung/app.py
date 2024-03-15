from flask import Flask, render_template, request, redirect, url_for
import os
from plate import plate_detection
from plate.licence_and_country import LicenceCountryDetection

import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# model1 = plate_detection
# model2 = country_eu_detection

class Video:
    def __init__(self, filename):
        self.filename = filename
        
def get_existing_videos():
    videos = []
    static_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(static_path):
        if filename.endswith('.mp4'):
            videos.append(Video(filename))
    return videos

videos = get_existing_videos()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            videos.append(Video(filename))
    return render_template('index.html', videos=videos)

@app.route('/delete/<filename>', methods=['GET'])
def delete_video(filename):
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    global videos
    videos = get_existing_videos()
    return redirect(url_for('index'))

@app.route('/run_model/<video>', methods=['GET'])
def run_model(video):
    input_numberPlate = request.args.get('number_plate')
    print('das hier wird übergeben', input_numberPlate)

    input_model_selected = request.args.get('select_model')

    plate_to_detect = input_numberPlate
    video_path = 'static/' + video
    print('videopath is ', video_path)
    result = ''


    if video_path.rsplit('.', 1)[1].lower() == 'jpg' or video_path.rsplit('.', 1)[1].lower() == 'png':
        lcd = LicenceCountryDetection(search_text_img=input_numberPlate, img=video_path, mode_anpr=input_model_selected, device='cpu')
        result, found_image = lcd.main()
        result +='<br>'
    if video_path.rsplit('.', 1)[1].lower() =='mp4':
        lcd = LicenceCountryDetection(search_text=input_numberPlate, vid=video_path, mode_anpr=input_model_selected, device='cpu')
        result, found_image = lcd.main()
        result +='<br>'

    if (result.startswith('Neg')):
        found = False
    else:
        found = True
    
    # found, found_image, frame_number, number_plate, country = plate_detection.inference(plate_to_detect, video_path)

    if found:
        cv2.imwrite("static/images/screenshot_foundPLate.jpg", found_image)
    else:
        return render_template('run_model.html', video=video, found=False, result=result)

    return render_template('run_model.html', video=video, found=True, result=result)


if __name__ == '__main__':
    app.run(debug=True)
