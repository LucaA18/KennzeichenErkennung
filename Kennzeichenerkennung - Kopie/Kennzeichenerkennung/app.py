from flask import Flask, render_template, request, redirect, url_for
import os
from plate import plate_detection
from plate import country_eu_detection
import cv2
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

model1 = plate_detection
model2 = country_eu_detection

class Video:
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label
        
def get_existing_videos():
    videos = []
    static_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(static_path):
        if filename.endswith('.mp4'):
            videos.append(Video(filename, ''))
    return videos

videos = get_existing_videos()  # Initialize videos list with existing files


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        label = request.form['label']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            videos.append(Video(filename, label))
    return render_template('index.html', videos=videos)

@app.route('/delete/<filename>', methods=['GET'])
def delete_video(filename):
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    global videos
    videos = get_existing_videos()  # Update videos list after deletion
    return redirect(url_for('index'))

@app.route('/run_model/<video>', methods=['GET'])
def run_model(video):
    plate_to_detect = "WSTBX1"
    video_path = 'static/' + video
    print('videopath is ', video_path)
    # Here you can implement your code to run object detection model on the selected video
    # You can pass the video filename to your model and process it accordingly

    found, found_image, frame_number, number_plate, country = plate_detection.inference(plate_to_detect, video_path)

    if found:
        cv2.imwrite("static/images/screenshot_foundPLate.jpg", found_image)
    else:
        return render_template('run_model.html', video=video, frame_number=frame_number, found=False)


    print('found?:', found)
    print(found_image)
    print(frame_number)
    return render_template('run_model.html', video=video, frame_number=frame_number, country=country, found=True, number_plate=number_plate)


if __name__ == '__main__':
    app.run(debug=True)
