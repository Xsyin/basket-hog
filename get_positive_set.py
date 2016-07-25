import cv2
import os

path = '/home/xsyin/videos/'
# rectangle formate is (x, y, w, h)
rects = {'01.avi': (883, 79, 80, 80), '01.mov': (505, 270, 48, 48), '02.avi': (951, 219, 74, 74), '02.mov': (475, 291, 42, 42), '03.avi': (
    664, 90, 71, 71), '03.mov': (472, 292, 40, 40), '04.avi': (1196, 246, 72, 72), '04.mov': (473, 291, 43, 43)}

# step is the gap between tow frames
step = 100
for file in os.listdir(path):
    video_path = path + file
    file = file.lower()
    if file not in rects:
        continue
    print "Now is cutting image from " + file
    rect = rects[file]

    print video_path
    capture = cv2.VideoCapture(video_path)
    frame_count = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    i = 0
    while capture.isOpened():
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
        ret, frame = capture.read()

        hoop = cv2.resize(
            frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]], (40, 40))

        cv2.imwrite(('positive/%s-%d.jpg') % (file, i / step), hoop)

        # if i/step < 20:
        #     cv2.imwrite(('positive/%s-%d.jpg') % (file, i / step), hoop)
        # else:
        #     cv2.imwrite(('test/%s-%d.jpg') % (file, i / step), hoop)
        if i <= frame_count - step:
            i += step
        else:
            break

print 'end!'
capture.release()
