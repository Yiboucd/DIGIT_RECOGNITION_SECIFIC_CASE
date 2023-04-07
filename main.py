import os.path
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import *

def screen_shot(video_path, save_path, desire_num):
    '''
    This function is used to sampling the video
    :param video_path: This is where your video is at
    :param save_path: The folder that you want to save the screenshot
    :param desire_num: number of sampling
    :return: screenshots of the video
    '''
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # frame per second
    print('FPS:{:.2f}'.format(fps))
    rate = cap.get(5)  # frame rate
    frame_num = cap.get(7)  # total frame number
    duration = frame_num / rate
    print('video total time:{:.2f}s', format(duration))

    interval = frame_num // desire_num
    process_num = frame_num // interval
    print('process frame:{:.0f}s', format(process_num))

    cnt = 0
    num = 0

    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cnt += 1
            if cnt % interval == 0:
                num += 1
                cv2.imwrite(save_path + "\\image_%d.jpg" % num, frame)
                remain_frame = process_num - num
                t1 = time.time() - t0
                t0 = time.time()
                print(
                    "Processing %d.jpg, remain frame: %d, remain time: %.2fs" % (num, remain_frame, remain_frame * t1))
        else:
            break
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Next Step: Predict")

def predict(input):
    dig = pytesseract.image_to_string(input, lang='eng', config=r'--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')
    if len(dig) == 0:
        dig = 7
        return str(dig)
    else:
        #dig = int(dig)
        return dig

def first_predict(input):
    dig = pytesseract.image_to_string(input, lang='eng', config=r'--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')
    if len(dig) == 0:
        dig = 0
        return str(dig)
    else:
        # dig = int(dig)
        return dig

def l2_norm(ground_truth, predict_value):
    gap = ground_truth - predict_value
    l2_norm = np.sum(np.power(gap, 2))
    return l2_norm

def l2_predict(sample_path, img ,l2_norm):
    '''
    G_T_x: The ground truth of number 'x'
    width of digit = 84
    start of the left digit: 440
    start of the middle digit: 549
    start of the right digit: 633
    '''
    loss_left = np.zeros((10, 1))
    loss_middle = np.zeros((10, 1))
    loss_right = np.zeros((10, 1))
    sample_folder = sample_path
    wid = 84
    left_start = 460
    middle_start = 549
    right_start = 633
    # Three digits segements (If the camera move, you could adjust the start point)
    img_left = img[120: 250, left_start: left_start + wid]
    img_middle = img[120: 250, middle_start: middle_start + wid]
    img_right = img[120: 250, right_start: right_start + wid]
     # Convert to the HSV color
    hsv_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2HSV)
    hsv_middle = cv2.cvtColor(img_middle, cv2.COLOR_BGR2HSV)
    hsv_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2HSV)
    # Get binary-mask
    lower_red = np.array([39, 0, 0])
    upper_red = np.array([216, 255, 255])
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

    msk_left = cv2.inRange(hsv_left, lower_red, upper_red)
    dlt_left = cv2.dilate(msk_left, krn, iterations=1)
    thr_left = 255 - cv2.bitwise_and(dlt_left, msk_left)

    msk_middle = cv2.inRange(hsv_middle, lower_red, upper_red)
    dlt_middle = cv2.dilate(msk_middle, krn, iterations=1)
    thr_middle = 255 - cv2.bitwise_and(dlt_middle, msk_middle)

    msk_right = cv2.inRange(hsv_right, lower_red, upper_red)
    dlt_right = cv2.dilate(msk_right, krn, iterations=1)
    thr_right = 255 - cv2.bitwise_and(dlt_right, msk_right)

    for x in range(0, 10):
        ground_truth = cv2.imread(sample_folder + "\\sample_%d.png" % x, -1)
        left_ground_truth = cv2.imread(sample_folder + "\\left_sample_%d.png" % x, -1)
        loss_left[x, 0] = l2_norm(thr_left, left_ground_truth)
        loss_middle[x, 0] = l2_norm(thr_middle, ground_truth)
        loss_right[x, 0] = l2_norm(thr_right, ground_truth)
        '''
        if np.min(loss) > 1000:
            digit_predict =
        '''

    '''
    for x in range(0, 2):
        ground_truth_left = cv2.imread(sample_folder_digit1 + "\\sample_%d.png" % x, -1)
        loss_left[x, 0] = l2_norm(thr_left, ground_truth_left)
    '''
    digit_left = np.argmin(loss_left)
    digit_middle = np.argmin(loss_middle)
    digit_right = np.argmin(loss_right)
    digit_predict = digit_left * 100 + digit_middle * 10 + digit_right * 1
    return digit_predict

def real_predict(sample_path, sample_number, path, l2_norm, l2_predict):
    results = np.zeros((sample_number, 1))
    for num in range(1, sample_number + 1):
        img = cv2.imread(path + "\\image_%d.jpg" % num)
        results[num -1, 0] = l2_predict(sample_path, img, l2_norm)
    return results



'''
video_path: the path of the video
save_path: The screen-shot will be saved in the folder, and you can delete it after getting the result
sample_path: The folder of the sample images
desire_num: How many sample you want
'''

if __name__ == '__main__':
    window = tk.Tk()
    window.title('Video to vector')
    window.geometry('800x600')
    result = tk.StringVar()
    result.set('')
    # Path of video
    video_path = tk.StringVar()
    video_path.set("")
    labelLine1 = tk.Label(window, text="Video_path:", font=('Arial', 15), width=10).place(x=75, y=50, anchor='nw')
    entryLine1 = tk.Entry(window, show=None, font=('Arial', 15), textvariable=video_path, width=20)
    entryLine1.place(x=250, y=50, anchor='nw')
    # Path of sample
    sample_path = tk.StringVar()
    sample_path.set("")
    labelLine2 = tk.Label(window, text="Sample_path:", font=('Arial', 15), width=10).place(x=75, y=100, anchor='nw')
    entryLine2 = tk.Entry(window, show=None, font=('Arial', 15), textvariable=sample_path, width=20)
    entryLine2.place(x=250, y=100, anchor='nw')
    # Path of saving
    save_path = tk.StringVar()
    save_path.set("")
    labelLine3 = tk.Label(window, text="Save_path:", font=('Arial', 15), width=10).place(x=75, y=150, anchor='nw')
    entryLine3 = tk.Entry(window, show=None, font=('Arial', 15), textvariable=save_path, width=20)
    entryLine3.place(x=250, y=150, anchor='nw')
    # Input Number
    sample_num = tk.StringVar()
    sample_num.set("")
    labelLine4 = tk.Label(window, text="Sampling number:", font=('Arial', 15), width=10).place(x=75, y=200, anchor='nw')
    entryLine4 = tk.Entry(window, show=None, font=('Arial', 15), textvariable=sample_num, width=20)
    entryLine4.place(x=250, y=200, anchor='nw')
    def close():
        window.destroy()
    button_add = tk.Button(window, text='Confirm', bg='silver', font=('Arial', 12),  command=close ,width=8)
    button_add.place(x=200, y=300, anchor='nw')

    window.mainloop()


print(video_path.get())
video_path = video_path.get()
save_path = save_path.get()
sample_path = sample_path.get()
desire_num = sample_num.get()
desire_num = int(desire_num)
results = np.zeros((desire_num, 1))
screen_shot(video_path, save_path, desire_num)
results = real_predict(sample_path, desire_num, save_path, l2_norm, l2_predict)
print('hello')
print(results)
