import time

import cv2 as cv

import nst
import utils


def main():
    cap = cv.VideoCapture("./cookie.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter('output.mp4', fourcc, 20, (1280, 720))
    # cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the camera")
        exit()

    frame_count = 0
    model = nst.get_transformer_net_model('./models/epoch_1.model')
    start_time = time.time()
    while cap.isOpened():
        result, frame = cap.read()
        if not result:
            print("Can't receive frame. Exiting...")
            break

        frame = utils.opencv_to_pil(frame)

        frame = nst.stylize(frame, model)
        frame = utils.pil_to_opencv(frame)
        out.write(frame)
        print(f'The {frame_count}-th frame is successfully processed')
        frame_count += 1
        # cv.imshow('frame', frame)
        # if cv.waitKey(1) == ord('q'):
        #     break

    print(f'The average time for processing a frame is {(time.time() - start_time) / frame_count} second')


if __name__ == '__main__':
    main()

