import cv2 as cv

import nst
import utils


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the camera")
        exit()

    model = nst.get_transformer_net_model('./models/epoch_1.model')
    while True:
        result, frame = cap.read()
        if not result:
            print("Can't receive frame. Exiting...")
            break

        frame = utils.opencv_to_pil(frame)

        frame = nst.stylize(frame, model)
        frame = utils.pil_to_opencv(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()

