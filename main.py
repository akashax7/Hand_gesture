 
import cv2
import time
import math
from collections import deque

from hand_tracking import HandTracker
from gesture_control import move_mouse, left_click, right_click, scroll_mouse


# ---------------- CONFIGURATION ----------------

gesture_history = deque(maxlen=5)

mouse_control = False

left_click_pressed = False
right_click_pressed = False

previous_middle_y = None


# ---------------- GESTURE RECOGNITION ----------------

def recognize_gesture(fingers):

    if fingers == 0:
        return "Fist"

    elif fingers == 1:
        return "One Finger"

    elif fingers == 2:
        return "Peace"

    elif fingers == 3:
        return "Three Fingers"

    elif fingers == 4:
        return "Four Fingers"

    elif fingers == 5:
        return "Open Palm"

    return "No Hand"


# ---------------- MAIN PROGRAM ----------------

def main():

    global mouse_control
    global left_click_pressed
    global right_click_pressed
    global previous_middle_y

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera open nahi hua!")
        return

    tracker = HandTracker(max_hands=1)

    previous_time = 0

    print("M = Virtual Mouse ON/OFF")
    print("Q = Quit")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Frame read nahi ho raha!")
                break

            frame = cv2.flip(frame, 1)

            frame = tracker.find_hands(frame)

            landmarks = tracker.find_position(frame)

            fingers = 0
            gesture = "No Hand Detected"

            # -------- FINGER COUNT --------

            if landmarks:

                fingers = tracker.count_fingers(landmarks)

                gesture = recognize_gesture(fingers)

                if (
                    not gesture_history
                    or gesture_history[-1] != gesture
                ):
                    gesture_history.append(gesture)

                coordinates = tracker.get_finger_coordinates()

                # -------- MOUSE CONTROL --------

                if mouse_control and coordinates:

                    index_x, index_y, thumb_x, thumb_y, middle_x, middle_y = coordinates

                    # Fist = pause mouse
                    if gesture == "Fist":

                        left_click_pressed = False
                        right_click_pressed = False
                        previous_middle_y = None

                    else:

                        # Index finger = mouse movement
                        move_mouse(
                            index_x,
                            index_y,
                            smoothness=5
                        )

                        # Thumb + Index = left click
                        distance = math.hypot(
                            index_x - thumb_x,
                            index_y - thumb_y
                        )

                        if distance < 0.04:

                            if not left_click_pressed:
                                left_click()
                                left_click_pressed = True

                        else:
                            left_click_pressed = False

                        # Thumb + Middle = right click
                        middle_distance = math.hypot(
                            middle_x - thumb_x,
                            middle_y - thumb_y
                        )

                        if middle_distance < 0.04:

                            if not right_click_pressed:
                                right_click()
                                right_click_pressed = True

                        else:
                            right_click_pressed = False

                        # Peace gesture = scroll
                        if gesture == "Peace":

                            if previous_middle_y is not None:

                                movement = middle_y - previous_middle_y

                                if movement > 0.02:
                                    scroll_mouse(-2)

                                elif movement < -0.02:
                                    scroll_mouse(2)

                            previous_middle_y = middle_y

                        else:
                            previous_middle_y = None

            else:

                left_click_pressed = False
                right_click_pressed = False
                previous_middle_y = None

            # -------- FPS CALCULATION --------

            current_time = time.time()

            if previous_time != 0:

                fps = 1 / (current_time - previous_time)

            else:

                fps = 0

            previous_time = current_time

            # -------- DISPLAY FINGER COUNT --------

            cv2.putText(
                frame,
                f"Fingers: {fingers}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # -------- DISPLAY GESTURE --------

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # -------- DISPLAY FPS --------

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            # -------- MOUSE STATUS --------

            status = "ON" if mouse_control else "OFF"

            cv2.putText(
                frame,
                f"Virtual Mouse: {status}",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if mouse_control else (0, 0, 255),
                2
            )

            # -------- GESTURE HISTORY --------

            cv2.putText(
                frame,
                "Gesture History:",
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_position = 275

            for old_gesture in reversed(gesture_history):

                cv2.putText(
                    frame,
                    old_gesture,
                    (30, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                y_position += 30

            # -------- SHOW CAMERA --------

            cv2.imshow(
                "AI Virtual Mouse | Hand Gesture Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # M = Virtual Mouse ON/OFF
            if key == ord("m"):

                mouse_control = not mouse_control

                left_click_pressed = False
                right_click_pressed = False

                print(
                    "Virtual Mouse:",
                    "ON" if mouse_control else "OFF"
                )

            # Q = Quit
            if key == ord("q"):
                break

    finally:

        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
