 
import cv2
import mediapipe as mp


class HandTracker:

    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7
    ):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:

            for hand_landmarks in self.results.multi_hand_landmarks:

                if draw:

                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

        return frame

    def find_position(self, frame, hand_number=0):

        landmark_list = []

        if self.results and self.results.multi_hand_landmarks:

            my_hand = self.results.multi_hand_landmarks[hand_number]

            height, width, _ = frame.shape

            for index, landmark in enumerate(my_hand.landmark):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmark_list.append([index, x, y])

        return landmark_list

    def count_fingers(self, landmark_list):

        fingers = []

        if not landmark_list:
            return 0

        # Thumb
        if landmark_list[4][1] > landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers
        finger_tips = [8, 12, 16, 20]
        finger_joints = [6, 10, 14, 18]

        for tip, joint in zip(finger_tips, finger_joints):

            if landmark_list[tip][2] < landmark_list[joint][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def get_finger_coordinates(self):

        if self.results and self.results.multi_hand_landmarks:

            hand = self.results.multi_hand_landmarks[0]

            index_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]
            middle_tip = hand.landmark[12]

            return (
                index_tip.x,
                index_tip.y,
                thumb_tip.x,
                thumb_tip.y,
                middle_tip.x,
                middle_tip.y
            )

        return None

    def close(self):
        self.hands.close()
