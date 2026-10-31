
import streamlit as st
import cv2
import time
import math
import mediapipe as mp
import pyautogui
import numpy as np

from hand_tracking import HandTracker


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Virtual Mouse",
    page_icon="🖱️",
    layout="wide"
)


# ---------------- SESSION STATE ----------------

if "mouse_enabled" not in st.session_state:
    st.session_state.mouse_enabled = False

if "running" not in st.session_state:
    st.session_state.running = False


# ---------------- TITLE ----------------

st.title("🖱️ AI Virtual Mouse")
st.subheader("Hand Gesture Recognition System")

st.info(
    "Control your computer using hand gestures "
    "with Computer Vision and MediaPipe."
)


# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Mouse Settings")

mouse_enabled = st.sidebar.toggle(
    "Virtual Mouse ON/OFF",
    value=st.session_state.mouse_enabled
)

st.session_state.mouse_enabled = mouse_enabled

smoothness = st.sidebar.slider(
    "Mouse Smoothness",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.write("### 🎮 Gesture Controls")

st.sidebar.write("☝️ Index Finger → Move Mouse")
st.sidebar.write("🤏 Thumb + Index → Left Click")
st.sidebar.write("🤏 Thumb + Middle → Right Click")
st.sidebar.write("✌️ Peace → Scroll")
st.sidebar.write("✊ Fist → Pause Mouse")


# ---------------- DASHBOARD LAYOUT ----------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Webcam")
    camera_placeholder = st.empty()

with col2:
    st.subheader("📊 Detection Information")

    finger_placeholder = st.empty()
    gesture_placeholder = st.empty()
    fps_placeholder = st.empty()
    mouse_placeholder = st.empty()

    st.subheader("📝 Gesture History")
    history_placeholder = st.empty()


# ---------------- START / STOP ----------------

start_button = st.button("▶️ Start Virtual Mouse")

stop_button = st.button("⏹️ Stop Virtual Mouse")


if start_button:
    st.session_state.running = True

if stop_button:
    st.session_state.running = False
    st.session_state.mouse_enabled = False


# ---------------- MAIN PROGRAM ----------------

if st.session_state.running:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        st.error("Camera open nahi hua!")

    else:

        tracker = HandTracker(max_hands=1)

        previous_time = time.time()

        gesture_history = []

        previous_middle_y = None

        left_click_pressed = False
        right_click_pressed = False

        previous_x = 0
        previous_y = 0

        screen_width, screen_height = pyautogui.size()

        try:

            while st.session_state.running:

                success, frame = camera.read()

                if not success:
                    st.error("Camera frame read nahi hua!")
                    break

                frame = cv2.flip(frame, 1)

                frame = tracker.find_hands(frame)

                landmarks = tracker.find_position(frame)

                fingers = 0
                gesture = "No Hand Detected"

                # -------- FINGER COUNT --------

                if landmarks:

                    fingers = tracker.count_fingers(landmarks)

                    if fingers == 0:
                        gesture = "Fist"

                    elif fingers == 1:
                        gesture = "One Finger"

                    elif fingers == 2:
                        gesture = "Peace"

                    elif fingers == 3:
                        gesture = "Three Fingers"

                    elif fingers == 4:
                        gesture = "Four Fingers"

                    elif fingers == 5:
                        gesture = "Open Palm"

                    if (
                        not gesture_history
                        or gesture_history[-1] != gesture
                    ):

                        gesture_history.append(gesture)

                        if len(gesture_history) > 5:
                            gesture_history.pop(0)

                    # -------- MOUSE CONTROL --------

                    if st.session_state.mouse_enabled:

                        hand = tracker.results.multi_hand_landmarks[0]

                        index_tip = hand.landmark[8]
                        thumb_tip = hand.landmark[4]
                        middle_tip = hand.landmark[12]

                        index_x = index_tip.x
                        index_y = index_tip.y

                        thumb_x = thumb_tip.x
                        thumb_y = thumb_tip.y

                        middle_x = middle_tip.x
                        middle_y = middle_tip.y

                        # Fist = pause
                        if gesture == "Fist":

                            left_click_pressed = False
                            right_click_pressed = False
                            previous_middle_y = None

                        else:

                            # Index finger = Mouse Movement
                            target_x = int(index_x * screen_width)
                            target_y = int(index_y * screen_height)

                            target_x = max(
                                0,
                                min(target_x, screen_width - 1)
                            )

                            target_y = max(
                                0,
                                min(target_y, screen_height - 1)
                            )

                            if previous_x == 0:
                                previous_x = target_x

                            if previous_y == 0:
                                previous_y = target_y

                            current_x = previous_x + (
                                target_x - previous_x
                            ) / smoothness

                            current_y = previous_y + (
                                target_y - previous_y
                            ) / smoothness

                            pyautogui.moveTo(
                                int(current_x),
                                int(current_y)
                            )

                            previous_x = current_x
                            previous_y = current_y

                            # Left Click
                            distance = math.hypot(
                                index_x - thumb_x,
                                index_y - thumb_y
                            )

                            if distance < 0.04:

                                if not left_click_pressed:
                                    pyautogui.click()
                                    left_click_pressed = True

                            else:
                                left_click_pressed = False

                            # Right Click
                            middle_distance = math.hypot(
                                middle_x - thumb_x,
                                middle_y - thumb_y
                            )

                            if middle_distance < 0.04:

                                if not right_click_pressed:
                                    pyautogui.rightClick()
                                    right_click_pressed = True

                            else:
                                right_click_pressed = False

                            # Scroll
                            if gesture == "Peace":

                                if previous_middle_y is not None:

                                    movement = (
                                        middle_y - previous_middle_y
                                    )

                                    if movement > 0.02:
                                        pyautogui.scroll(-2)

                                    elif movement < -0.02:
                                        pyautogui.scroll(2)

                                previous_middle_y = middle_y

                            else:
                                previous_middle_y = None

                # -------- FPS --------

                current_time = time.time()

                fps = 1 / (
                    current_time - previous_time
                )

                previous_time = current_time

                # -------- SHOW WEBCAM --------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                camera_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

                # -------- SHOW INFORMATION --------

                finger_placeholder.metric(
                    "🖐️ Finger Count",
                    fingers
                )

                gesture_placeholder.metric(
                    "✋ Current Gesture",
                    gesture
                )

                fps_placeholder.metric(
                    "⚡ FPS",
                    int(fps)
                )

                mouse_placeholder.metric(
                    "🖱️ Virtual Mouse",
                    "ON" if st.session_state.mouse_enabled else "OFF"
                )

                # -------- HISTORY --------

                history_placeholder.write(
                    "\n".join(
                        f"• {g}" for g in reversed(gesture_history)
                    )
                )

        finally:

            camera.release()
            tracker.close()

            st.session_state.running = False

else:

    st.warning(
        "Webcam start karne ke liye "
        ' "Start Virtual Mouse" button dabao.'
    )
