import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, max_hands=1, detection_con=0.3, track_con=0.3):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,  # Lowered for better detection
            min_tracking_confidence=track_con,  # Lowered for better tracking
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return img, results

    def find_position(self, img, results, hand_no=0, draw=True):
        lm_list = []
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def get_fingers_state(self, lm_list):
        if len(lm_list) < 21:
            return []

        fingers = []
        # Thumb
        if lm_list[4][1] < lm_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if lm_list[4 * id + 3][2] < lm_list[4 * id + 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_gesture(self, lm_list):
        if len(lm_list) < 21:
            return "none"

        fingers = self.get_fingers_state(lm_list)

        # Pinch (thumb and index)
        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        distance = (
            (thumb_tip[1] - index_tip[1]) ** 2 + (thumb_tip[2] - index_tip[2]) ** 2
        ) ** 0.5

        if distance < 30:
            return "pinch"
        elif all(fingers[i] == 1 for i in [1, 2]) and all(
            fingers[i] == 0 for i in [0, 3, 4]
        ):
            return "peace"
        elif fingers[1] and not any(fingers[2:]):
            return "point"
        elif all(fingers):
            return "open"
        elif not any(fingers):
            return "fist"
        return "unknown"
