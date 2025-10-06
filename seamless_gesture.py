import cv2
import time
import math
from collections import deque
import numpy as np

from hand_tracker import HandTracker
from filters import OneEuroFilter
from gesture_config import load_config
from gesture_logger import GestureLogger

# States
STATE_IDLE = "IDLE"
STATE_ACTIVE = "ACTIVE"        # Wake/active listening (no action yet)
STATE_MEDIA = "MEDIA_PLAYING"   # Context: controlling media
STATE_MENU = "IN_MENU"          # Quick settings palette open

# Cooldown (seconds)
COOLDOWN = 0.2

# Hold thresholds
FIST_HOLD_SEC = 1.0

# Smoothing
SMOOTH_N = 5


def norm_to_px(p, width, height):
    if p is None:
        return None
    return int(p[0] * width), int(p[1] * height)


def fingers_up(landmarks, handedness):
    """Return [thumb, index, middle, ring, pinky] flags using normalized landmarks."""
    if not landmarks or len(landmarks) < 21:
        return [0, 0, 0, 0, 0]
    # Thumb: compare x of tip (4) vs IP (3) depending on hand
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness == 'Right':
        thumb_flag = 1 if thumb_tip[0] > thumb_ip[0] else 0
    else:
        thumb_flag = 1 if thumb_tip[0] < thumb_ip[0] else 0
    # Others: tip y < pip y
    idx = 1 if landmarks[8][1] < landmarks[6][1] else 0
    mid = 1 if landmarks[12][1] < landmarks[10][1] else 0
    ring = 1 if landmarks[16][1] < landmarks[14][1] else 0
    pinky = 1 if landmarks[20][1] < landmarks[18][1] else 0
    return [thumb_flag, idx, mid, ring, pinky]


def is_open_hand(landmarks, handedness):
    f = fingers_up(landmarks, handedness)
    return sum(f) == 5


def is_fist(landmarks, handedness):
    f = fingers_up(landmarks, handedness)
    return sum(f) == 0


def is_thumbs_up(landmarks):
    if not landmarks or len(landmarks) < 5:
        return False
    # Thumb up (tip above IP)
    thumb_up = landmarks[4][1] < landmarks[3][1]
    # Others down
    index_down = landmarks[8][1] > landmarks[6][1]
    middle_down = landmarks[12][1] > landmarks[10][1]
    ring_down = landmarks[16][1] > landmarks[14][1]
    pinky_down = landmarks[20][1] > landmarks[18][1]
    # Thumb vertical-ish
    vertical = abs(landmarks[4][1] - landmarks[2][1]) > abs(landmarks[4][0] - landmarks[2][0])
    return thumb_up and index_down and middle_down and ring_down and pinky_down and vertical


def is_peace_sign(landmarks, sep_thresh=0.03):
    if not landmarks or len(landmarks) < 21:
        return False
    index_up = landmarks[8][1] < landmarks[6][1]
    middle_up = landmarks[12][1] < landmarks[10][1]
    ring_down = landmarks[16][1] > landmarks[14][1]
    pinky_down = landmarks[20][1] > landmarks[18][1]
    sep = abs(landmarks[8][0] - landmarks[12][0])
    return index_up and middle_up and ring_down and pinky_down and sep > sep_thresh


def palm_center(landmarks):
    if not landmarks:
        return (0.5, 0.5)
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hand_angle(landmarks):
    """Angle (radians) of vector wrist->index MCP relative to +x axis."""
    if not landmarks or len(landmarks) <= 5:
        return 0.0
    wrist = landmarks[0]
    idx_mcp = landmarks[5]
    dx = idx_mcp[0] - wrist[0]
    dy = idx_mcp[1] - wrist[1]
    return math.atan2(dy, dx)


def draw_subtle_hand_indicator(img):
    h, w = img.shape[:2]
    overlay = img.copy()
    # Semi-transparent rounded rectangle in corner
    x, y, rw, rh = 15, 15, 80, 60
    cv2.rectangle(overlay, (x, y), (x + rw, y + rh), (255, 255, 255), -1)
    img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    # Outline
    cv2.rectangle(img, (x, y), (x + rw, y + rh), (220, 220, 220), 1)
    # Small hand glyph
    cv2.circle(img, (x + 20, y + 35), 8, (230, 230, 230), 2)
    for i in range(5):
        cv2.line(img, (x + 35 + i * 8, y + 20), (x + 35 + i * 8, y + 45), (230, 230, 230), 2)


def draw_progress_bar(img, progress, active=False):
    h, w = img.shape[:2]
    bar_w, bar_h = int(w * 0.6), 12
    x = (w - bar_w) // 2
    y = h - 40
    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), 1)
    if active:
        fill_w = int(bar_w * np.clip(progress, 0, 1))
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + fill_w, y + bar_h), (0, 255, 200), -1)
        img[:] = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)


def draw_volume_indicator(img, vol):
    h, w = img.shape[:2]
    cx, cy = w - 60, h - 60
    # Icon (speaker triangle)
    pts = np.array([[cx-18, cy+10], [cx-18, cy-10], [cx, cy]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(200, 255, 200), thickness=2)
    # Bars
    for i in range(1, 4):
        amp = i * 8
        color = (150 + i*30, 255, 150)
        if vol > i / 4.0:
            cv2.ellipse(img, (cx+8, cy), (amp, amp), 0, -30, 30, color, 2)


def draw_play_pause_bloom(img, is_playing, center):
    if center is None:
        return
    cx, cy = center
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), 28, (255, 255, 255), -1)
    img[:] = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    # Symbol
    if is_playing:
        # Pause bars
        cv2.rectangle(img, (cx-10, cy-12), (cx-4, cy+12), (255, 255, 255), -1)
        cv2.rectangle(img, (cx+4, cy-12), (cx+10, cy+12), (255, 255, 255), -1)
    else:
        # Play triangle
        pts = np.array([[cx-8, cy-14], [cx-8, cy+14], [cx+12, cy]], np.int32)
        cv2.fillConvexPoly(img, pts, (255, 255, 255))


def draw_thumbs_up(img, wrist_px):
    if wrist_px is None:
        return
    cx, cy = wrist_px[0] + 40, wrist_px[1] - 20
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), 20, (255, 255, 255), -1)
    img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    # Thumb glyph
    cv2.line(img, (cx-8, cy+8), (cx-2, cy-8), (255, 255, 255), 2)
    cv2.line(img, (cx-2, cy-8), (cx+10, cy-10), (255, 255, 255), 2)


def draw_page_transition(img, direction):
    # direction: -1 (left), +1 (right)
    h, w = img.shape[:2]
    overlay = img.copy()
    width = int(w * 0.35)
    if direction < 0:
        cv2.rectangle(overlay, (0, 0), (width, h), (200, 200, 255), -1)
    else:
        cv2.rectangle(overlay, (w - width, 0), (w, h), (200, 200, 255), -1)
    img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)


def draw_quick_settings(img):
    h, w = img.shape[:2]
    x, y, rw, rh = w - 240, 40, 200, 140
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + rw, y + rh), (255, 255, 255), -1)
    img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    cv2.rectangle(img, (x, y), (x + rw, y + rh), (220, 220, 220), 1)
    cv2.putText(img, "Quick Settings", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    items = ["Wi-Fi", "BT", "Brightness", "Do Not Disturb"]
    for i, it in enumerate(items):
        cv2.putText(img, f"- {it}", (x + 12, y + 50 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    width, height = 1280, 720
    cap.set(3, width)
    cap.set(4, height)

    tracker = HandTracker(max_hands=1, detection_con=0.8, track_con=0.7, static_image_mode=False)

    # Load config and init logger
    cfg = load_config()
    logger = GestureLogger(enabled=cfg.get("logging", {}).get("enabled", False),
                           path=cfg.get("logging", {}).get("path", "gesture_events.jsonl"))

    # App state
    state = STATE_IDLE
    is_playing = False
    like_on = False
    track_index = 1
    progress = 0.2   # 0..1 timeline
    volume = 0.5     # 0..1
    menu_open = False

    # Gesture runtime
    last_action = {}
    fist_start_time = None
    last_center = None
    center_hist = deque(maxlen=SMOOTH_N)
    angle_hist = deque(maxlen=SMOOTH_N)
    pinch_active = False
    pinch_start_x = None

    # Swipe transition effect timer
    swipe_effect_until = 0

    # One Euro filters for center and angle
    of_center_x = OneEuroFilter(min_cutoff=cfg["filters"]["min_cutoff"], beta=cfg["filters"]["beta"], d_cutoff=cfg["filters"]["d_cutoff"])
    of_center_y = OneEuroFilter(min_cutoff=cfg["filters"]["min_cutoff"], beta=cfg["filters"]["beta"], d_cutoff=cfg["filters"]["d_cutoff"])
    of_angle = OneEuroFilter(min_cutoff=cfg["filters"]["min_cutoff"], beta=cfg["filters"]["beta"], d_cutoff=cfg["filters"]["d_cutoff"])

    while True:
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)

        img, landmarks, hand_info = tracker.find_hands(img, draw=False)
        now = time.time()

        # Reset state if no hand
        if not hand_info["detected"]:
            if state != STATE_IDLE and now - last_action.get("no_hand", 0) > 1.5:
                state = STATE_IDLE
            last_action["no_hand"] = now
            # Idle UI minimal
            cv2.putText(img, "[IDLE] Show palm to wake", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            if swipe_effect_until > now:
                draw_page_transition(img, -1 if last_action.get("swipe_dir", 1) < 0 else 1)
            cv2.imshow("SeamLess Gesture Demo", img)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break
            continue

        # Compute smoothed center and angle (One Euro)
        center = palm_center(landmarks)
        center_hist.append(center)
        raw_cx = sum(p[0] for p in center_hist) / len(center_hist)
        raw_cy = sum(p[1] for p in center_hist) / len(center_hist)
        cx = of_center_x(raw_cx, now)
        cy = of_center_y(raw_cy, now)
        center_smooth = (cx, cy)
        ang = hand_angle(landmarks)
        angle_hist.append(ang)
        raw_ang = sum(angle_hist) / len(angle_hist)
        ang_smooth = of_angle(raw_ang, now)

        # Wake gesture
        if state == STATE_IDLE and is_open_hand(landmarks, hand_info.get("handedness", "Unknown")):
            state = STATE_ACTIVE
            last_action["wake"] = now

        # Promote to MEDIA context when active
        if state == STATE_ACTIVE:
            draw_subtle_hand_indicator(img)
            # If any control gesture happens, switch to MEDIA
            if is_fist(landmarks, hand_info.get("handedness", "Unknown")) or is_thumbs_up(landmarks) or is_peace_sign(landmarks, sep_thresh=cfg["thresholds"]["peace_sep"]):
                state = STATE_MEDIA
                logger.log("activate_media", {})

        # System-wide gestures (always available)
        # Fist hold for play/pause
        if is_fist(landmarks, hand_info.get("handedness", "Unknown")):
            if fist_start_time is None:
                fist_start_time = now
            hold = now - fist_start_time
            # Draw subtle ring to show hold
            px = norm_to_px(center_smooth, width, height)
            if px:
                cv2.circle(img, px, 22, (230, 230, 230), 2)
            if hold >= cfg["thresholds"]["fist_hold_sec"] and now - last_action.get("playpause", 0) > cfg["cooldowns"]["playpause"]:
                is_playing = not is_playing
                last_action["playpause"] = now
                fist_start_time = None
                draw_play_pause_bloom(img, is_playing, px)
                logger.log("playpause", {"playing": is_playing})
        else:
            fist_start_time = None

        # Thumbs up => like
        if is_thumbs_up(landmarks) and now - last_action.get("like", 0) > cfg["cooldowns"]["like"]:
            like_on = not like_on
            last_action["like"] = now
            wrist_px = norm_to_px((landmarks[0][0], landmarks[0][1]), width, height)
            draw_thumbs_up(img, wrist_px)
            logger.log("like_toggle", {"like": like_on})

        # Peace sign => toggle quick settings palette
        if is_peace_sign(landmarks, sep_thresh=cfg["thresholds"]["peace_sep"]) and now - last_action.get("palette", 0) > cfg["cooldowns"]["palette"]:
            menu_open = not menu_open
            last_action["palette"] = now
            state = STATE_MENU if menu_open else STATE_MEDIA
            logger.log("palette_toggle", {"open": menu_open})

        # Context-specific controls
        if state in (STATE_MEDIA, STATE_MENU, STATE_ACTIVE):
            # Pinch + drag for seeking
            is_pinching, pinch_dist = tracker.detect_pinch()
            if is_pinching and not pinch_active:
                pinch_active = True
                pinch_start_x = center_smooth[0]
            elif not is_pinching and pinch_active:
                pinch_active = False
                pinch_start_x = None
            if pinch_active and pinch_start_x is not None:
                delta = (center_smooth[0] - pinch_start_x) * 1.5  # sensitivity
                if now - last_action.get("seek_update", 0) > cfg["cooldowns"]["seek_update"]:
                    progress = float(np.clip(progress + delta, 0.0, 1.0))
                    last_action["seek_update"] = now
                    logger.log("seek", {"progress": progress})
                pinch_start_x = center_smooth[0]
                draw_progress_bar(img, progress, active=True)
            else:
                draw_progress_bar(img, progress, active=False)

            # Open-hand rotation for volume
            if is_open_hand(landmarks, hand_info.get("handedness", "Unknown")):
                if len(angle_hist) >= 2:
                    dtheta = (angle_hist[-1] - angle_hist[0])
                    while dtheta > math.pi:
                        dtheta -= 2 * math.pi
                    while dtheta < -math.pi:
                        dtheta += 2 * math.pi
                    new_volume = float(np.clip(volume - dtheta * 0.2, 0.0, 1.0))
                    if abs(new_volume - volume) > 0.01:
                        volume = new_volume
                        logger.log("volume", {"volume": volume})
                draw_volume_indicator(img, volume)

            # Swipe left/right for next/previous (flat/open hand, quick lateral move)
            if last_center is not None:
                vx = (center_smooth[0] - last_center[0])
                speed = abs(vx)
                if speed > cfg["thresholds"]["swipe_speed"] and is_open_hand(landmarks, hand_info.get("handedness", "Unknown")) and not pinch_active:
                    if now - last_action.get("swipe", 0) > cfg["cooldowns"]["swipe"]:
                        direction = 1 if vx > 0 else -1
                        track_index += direction
                        last_action["swipe"] = now
                        last_action["swipe_dir"] = direction
                        swipe_effect_until = now + 0.35
                        logger.log("swipe", {"direction": direction, "track": track_index})
            last_center = center_smooth

        # Menu rendering and dismissal
        if state == STATE_MENU and menu_open:
            draw_quick_settings(img)
            # Dismiss by swipe or peace sign (handled above)
            if swipe_effect_until > now:
                draw_page_transition(img, -1 if last_action.get("swipe_dir", 1) < 0 else 1)

        # Show status (subtle)
        status = f"STATE: {state} | PLAY:{'ON' if is_playing else 'OFF'} VOL:{int(volume*100)}% POS:{int(progress*100)}% TRK:{track_index} {'LIKE' if like_on else ''}"
        cv2.putText(img, status, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if swipe_effect_until > now:
            draw_page_transition(img, -1 if last_action.get("swipe_dir", 1) < 0 else 1)

        cv2.imshow("SeamLess Gesture Demo", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

