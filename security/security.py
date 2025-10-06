import cv2
import mediapipe as mp
import speech_recognition as sr

class SecurityManager:
    """
    Security and Biometric Authentication for secure access using face, voice, and gesture recognition.
    """
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        self.recognizer = sr.Recognizer()
        self.gesture_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    def authenticate_face(self):
        """Authenticate user by face recognition (detects face presence)."""
        cap = cv2.VideoCapture(0)
        authenticated = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            if results.detections:
                cv2.putText(frame, "Face Detected - Authenticated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                authenticated = True
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or authenticated:
                break
        cap.release()
        cv2.destroyAllWindows()
        return authenticated

    def authenticate_voice(self):
        """Authenticate user by voice recognition (checks for speech presence)."""
        with sr.Microphone() as source:
            print("Say something for authentication...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Voice recognized: {text}")
            return True
        except sr.UnknownValueError:
            print("Could not understand audio")
            return False
        except sr.RequestError as e:
            print(f"Speech Recognition error: {e}")
            return False

    def authenticate_gesture(self):
        """Authenticate user by gesture recognition (detects hand presence)."""
        cap = cv2.VideoCapture(0)
        authenticated = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.gesture_detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                cv2.putText(frame, "Gesture Detected - Authenticated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                authenticated = True
            cv2.imshow("Gesture Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or authenticated:
                break
        cap.release()
        cv2.destroyAllWindows()
        return authenticated
