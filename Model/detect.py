import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import argparse
    
def get_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
    if results.face_landmarks:
        
        for landmark in results.face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
    return landmarks

def preprocess_landmarks(landmarks, input_shape):
    
        landmarks = np.array(landmarks).flatten()
        
        if landmarks.shape[0] < input_shape[0]:
            padding = np.zeros(input_shape[0] - landmarks.shape[0])
            landmarks = np.concatenate((landmarks, padding))
            
        landmarks = landmarks.reshape(input_shape)
        
        return landmarks

def detect_emotion(args):
    # Load the pre-trained model
    model = load_model(args.model_path)

    # Load label encoder
    labels = np.load('labels.npy' , allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = labels

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Determine the correct input shape for the model
    input_shape = (model.input_shape[1], model.input_shape[2])

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            landmarks = get_landmarks(results)
            if landmarks:
                processed_landmarks = preprocess_landmarks(landmarks, input_shape)
                processed_landmarks = np.expand_dims(processed_landmarks, axis=0)  # Add batch dimension

                # Predict emotion
                y_pred = model.predict(processed_landmarks)
                emotion = label_encoder.inverse_transform([np.argmax(y_pred)])
                probability = np.max(y_pred)
                
                # Display prediction
                cv2.putText(frame, f'Emotion: {emotion[0]} ({probability:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw landmarks
            if args.with_holistic :
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

            cv2.imshow('Real-Time Emotion Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Real Time Emotion Detection')
    
    parser.add_argument('-wh' , '--with_holistic' , action= 'store_true')
    parser.add_argument('-mp' , '--model_path', required= True , type= str , help='The path for the model')
    
    args = parser.parse_args()
    detect_emotion(args)