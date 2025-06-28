import time
import warnings
import argparse 
import csv

import mediapipe as mp  
import cv2  
import numpy as np

from utils import create_csv_structure

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions


def collect_data(args):
    
    total_num_coords = 33 + 468 # (https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) 
    
    # Create CSV header ONCE before processing any emotions
    create_csv_structure(total_num_coords, path_of_csv= args.path_of_csv)
                         
    for idx in range(len(args.emotions_list)):
        cap = cv2.VideoCapture(0)
        # Initiate holistic model
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
                )

                # 2. Right hand
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
                )

                # 3. Left Hand
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(
                        np.array(
                            [
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                                for landmark in pose
                            ]
                        ).flatten()
                    )

                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(
                        np.array(
                            [
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                                for landmark in face
                            ]
                        ).flatten()
                    )
                    # Concate rows
                    row = pose_row + face_row

                    # Append class name
                    row.insert(0, args.emotions_list[idx])

                    # Export to CSV
                    with open(args.path_of_csv, mode="a", newline="") as f:
                        csv_writer = csv.writer(
                            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                        )
                        csv_writer.writerow(row)
                        print(len(row))

                except: 
                    pass

                cv2.imshow("Raw Webcam Feed", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(args.pause)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Collecting Data Script")
    
    parser.add_argument(
        '-el',
        '--emotions_list',
        required= True,
        nargs= '*',
        type= str ,
        help= 'List of emotions that you will collect the data for them.'
    )
    parser.add_argument(
        '-ps',
        '--path_of_csv',
        required= True ,
        type= str,
        help= "Path to the CSV file that you want to save data in it."
        )
    parser.add_argument(
        '-p',
        '--pause',
        default= 2,
        type= int,
        help= 'The duration (in seconds) for which the camera remains inactive after completing the capture of each emotion.'
    )
    args = parser.parse_args()
    
    collect_data(args)