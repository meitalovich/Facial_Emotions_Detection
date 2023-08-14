import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
stage = None
create = None
def findPosition(image, draw=True):
  lmList = []
  if results.pose_landmarks:
      mp_drawing.draw_landmarks(
         image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      for id, lm in enumerate(results.pose_landmarks.landmark):
          h, w, c = image.shape
          cx, cy = int(lm.x * w), int(lm.y * h)
          lmList.append([id, cx, cy])
  return lmList
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.96,
    min_tracking_confidence=0.95) as pose:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (640,480))
    if not success:
      print("Ignoring empty camera frame.")
      continue
 
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
   
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lmList = findPosition(image, draw=True)

    if len(lmList) != 0:
        if(lmList[6] and lmList[3] and lmList[0] and lmList[10] and lmList[9]):
            distance1=lmList[15][1]-lmList[11][1]
            distance2=lmList[16][1]-lmList[12][1]
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            
            if(lmList[16][2] < lmList[6][2] and lmList[15][2] < lmList[3][2] and distance1<5 and distance2<5):
                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "Asking for Help!" 
                print("D1 " , distance1)
                print("D2 " , distance2)
                
            elif(lmList[16][2] < lmList[6][2] and lmList[15][2] > lmList[3][2]):
                  #if(lmList[14][2] > lmList[6][2] and lmList[13][2] < lmList[3][1]):
                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                  #  cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "Question?"
                
        
            elif(lmList[16][2] > lmList[6][2] and lmList[15][2] < lmList[3][2]):
                  #if(lmList[13][2] > lmList[3][1] and lmList[14][2] < lmList[6][2]):
                  #  cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "Question?"
                
      
            elif(lmList[16][2] > lmList[6][2] and lmList[15][2] > lmList[3][2]):
                stage = "Normal"
                counter=0
                

            
                
            
        else:
            stage = " "
        text = stage
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)
    cv2.imshow('MediaPipe Pose', image)
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, exit
    if key == ord("q"):
      break
cv2.destroyAllWindows()