import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
else:
    ret, frame = cap.read()
    if ret:
        print("✅ Camera works!")
        cv2.imshow("Test", frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cap.release()