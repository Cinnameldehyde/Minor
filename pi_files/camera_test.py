import cv2

def check_usb_camera():
    # Open the first available camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("USB Camera not detected. Please check the connection.")
        return
    
    print("USB Camera detected successfully.")
    print("Press 'q' to quit the camera preview.")
    
    # Display the camera feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame from USB Camera.")
            break

        # Show the frame in a window
        cv2.imshow('USB Camera Feed', frame)

        # Quit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_usb_camera()
