import cv2

def crop_center(frame, target_width=320, target_height=240):
    height, width, _ = frame.shape
    start_x = max((width - target_width) // 2, 0)
    start_y = max((height - target_height) // 2, 0)
    end_x = start_x + target_width
    end_y = start_y + target_height
    return frame[start_y:end_y, start_x:end_x]

def main():
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)  # Change to detected index
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Fix format issues
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution (change if needed)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully!")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cropped_frame = crop_center(frame)
        cv2.imshow('Cropped Camera Feed', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
