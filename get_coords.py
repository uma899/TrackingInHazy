# get_coords.py
import cv2
import argparse
import os

def main():
    # 1. Define command-line arguments
    parser = argparse.ArgumentParser(description="Interactive tool to get bounding box coordinates from an image.")
    
    # Image path argument
    parser.add_argument('--image', type=str, default='../IIT_HAZY/vid8/test/hazy/00000001.png', 
                        help='Path to the first frame/image file')

    args = parser.parse_args()

    # 2. Load your image file
    image_path = args.image 
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not open or find the image at '{image_path}'.")
        print("Please check the file path and try again.")
        return

    # Create a window for display
    window_name = "Select Target Object (Press ENTER/SPACE when done, 'c' to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--> INSTRUCTIONS:")
    print("1. Click and drag your mouse over the target object to draw the bounding box.")
    print("2. Press ENTER or SPACE to confirm the selection.")
    print("3. Press 'c' to clear selection and retry, or ESC to exit.")

    # 3. Open OpenCV's built-in interactive ROI (Region of Interest) selector
    # showCrosshair=True displays a crosshair inside the box for precise targeting
    init_rect = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)

    # 4. Parse and print out the exact coordinates
    # init_rect returns: (xmin, ymin, width, height)
    xmin, ymin, width, height = init_rect

    cv2.destroyAllWindows()

    if width > 0 and height > 0:
        print("\n" + "="*40)
        print("SUCCESS! Copy the array below for your tracker:")
        print(f"init_box = [{xmin}, {ymin}, {width}, {height}]")
        print("="*40)
    else:
        print("\nSelection canceled or invalid box size generated.")

if __name__ == "__main__":
    main()