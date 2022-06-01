# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import functions as f

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, "C:\\Users\\Bartek\\Documents\\3.bag")

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth,640,480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    # Start streaming from file
    pipeline.start(config)

    # Create colorizer object
    colorizer = rs.colorizer()
    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        # Get depth frame and color
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)
        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(color_image)
        white_line = f.white_lane_detect(color_image)
        crop = f.crop_image(white_line,[(0,380),(150,480/2),(640/2,480/2),(640,480),(0,480)],False)
        blur = f.gauss_filter(color_image, (3, 3), 9)
        edges = f.edges(crop)

        detect_lines = f.transform_hough(blur,edges,(0,255,0))


        # Render image in opencv window
        #stack = np.hstack((crop,depth_color_image))
        cv2.imshow("depth image", depth_color_image)
        cv2.imshow("egdes",white_line)
        cv2.imshow("color image", blur)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass