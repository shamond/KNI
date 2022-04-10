###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import functions as f

# Configure depth and color streams

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        hole_filling = rs.hole_filling_filter(2)
        filled_depth = hole_filling.process(depth_frame)
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))

        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # color_image, depth_colormap

        # first step draw lines to crop image
        f.draw_lines(color_image, (30, 480), (160, 80))
        f.draw_lines(color_image, (160, 80), (480, 80))
        f.draw_lines(color_image, (480, 80), (610, 480))
        # f.draw_lines(images,(670, 480), (800, 80))
        # f.draw_lines(images,(800, 80), (1120, 80))
        # f.draw_lines(images, (1120, 80), (1250, 480))
        # second step trim the image as the lines lead, optionally convert back to bgr its boolean type default = True
        crop = f.crop_image(color_image)
        color_crop = f.crop_image(color_image, False)
        # third step use filter canny to extract edges
        edges = f.edges(crop)

        # test of blurring
        dict_blurring ={ "gauss" : f.gauss_filter(crop, (3, 3), 9),
                         "mean" : f.mean_filter(crop, (5, 5)),
                         "median" : f.median_filter(crop, 5) }
        color = (
        f.gauss_filter(color_crop, (3, 3), 1), f.mean_filter(color_crop, (5, 5)), f.median_filter(color_crop, 5))

        # set brightness
        dict_brightness = { "light_median" : f.brightness(dict_blurring["median"], 2, 25),
                            "dark_median" : f.brightness(dict_blurring["median"], 1, -25),
                            "light_mean" : f.brightness(dict_blurring ["mean"], 2, 25),
                            "dark_mean" : f.brightness(dict_blurring["mean"], 1, -25),
                            "light_gauss" : f.brightness(dict_blurring["gauss"], 2, 25),
                            "dark_gauss" : f.brightness(dict_blurring["gauss"], 1, -25) ,
                            "bright_color" : f.brightness(color[0], 2, 25),
                            "dark_color" : f.brightness(color[0], 1, -25) }
        # detect lines by edges
        white_lines = (255, 255, 255)
        black_lines = (0, 0, 0)
        green_lines = (0, 255, 0)
        dict_lines = {"lines for_gauss_bright": f.transform_hough(dict_brightness["light_gauss"], edges, black_lines),
                      "lines for_mean_bright": f.transform_hough(dict_brightness["light_mean"], edges, black_lines),
                      "lines for_median_bright": f.transform_hough(dict_brightness["light_median"], edges, black_lines),
                      "lines for_gauss_dark": f.transform_hough(dict_brightness["dark_gauss"], edges, white_lines),
                      "lines for_mean_dark": f.transform_hough(dict_brightness["dark_mean"], edges, white_lines),
                      "lines for_median_dark": f.transform_hough(dict_brightness["dark_median"], edges, white_lines),
                      "color bright": f.transform_hough(dict_brightness["bright_color"], edges, green_lines),
                      "color dark": f.transform_hough(color[0], edges, green_lines),
                      "normal color": f.transform_hough(color[0], edges, green_lines)
                                                                                        }



        # f.show("gauss light", light_gauss)
        # f.show("gauss dark", dark_gauss)
        # f.show("median light", light_median)
        # f.show("median dark", dark_median)
        # f.show("mean light", light_mean)
        # f.show("mean dark", dark_mean)
        # f.show("color light",dict_brightness["bright_color"])
        # f.show("color dark", dict_brightness["dark_color"])
        f.show("normal color", color[0])

        # show histogram
        # f.histogram_gray(light_gauss)
        # f.histogram_gray(dark_gauss)
        # f.histogram_gray(light_median)
        # f.histogram_gray(dark_median)
        # f.histogram_gray(light_mean)
        # f.histogram_gray(dark_mean)
        # f.histogram_color(dark_color)
        # f.histogram_color(bright_color)
        f.histogram_color(color[0])



finally:
    pipeline.stop()

    # Cropping an image
    # cropped_image = images[80:120, 50:100]
    # Display cropped image
    # cv2.imshow("cropped", cropped_image)
