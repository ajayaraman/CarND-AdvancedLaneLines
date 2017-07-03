import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os

"""These functions are used to generate the calibration data
"""

""" Camera Calibration Matrix """
def compute_cam_calib():
    #read checkerboard images
    images = glob.glob("camera_cal/*.jpg")

    imgpoints = []
    objpoints = []

    #prepare object coordinates (0,0,0) (1, 0, 0) , ... (9, 6, 0)
    objp = np.zeros( (nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) #x, y cols

    #For image and object points list
    for fname in images:
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    #calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def compute_perspective_transform(image, mtx, dist):
    #plot this trapezoid on the image
    bot_left = [100, 700]
    top_left = [550, 460]
    top_right = [732, 460]
    bot_right = [1250, 700]

    undist = cv2.undistort(image, mtx, dist, mtx)

    #Plot the trapezoid on the image
    pts = np.array([bot_left, top_left, top_right, bot_right], np.int32)
    polypts = pts.reshape((-1,1,2))
    image = cv2.polylines(undist,[polypts],True,(0,255,255))

    plt.figure(figsize= (8,5))
    plt.imshow(image)
    plt.title('trapezoidal region to apply perspective transform')

    #Calculate perspective transform
    offset = 100
    width = image.shape[1] - offset
    height = image.shape[0] - offset
    src = np.stack([top_left, top_right, bot_right, bot_left]).astype(np.float32)
    dst = np.float32([ [offset, offset], [offset+width, offset], [offset+width, offset+height], 
                        [offset, offset+height]])
    M = cv2.getPerspectiveTransform(src, dst)

    #Find the inverse of the perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    return (M, Minv)

#To generate calibrations, set to True
gen_calib = False

if gen_calib == True:
    mtx, dist = compute_cam_calib()
    imgfile = "test_images/straight_lines1.jpg"
    image = mpimg.imread(imgfile)
    M, Minv = compute_perspective_transform(image, mtx, dst)
    dict_pickle = { "pM" : M,
                    "pMinv" : Minv,
                    "mtx" : mtx, 
                    "dist" : dist }
    pickle.dump(dict_pickle, open("saved_cals.p", "wb")) 

#Load calibration data from pickle file
cals = pickle.load(open('saved_cals.p','rb'))
mtx = cals["mtx"]
dist = cals["dist"]
perspectiveM = cals["pM"]
perspectiveInvM = cals["pMinv"]

# Define a class to receive the characteristics of each line detection
class LaneTracker():
    def __init__(self, smooth_window = 7):
        #polynomial coefficients averaged over the last n iterations
        self.pastN_left_fits = np.empty ( (0,3), np.float32)
        self.pastN_right_fits = np.empty ( (0,3), np.float32)

        self.current_left_fit = None
        self.current_right_fit = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

        self.smooth_window = smooth_window

        self.threshold_curvature_diff = 0.1

    def sanity_check(self, left_fit, right_fit):
        #Are curvatures similar
        isGood = np.abs( left_fit[0] - right_fit[0] ) < self.threshold_curvature_diff
        #print("step 1: " + str(isGood))
        if isGood:
            #Are they separated by approx 3.7 m
            isGood = np.abs(left_fit[2] - right_fit[2]) < 900
            #print("step 2: " + str(isGood))
            if isGood:
                #Are they roughly parallel
                #measure distances at certain y locations and confirm that they are similar
                ploty = np.array([20, 200, 400, 700])
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

                #Take dist between 2 points at the bottom of the image where it is more reliable
                ref_distance = left_fitx[-1] - right_fitx[-1]
                isGood = np.all( ((left_fitx - right_fitx) - ref_distance) < 300)
                #print("step 3: " + str(isGood))
        return isGood

    def smooth_lane_detection(self, binary_warped, nwindows = 9):
        #Fit lanes
        left_fit, right_fit = fit_lane_poly(binary_warped, 
                                            self.current_left_fit, 
                                            self.current_right_fit, 
                                            nwindows = nwindows)
        isGoodLaneDetection = self.sanity_check(left_fit, right_fit)

        if isGoodLaneDetection:
            self.current_left_fit = np.array(left_fit).reshape(1,3)
            self.current_right_fit = np.array(right_fit).reshape(1,3)

            if (self.pastN_left_fits.shape[0] == self.smooth_window) :
                #shift arrays and add current fit to history
                self.pastN_left_fits = np.vstack( (self.pastN_left_fits[1:], self.current_left_fit))
                self.pastN_right_fits = np.vstack( (self.pastN_right_fits[1:], self.current_right_fit))
                self.current_left_fit = np.average(self.pastN_left_fits, axis = 0)
                self.current_right_fit = np.average(self.pastN_right_fits, axis = 0)
            else:
                #build up past history
                self.pastN_left_fits = np.append(self.pastN_left_fits, self.current_left_fit, axis = 0)
                self.pastN_right_fits = np.append(self.pastN_right_fits, self.current_right_fit, axis = 0)
        
            return self.current_left_fit.flatten(), self.current_right_fit.flatten()
        else:
            if self.pastN_left_fits.shape[0] > 0:
                left_fit = self.pastN_left_fits[-1]
            else:
                left_fit = None
            if self.pastN_right_fits.shape[0] > 0:
                right_fit = self.pastN_right_fits[-1]
            else:
                right_fit = None
            
            return left_fit, right_fit

# Function that selects and thresholds the hsv colorspace
def hsv_select(img, thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    binary_output = np.zeros_like(v)
    binary_output[ (v >= thresh[0]) & (v <= thresh[1]) ] = 1
    return binary_output

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(gray, orient='x', sobel_kernel = 7, thresh = (0, 255)):
    sobelout = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize = sobel_kernel)
    abssobelout = np.abs(sobelout)
    scaled8bitsobel = np.uint8(255 * abssobelout/np.max(abssobelout))
    binary_output = np.zeros_like(sobelout)
    binary_output[ (scaled8bitsobel >= thresh[0]) & (scaled8bitsobel <= thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize = sobel_kernel)
    abs_sobelxy = np.sqrt( sobelx**2 + sobely**2)
    abs_sobelxyscaled = np.uint8( 255 * abs_sobelxy / np.max(abs_sobelxy))
    binary_output = np.zeros_like(abs_sobelxyscaled)
    binary_output[ (abs_sobelxyscaled >= mag_thresh[0]) & (abs_sobelxyscaled <= mag_thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    abs_sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F,1,0,ksize= sobel_kernel))
    abs_sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F,0,1,ksize= sobel_kernel))
    theta = np.abs(np.arctan2(abs_sobely, abs_sobelx))
    binary_output = np.zeros_like(abs_sobelx)
    binary_output[ (theta >= thresh[0]) & (theta <= thresh[1]) ] = 1
    return binary_output

def fit_lane_poly(binary_warped, left_fit, right_fit, nwindows = 9):
    
    if left_fit is None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
   
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = binary_warped.shape[0]//nwindows
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
    
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
    
            # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        if leftx.shape[0] > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = [0, 0, 0]
        if rightx.shape[0] > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = [0, 0, 0]
    else:
        #We already have an established lane fit. So make use of that.
        left_fit = left_fit.flatten()
        right_fit = right_fit.flatten()

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if leftx.shape[0] > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if rightx.shape[0] > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        #else just retain old lane value
    return (left_fit, right_fit)

def compute_cam_center(pts_left, pts_right, image_width):
    #The average of the left & right intercept coefficients is the center
    # Using the 10th value for x to get a point in the middle of the lane
    idx = 200
    lane_middle = int((pts_right[0][idx][0] - pts_left[0][idx][0])/2.)+pts_left[0][idx][0]
    image_half_width = image_width // 2
    laneHalfWidth = 3.7/2 # Assume avg lane width of 3.66 m

    if (lane_middle-image_half_width > 0):
        offset = ((lane_middle-image_half_width)/image_half_width * laneHalfWidth)
        head = ("right", offset)
    else:
        offset = ((lane_middle-image_half_width)/image_half_width * laneHalfWidth)*-1
        head = ("left", offset)
    return head

def radius_of_curvature(leftx, rightx, ploty, ym_per_pix, xm_per_pix):
    # Define y-value where we want radius of curvature
    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval_m = np.max(ploty) * ym_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the radii of curvature in real world units of meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters and will be the average of these two radii
    roc = np.average( [left_curverad, right_curverad])
    return roc

def detect_lanes(image, tracker):
    
    #Undisort image using camera calibration matrix
    undist_image = cv2.undistort(image, mtx, dist, mtx)

    #Threshold image using color and gradients
    hsv_binary = hsv_select(undist_image, thresh=(230, 255))
    gray = cv2.cvtColor(undist_image, cv2.COLOR_RGB2GRAY)
    ksize = 7
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 255))

    #Combine different thresholds
    binary_th = np.zeros_like(hsv_binary)
    binary_th[ ((gradx == 1 & (grady == 1)) | (mag_binary == 1)) & ((hsv_binary == 1)) ] = 1

    #Warp thresholded binary image to do lane detection
    width = gray.shape[1]
    height = gray.shape[0]
    offset = 50
    binary_warped = cv2.warpPerspective(binary_th, perspectiveM, (width + offset, height+offset))
    
    #Fit lanes
    #left_fit, right_fit = fit_lane_poly(binary_warped, nwindows = 9)
    left_fit, right_fit = tracker.smooth_lane_detection(binary_warped)

    if left_fit is not None and right_fit is not None:

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        #Visualize lane marking detections with cv2.fillPoly
        marginpix = 20
        pts_left_lt = left_fitx - marginpix
        pts_left_rt = left_fitx + marginpix
        pts_left_lane = np.array(list(zip(np.concatenate( (pts_left_lt, pts_left_rt[::-1]), axis = 0 ), np.concatenate( (ploty, ploty[::-1]), axis = 0) )))
    
        pts_right_lt = right_fitx - marginpix
        pts_right_rt = right_fitx + marginpix
        pts_right_lane = np.array(list(zip(np.concatenate( (pts_right_lt, pts_right_rt[::-1]), axis = 0 ), np.concatenate( (ploty, ploty[::-1]), axis = 0) )))

        #Empty images to fill in lane marker pixels
        road = color_warp.copy()
        road_bkg = color_warp.copy()

        # Mark left lane a blue color
        cv2.fillPoly(road, np.int_([pts_left_lane]), (0, 0, 255))
        cv2.fillPoly(road_bkg, np.int_([pts_left_lane]), (255, 255, 255))
        # Mark right lane a red color
        cv2.fillPoly(road, np.int_([pts_right_lane]), (255, 0, 0))
        cv2.fillPoly(road_bkg, np.int_([pts_right_lane]), (255, 255, 255))

        # Draw the lane onto the warped blank image in green color
        cv2.fillPoly(road, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        roadwarp = cv2.warpPerspective(road, perspectiveInvM, (image.shape[1], image.shape[0])) 
        road_bkg_warp = cv2.warpPerspective(road_bkg, perspectiveInvM, (image.shape[1], image.shape[0])) 
    
        #Combine the result with the original image
        base = cv2.addWeighted(undist_image, 1, road_bkg_warp, -1.0, 0)
        result = cv2.addWeighted(base, 1, roadwarp, 1.0, 0)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/binary_warped.shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/810. # meters per pixel in x dimension

        position, cam_center_to_mid = compute_cam_center(pts_left, pts_right, binary_warped.shape[1])

        roc = radius_of_curvature(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix)

        font = cv2.FONT_HERSHEY_SIMPLEX
        vehicleTxt = "Vehicle is " + str(cam_center_to_mid) + " m " + "to the " + position + " of center" 
        rocTxt = "Radius of curvature is " + str(roc) + " m"

        cv2.putText(result, vehicleTxt,(50,50), font, 1, (255,255,255), 1)
        cv2.putText(result, rocTxt, (50, 100), font, 1, (255, 255, 255), 1)
    else:
        result = image
    return result

def test_lane_detection():
    images = glob.glob("test_images/*.jpg")
    for imgfile in images:
        print(imgfile)
        image = mpimg.imread(imgfile)
        tracker = LaneTracker()
        outim = detect_lanes(image, tracker)
        outfilename = os.path.join("output_images" , "out_" + os.path.basename(imgfile)) 
        mpimg.imsave(outfilename, outim)

#test_lane_detection()

#run on video
def run_lane_det(videofile):
    cap = cv2.VideoCapture(videofile)
    #test
    #cap.set(1, 900)

    width = int(cap.get(3))
    height = int(cap.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    outfilename = os.path.join("output_images" , "out2_" + os.path.basename(videofile)) 
    out = cv2.VideoWriter(outfilename ,fourcc, 30.0, (width, height))

    tracker = LaneTracker()
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            
            outframe = detect_lanes(frame, tracker)
            out.write(outframe)
            
            frameNum = int(cap.get(1))
    
            cv2.putText(outframe, "frame : " + str(frameNum) ,(50,150), font, 1, (255,255,255), 1)
            cv2.imshow('frame', outframe)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.imwrite("test_images/video" + str(frameNum) + ".jpg", frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

run_lane_det("project_video.mp4")