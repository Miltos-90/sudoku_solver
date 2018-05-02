import numpy as np
import cv2 as cv


class Grab:
    
    
    def __init__(self, frame):
        self.error = False # Hopefully this will remain as it is
        # Preprocess the frame of the videocam
        img = self.videoframe_processing(frame)
        # Extract the portion of the frame that contains the sudoku
        sudoku_frame = self.max_contour_extraction(img)
        # If the frame is not a rectangle, exit and retry (through main)
        if sudoku_frame.shape != (4, 1, 2):
            self.error = True # Error 
            return
        # Get the coordinates of the frame
        rect = self.grid_coordinates(sudoku_frame)
        # Apply a perspective transformation on the frame
        mtrx, width, height = self.topdown_transform(rect)
        # Extract the inner grid of the sudoku
        img = self.grid_extraction(frame, rect, mtrx, width, height)
        self.perspective_frame = img
        self.perspective_transform = (rect, mtrx, width, height)
        # Preprocess the extracted grid
        img = self.grid_preprocessing(img)
        # Extract images that contain only horizontal and vertical lines
        horiz_lines, vert_lines = [self.gridline_extraction(img, linetype) for
                                   linetype in ["horizontal", "vertical]"]]
        # Count the number of lines in each image
        grid_dims = self.grid_dimensions(horiz_lines, vert_lines)
        if grid_dims[0] != grid_dims[1]: # Error reading the grid
            self.error = True
            return
        # Remove the inner grid and denoise the image
        img = self.denoising(img - horiz_lines - vert_lines)
        # Segment the image into cells
        cells = self.cell_segmentation(img, grid_dims)
        if not all([cell.shape >= (28,28) for cell in cells]):
            self.error = True
            return
        try: # When grid is moving too fast, cv.moments return zeros (which I divide with later on)
            # Normalize and center the contents of each cell (to be fed to the ConvNet)
            self.cells = [cell if cv.countNonZero(cell) == 0 else 
                     self.cell_normalization(self.cell_centering(cell)) for cell in cells]
        except (IndexError, ZeroDivisionError): # Error due to cv.moments zeros
            self.error = True
            return
        # All correct, return the image, and dimensions
        self.frame = img
        self.dims = grid_dims


    @staticmethod
    def videoframe_processing(frame):
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Grayscale Conversion
        img = cv.GaussianBlur(img, (11, 11), 3) # Gaussian blur filtering
        # Adaptive Mean Thresholding
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                    cv.THRESH_BINARY_INV, 11, 2)
        return img


    @staticmethod 
    def max_contour_extraction(img):
        # Contour Extraction
        _, contours ,_  = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = 0
        for idx, contour in enumerate(contours):
            # approximate each contour with a polygon
            epsilon = 0.02*cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            # and keep the one wit the maximum area
            area = cv.contourArea(approx)
            if area > max_area:
                max_contour = approx
                max_area = area
        return max_contour


    @staticmethod 
    def grid_coordinates(sudoku_frame):
        # Create a rectagle to store the grid's coordinates
        rect = np.zeros((4, 2), dtype = "float32")
        points = [elem[0] for elem in sudoku_frame]
        # the first entry in the list is the top-left point
        rect[0] = points.pop(np.argmin(np.apply_along_axis(np.linalg.norm, axis = 1, arr = points)))
        #the third entry is the bottom right corner
        rect[2] = points.pop(np.argmax(np.apply_along_axis(np.linalg.norm, axis = 1, arr = points)))
        # the second entry is the top right corner, and the fourth is the bottom-left
        rect[1], rect[3] = sorted(points, key = lambda elem: elem[0], reverse = True)
        return rect


    @staticmethod 
    def topdown_transform(inp_mtrx):
        # Get the maximum width of the sudoku
        width = max(np.linalg.norm(inp_mtrx[2] - inp_mtrx[3]), # bottom width
                    np.linalg.norm(inp_mtrx[1] - inp_mtrx[0])) # top width
        # Get the maximum height
        height = max(np.linalg.norm(inp_mtrx[1] - inp_mtrx[2]), # left height
                     np.linalg.norm(inp_mtrx[0] - inp_mtrx[3])) # right height
        # top-down view of the new image
        outpt_mtrx = np.array([[0, 0], # top left corner 
                        [width - 1, 0], # top right corner 
                        [width - 1, height - 1], # bottom right corner
                        [0, height - 1]],  # bottom left corner
                        dtype = "float32")
        return outpt_mtrx, width, height
    
    
    @staticmethod 
    def grid_extraction(frame, source_coords, dst_coords, width, height):
        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(source_coords, dst_coords)
        img = cv.warpPerspective(frame, M, (width, height))
        return img


    @staticmethod 
    def grid_preprocessing(src_img):
        # Apply adaptiveThreshold at the bitwise_not of gray
        gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
        gray_img = cv.bitwise_not(gray_img)
        dst_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                        cv.THRESH_BINARY, 15, -2)
        return dst_img


    @staticmethod 
    def gridline_extraction(src_img, linetype):
        # Create the images that will use to extract the horizontal and vertical lines
        dst_img = src_img.copy()
        # Specify size on horizontal axis
        if linetype == "horizontal":
            elem_size = (dst_img.shape[1] // 17, 1)
        else:
            elem_size = (1, dst_img.shape[0] // 17)
        # Create structure element for extracting horizontal lines through morphology operations
        structure = cv.getStructuringElement(cv.MORPH_RECT, elem_size)
        # Apply morphology operations
        dst_img = cv.erode(dst_img, structure, 1)
        dst_img = cv.dilate(dst_img, structure, 1)
        return dst_img


    @staticmethod 
    def grid_dimensions(horz_grid, vert_grid):
        # Make a copy of the horizontal grid and get its dimensions
        temp_img = horz_grid.copy()
        dimensions = temp_img.shape
        # Fill outtop and bottom gridlines
        temp_img[0:2, :] = 255
        temp_img[dimensions[0] - 3: dimensions[0] - 1,:] = 255
        # Find contours
        _, contours, _  = cv.findContours(temp_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        rows = len(contours) - 1
        # Make a copy of the vertical grid and get its dimensions
        temp_img = vert_grid.copy()
        dimensions = temp_img.shape
        # Fill leftmost and rightmost columns
        temp_img[:, 0:2] = 255
        temp_img[:, dimensions[1] - 3: dimensions[1] - 1] = 255
        # Find contours
        _, contours, _  = cv.findContours(temp_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cols = len(contours) - 1
        # Return dimensions
        return (rows, cols)


    @staticmethod 
    def denoising(src_img):
        # Find all connected components: (white blobs in the image)
        # yield every seperated component with information on each of them, such as size
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(src_img, connectivity = 8)
        # Take out the background (it is considered a component as well)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        # Define minimum size of elements to keep
        min_size = min(src_img.shape) // 7
        # Initialize the destination image
        dst_img = np.zeros((output.shape)) 
        # keep a component only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                dst_img[output == i + 1] = 255
        return dst_img


    @staticmethod 
    def cell_segmentation(img, dims):
        # Break down the warped image in segments
        cells = []
        # Horizontal direction
        for i in range(1, dims[0] + 1):
            xmax = img.shape[0] * (i) // dims[0]
            if i == 1: 
                xmin = 0 # First cell on x-axis
            else: 
                xmin = img.shape[0] * (i-1) // dims[0]
                # Vertical direction
            for j in range(1, dims[1] + 1):
                if j == 1: 
                    ymin = 0 # First cell on y-axis
                else: 
                    ymin = img.shape[1] * (j-1) // dims[1]
                ymax = img.shape[1] * j // dims[1]
                cells.append(img[xmin:xmax, ymin:ymax])
        return cells


    @staticmethod 
    def centerpoint_extraction(src_img):
        # Blur and threshold the image to create one blob
        img = cv.GaussianBlur(src_img, (7, 7), 11); 
        _, img = cv.threshold(img, 10, 255, cv.THRESH_BINARY)
        # Scale, calculate absolute values, and convert the result to 8-bit.
        img = cv.convertScaleAbs(img) 
        # Extract contour of the blob (only one at this point)
        _, contour, _  = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Get the moments of the contour
        M = cv.moments(contour[0])
        # Deterrmine the centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)
    
    
    def cell_centering(self, src_img):
        # Determine the centerpoint of the image
        cx, cy = self.centerpoint_extraction(src_img)
        # Get the dimensions
        width, height = src_img.shape
        # Define the translation matrix
        M = np.float32([[1, 0, height // 2 - cx],[0,1, width // 2 - cy]])
        # And apply it
        dst_img = cv.warpAffine(src_img, M, (width, height))
        # Crop the surrounding blank space to produce a 28 x 28 pixel image
        return dst_img[(width - 28) // 2 : (width + 28) // 2, 
                       (height - 28) // 2 : (height + 28) // 2]


    @staticmethod 
    def cell_normalization(src_img):
        # Convert each image's pixels to 0 and 1 (to comply with the ConvNet std input)
        return src_img / 255.0
    

