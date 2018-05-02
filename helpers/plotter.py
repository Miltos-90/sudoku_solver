import numpy as np
import cv2 as cv

class Plot:
    
    def __init__(self, img, solution, sudoku, font = cv.FONT_HERSHEY_DUPLEX):
        img = img.perspective_frame.copy()
        for i in range(solution.res.shape[0]): # For every line of the solution mtrx
            # Subtract input & output, to identify only the chars to be printed
            # (the non-zero elements of the new array)
            text = solution.res[i, :] - sudoku[i, :]
            # Strip the string of the beginning and ending []
            # Replace zero elements with blank spaces
            text = np.array_str(text).strip('[]').replace(str(0), ' ')
            # Get textsize
            textsize = cv.getTextSize(text, font, 1, 2)[0]
            # Identify the appropriate font scale so that the text fits nicely to the img
            # (text_width / image_width = 0.95)
            font_scale = 0.95 * img.shape[1] / textsize[0]
            # Put the text in the appropriate line 
            textY = (i + 1) * (img.shape[0] + textsize[1]) // (solution.res.shape[0] + 1)
            # with a 5% space from the left border
            textX = int(0.05 * img.shape[1]) // 2 
            # add text centered on image
            cv.putText(img, text, (textX, textY ), font, font_scale, (0, 255, 0), 2)
        cv.imshow('solution', img)
