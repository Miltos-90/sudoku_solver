import cv2 as cv
from helpers.grabber import Grab
from helpers.recognizer import Digit_Recognizer
from helpers.solver import Solve
from helpers.plotter import Plot


if __name__=="__main__":
    
    model = Digit_Recognizer()
    cap = cv.VideoCapture(0)
    
    while(True):
        if cv.waitKey(100) & 0xFF == ord('q'): 
            break 
        _, frame = cap.read() # Capture frame-by-frame
        cv.imshow("original", frame)
        sudoku_cap = Grab(frame)
        if not sudoku_cap.error:
            sudoku = model.predict(sudoku_cap)
            if not model.error:
                solution = Solve(sudoku)
                if not solution.error:
                    final = Plot(sudoku_cap, solution, sudoku)
                
    cap.release()
    cv.destroyAllWindows()