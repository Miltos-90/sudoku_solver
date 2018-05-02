import numpy as np
from math import floor


class Solve:


    def __init__(self, sudoku):
        self.error = False
        # Determine the number of rows and columns
        n = int(np.sqrt(sudoku.size))
        # Calculate the smallest divisor of the number defined before
        divisor = self.smallestDivisor(n)
        # Check if everything makes sense
        self.validity_check(n, divisor, sudoku)
        if self.error:
            return
        # Translate the sudoku to an exact cover problem
        cells = []; rows = []
        [self.exact_cover(cells, rows, i, j, sudoku, n, 4 * n**2, divisor)
        for i in np.arange(0, n) for j in np.arange(0, n)]
        # And solve it using Knuth's algorithm X
        solution = list(self.alg_X(np.array(rows), cells))
        if solution: # If a solution was found
            solution = solution[0] # Get it
            # And reshape it as the original grid
            self.res = self.reshape_solution(solution, n)
        else:
            self.error = True # No solution found
        
        
    def alg_X(self, A, X, solution = [], r=[], c=[]):
        # Applies aglorithm X as explained in Wikipedia
        # Determine which columns to remove
        cols_to_remove = [idx for idx, elem in enumerate(A[r]) if elem == 1] 
        # Determine which rows to remove
        rows_to_remove = set([idx for col in cols_to_remove for idx, elem 
                              in enumerate(A.T[col]) if elem == 1])
        # Remove rows from matrix A
        A = np.delete(A, (list(rows_to_remove)), axis = 0)
        # Remove cols from matrix A
        A = np.delete(A, (cols_to_remove), axis = 1)
        # Determine which rows to remove for the next iteration
        rows_to_remove = [X[elem] for elem in rows_to_remove]
        X = [elem for elem in X if elem not in rows_to_remove]
        if A.size == 0: # Solution found
            yield list(solution)
        else: # keep iterating
            # Determine which columns to use for the next iteration
            c = A.sum(0).argmin() 
            # Same for rows
            rows = [idx for idx, elem in enumerate(A.T[c]) if elem == 1]
            for r in rows: # Apply algorithm X recursively
                solution.append(X[r])
                for s in self.alg_X(A, X, solution, r, c): yield s
                solution.pop()
    
    
    @staticmethod
    def smallestDivisor(n): 
        # Returns the smallest divisor of the number of rows and columns
        if n % 2 == 0: return 2
        else:
            squareRootOfn = floor(np.sqrt(n))
            for i in (3,squareRootOfn,2):
                if n % i == 0: 
                    return i
                elif i == squareRootOfn: 
                    return 1
    
    
    @staticmethod
    def get_block(i, j, smallest_divisor, n):
        # Return the number of the block of i'th row and j'th column
        row = i // smallest_divisor
        column = j // smallest_divisor
        return int(row * np.sqrt(n) + column)
    
    
    def create_row(self, col_number, i, j, n, cell, n_smallest_divisor):
        # Creates one row for the binary matrix on which alg_X will be applied
        # build up a row for this cell
        row = np.zeros((col_number,), dtype = int) 
        # cell constraint
        row[i * n + j] = 1 
        # row constraint
        row[n**2 + i * n + cell - 1] = 1 
        # column constraint
        row[2 * n**2 + j * n + cell - 1] = 1 
        # Box constraints
        block = self.get_block(i, j, n_smallest_divisor, n)
        row[3 * n**2 +  block * n + (cell - 1)] = 1
        return row
        
        
    @staticmethod
    def define_cell(i, j, cell):
        # Returns a description for each cell 
        return 'R' + str(i) + 'C' + str(j) + '#' + str(cell)
    
    
    def exact_cover(self, cells, rows, i, j, sudoku, n, total_cols, n_smallest_divisor):
        # Returns a set of rows for each sudoku cell for the binary matrix on which alg_X will be applied
        if sudoku[i][j] != 0: # clue
            cells.append(self.define_cell(i, j, sudoku[i][j]))
            rows.append(self.create_row(total_cols, i, j, n, sudoku[i][j], n_smallest_divisor))
        else:
            [(cells.append(self.define_cell(i, j, candidate)),
              rows.append(self.create_row(total_cols, i, j, n, candidate, n_smallest_divisor)))
            for candidate in np.arange(1, n + 1)]    
    
    
    @staticmethod        
    def blocks(arr, nrows, ncols):
        # Returns the contents of each block of the sudoku as a vector
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))
            
            
    def check_dupes(self, array):
        # Check an array for duplicates
        array = array[~(array == 0)] # Remove zero elements
        if len(array) != len(set(array)): # contents not unique
            self.error = True # Found duplicates
                

    def validity_check(self, n, divisor, sudoku):
        # Checks if sudoku contents are valid
        # Invalid grid
        if sudoku.shape[0] != sudoku.shape[1]:
            self.error = True # Irregular grid dimensions
            return
        # Duplicate numbers
        for axis in [0, 1]:
            np.apply_along_axis(self.check_dupes, axis, sudoku)
        for block in self.blocks(sudoku, divisor, divisor):
            self.check_dupes(block)
                
            
    @staticmethod
    def reshape_solution(array, n):
        # Utility function to return the solution with the same dimensions
        # as the matrix used as input
        array.sort()
        array = np.array([elem.split('#')[1] for elem in array])
        return array.reshape((n, n)).astype(int)
    