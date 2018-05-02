from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os 
import pandas as pd
import numpy as np

font_path = "C:\Windows\Fonts"

# Image dimensions
W, H = (28, 28) 
fonts = [font for font in os.listdir(font_path) if '.ttf' 
         in font and font <= 'himalaya.ttf']
# after himalaya.ttf the numerics get messed up

# Initialize list to store the data
data = []

for font_type in fonts:
    for number in range(0,10):
        # Draw a new image
        img = Image.new('L', (W, H))
        draw = ImageDraw.Draw(img)
        
        # Number to be drawn
        msg = str(number)
        
        # Set the font
        font = ImageFont.truetype(font_type, 28)
        
        # Get the dimensions of the number to be drawn
        w, h = draw.textsize(msg, font = font)
        
        # Draw it in the center
        draw.text(((W-w) / 2,(H-h) / 2), msg, font = font, fill = "white")
        '''
        # Display results
        plt.imshow(np.array(img))
        plt.title(font_type)
        plt.show() 
        '''
        # Convert image to an int numpy array
        img = np.array(img).astype(int).flatten()
        img = np.insert(img, 0, number)
        # Append to list, after appending the label
        data.append(img)
        
# Create the dataframe that will hold the images
# Same as MNIST dataset
columns=['pixel' + str(pixel) for pixel in range (0, W * H)]
columns = ['label'] + columns
df = pd.DataFrame(columns = columns, data = data)
df.to_csv('not_mnist.csv', index = False)