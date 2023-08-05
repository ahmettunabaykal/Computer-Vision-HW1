
import math

def getNeighbor(coords, binaryImage, struct):
    """
    Given the coordinates of a pixel in a binaryImage matrix and a structure matrix,
    returns the coordinates of the neighboring pixels that are non-zero in the binaryImage matrix.
    The struct matrix defines the neighborhood shape and size.
    """
    # Get the dimensions of the binaryImage and the struct matrix
    m, n = binaryImage.shape
    k, l = struct.shape

    # Compute the center pixel of the struct matrix
    center = (k // 2, l // 2)

    # Initialize a list to store the neighboring pixel coordinates
    neighbors = []

    # Iterate over the struct matrix
    for i in range(k):
        for j in range(l):
            # If the struct matrix has a non-zero value at this position
            if struct[i, j] != 0:
                # Compute the corresponding position in the binaryImage matrix
                x = coords[0] + i - center[0]
                y = coords[1] + j - center[1]

                # Check if the corresponding position is within the binaryImage matrix bounds
                if x >= 0 and x < m and y >= 0 and y < n:
                    # Check if the corresponding position is non-zero in the binaryImage matrix
                    if binaryImage[x, y] != 0:
                        # Add the corresponding position to the neighbors list
                        neighbors.append((x, y))

    return neighbors


def dilation(binaryImage, structure):
    # boundaries of image width length
    image_x , image_y = binaryImage.shape
    
    # create empty matrix same sizes with binaryImage
    dilatedImage = np.zeros((image_x, image_y))

    #boundaries of structure matrix width length
    structure_x ,structure_y = structure.shape

    for i in range(image_x):
        for j in range(image_y):

            pixelValue = 1
            for k in range(structure_x):
                for l in range(structure_y):

                    binaryX = i + k - math.floor(structure_x/2)
                    binaryY = j + l - math.floor(structure_y/2)

                    if binaryX >= 0 and binaryX < image_x and binaryY >= 0 and binaryY < image_y and binaryImage[binaryX, binaryY] == 0:
                        # Set the maximum pixel value to 1
                        pixelValue = 0
                        break

            dilatedImage[i, j] = pixelValue
    
    return dilatedImage

def erosion(image, kernel):
    kernel_x, kernel_y = kernel.shape
    kernel_center_x, kernel_center_y = int(kernel_x / 2), int(kernel_y / 2) # index, not width
    image_x, image_y = image.shape
    output = np.zeros_like(image)

    kernel_points = []
    for x in range(kernel_x):
        for y in range(kernel_y):
            if kernel[x][y] == 1:
                kernel_points.append((x - kernel_center_x, y - kernel_center_y))

    for i in range(image_x):
        for j in range(image_y):
            flag = False
            for k in range(len(kernel_points)):
                offset_x, offset_y = kernel_points[k][0], kernel_points[k][1]
                neighbor_x, neighbor_y = i + offset_x, j + offset_y
                if (0 <= neighbor_x < image_x) and (0 <= neighbor_y < image_y):
                    if image[neighbor_x][neighbor_y] != 0:
                        flag = True
                        break
            if flag == False:
                output[i][j] = 1

    # Invert the output and scale it to the input data type range
    output = np.logical_not(output).astype(image.dtype) * np.iinfo(image.dtype).max

    return output


def histogram(image):
    hist = np.zeros((256, 2), dtype=int)
    image_width, image_length = image.shape
    
    for i in range(image_width):
        for j in range(image_length):
            pixel_value = int(image[i,j])
            hist[pixel_value, 0] = pixel_value
            hist[pixel_value, 1] += 1
    return hist

def otsu_threshold(image):

    hist = histogram(image) # compute the histogram written before

    
    width,length = image.shape # find total number of pixels in the image
    pixels = width * length
    max_pixel_value = 256

    threshold = 0
    max_var = 0
    
    # Iterate over all possible threshold values
    for t in range(max_pixel_value):
        
        w0 = np.sum(hist[:t, 1]) / pixels
        w1 = 1 - w0
        
        if np.sum(hist[:t, 1]) > 0:
            m0 = np.sum(hist[:t, 0] * hist[:t, 1]) / (w0 * pixels)
        else:
            m0 = 0
        if np.sum(hist[t:, 1]) > 0:
            m1 = np.sum(hist[t:, 0] * hist[t:, 1]) / (w1 * pixels)
        else:
            m1 = 0
        
        # Compute variance
        var = w0 * w1 * (m0 - m1) ** 2
        
        if var > max_var:
            threshold = t
            max_var = var
    
  
    #create empty 2d array same sizes as image provided
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #if pixel is smaller than threshold make it black 
            if image[i,j] <= threshold:
                output[i,j] = 0
                #else make it white
            else:
                output[i,j] = 255


    return output



def convolution(image, structure):

    width, length = image.shape
    kx, ky = structure.shape
    
    
    pad_x, pad_y = kx // 2, ky // 2
    
    # Pad the image with zeros
    padded = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    
    # Initialize output image
    out = np.zeros((width, length))
    
    # Iterate through each pixel of the image
    for i in range(pad_x, width + pad_x):
        for j in range(pad_y, length + pad_y):
            # Define the row and column indices of the top-left corner of the edge
            row_start = i - pad_x
            col_start = j - pad_y

            # Initialize an empty edge array
            edge = np.zeros((kx, ky))

            # Fill the edge array with values from the padded input image
            for row in range(kx):
                for col in range(ky):
                    # Calculate the row and column indices in the padded image
                    padded_row = row_start + row
                    padded_col = col_start + col

                    # Extract the value from the padded image and add it to the edge array
                    edge[row, col] = padded[padded_row, padded_col]

            # Perform the convolution
            out[i - pad_x, j - pad_y] = np.sum(edge * structure)
    
    return out


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cycler import cycler


################ question 1 ##################
image = Image.open('Figure1.png').convert('L')
binaryImage = np.array(image)

# Define the structuring element
struct = np.array([[0, 0, 0 ],
                   [0, 1, 0 ] ,
                   [0, 0, 0 ]])

# Perform dilation
dilated = dilation(binaryImage, struct)
eroded = erosion(binaryImage,struct)

 #Save the result as a new image
result = Image.fromarray(dilated * 255).convert('P')
result1 = Image.fromarray(eroded).convert('L')
result.save('Question1a.png')
result1.save('Question1b.png')

############### question 2 ####################
#

img = cv2.imread('Figure2_a.jpg', 0)
# Calculate histogram using my function
hist = histogram(img)
plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b'])
# Plot histogram
plt.bar(hist[:, 0], hist[:, 1], color='b')
plt.title('Histogram of Grayscale Image Figure2_a')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
plt.figure
plt.show()
# Plot histogram

img1 = cv2.imread('Figure2_b.jpg', 0)
hist2=histogram(img1)
plt.bar(hist2[:, 0], hist2[:, 1], color='b')
plt.title('Histogram of Grayscale Image Figure2_b')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
plt.figure
plt.show()



# Calculation with the matlib library
#img1 = cv2.imread('Figure2_b.jpg', 0)

# Calculate histogram
#hist, bins = np.histogram(img1.ravel(), 256, [0, 256])

 #Plot histogram
#plt.hist(img1.ravel(), 256, [0, 256], color='b')
#plt.title('Histogram1 of Grayscale Image')
#plt.xlabel('Intensity Value')
#plt.ylabel('Pixel Count')
#plt.show()

############### question 3 ##################

image1 = Image.open('Figure3_a.jpg').convert('L')
binaryImage1 = np.array(image1)
otsu_thresholdImage = otsu_threshold(binaryImage1)
result3 = Image.fromarray(otsu_thresholdImage).convert('L')
result3.save('Question3a.png')


image2 = Image.open('Figure3_b.png').convert('L')
binaryImage2 = np.array(image2)
otsu_thresholdImage = otsu_threshold(binaryImage2)
result4 = Image.fromarray(otsu_thresholdImage).convert('L')
result4.save('Question3b.png')

############# question4 #############
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Sobel operator for vertical edges
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Prewitt operator for horizontal edges
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

# Prewitt operator for vertical edges
prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

image4 = Image.open('Figure4.jpg').convert('L')
binaryImage4 = np.array(image4)
conv1 = convolution(binaryImage4 , sobel_x)
conv2 = convolution(binaryImage4 , sobel_y)
conv3 = convolution(binaryImage4 , prewitt_x)
conv4 = convolution(binaryImage4 , prewitt_y)
result5 = Image.fromarray(conv1).convert('L')

result6 = Image.fromarray(conv2).convert('L')

result7 = Image.fromarray(conv3).convert('L')

result8 = Image.fromarray(conv4).convert('L')
result5.save('Question4aSobel_x.png')

result6.save('Question4bSobel_y.png')

result7.save('Question4cPrewitt_x.png')

result8.save('Question4dPrewitt_y.png')