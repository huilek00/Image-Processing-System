"""
This is a program about the application of linear algebra in image processing.
The program is separated into two parts:
1st part: Explanation of matrix application in image processing system
2nd part: Show the output of image processing
 
Please download the following 4 images which are included in the zip file and 
ensure that they are saved in the SAME FILE before running the program. Thanks.
image 1: "brown.png"
image 2: "insta.png"
image 3: "lena.png"
image 4: "receipt.png" 
"""

import numpy as np
from numpy import array
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

#***********************************************************************
# Function definition to get the choice from user at the beginning     *
#***********************************************************************
def getChoice(numChoice):
    
    # input the user's choice, validate the input
    validInput = False
    while not validInput:
        try:
            choice = int(input('Please choose the number: '))
            if (choice > 0 and choice <= numChoice):
                validInput = True
            else:
                print("Invalid input. You can only enter 1 to",numChoice, "only.")
        except ValueError:
            print("This is not an integer. Please try again. ")
    
    # return the user's choice
    return choice 

#*************************************************************************
# Function definition to display the message to let user choose different*
# matrix application in image processing                                 *
#*************************************************************************
def displayMessage_application():
    
    # display the message to let user choose different matrix application
    print('1. Addition')
    print('2. Subtraction')
    print('3. Multiplication')
    print('4. Gray scale image')
    print('5. Enlargement')
    print('6. Cropping of image')
    print('7. Rotation')
    
    return

#***********************************************************************
# Function definition for EXPLANATION of ADDITION of two images        *
#***********************************************************************
def addition_explanation():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF ADDITION OF TWO IMAGES')
    print('\t\t', '-' * 60)
    
    # initialize the matrix of image1 and image2
    array_exp1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    array_exp2 = np.array([[10,11,12],[13,14,15],[16,17,18]])
    
    print('An image is made up of a lot of pixels which can be represented in')
    print('the form of matrix. So, how does addition of two images works?')
    
    # display the matrix of image1 and image2
    print('\nExample of addition of two images: ')
    print("Matrix of image 1 :\n")
    print(array_exp1)
    print("\nMatrix of image 2 :\n")
    print(array_exp2)
    
    # calculate and display addition of both matrix
    print("\nSum of the both matrix is :-\n")
    print(array_exp1)
    print(" + ")
    print(array_exp2)
    print(" = ")
    array_add = array_exp1 + array_exp2
    print(array_add)
    print('\nHence, the addition of matrix represents a new image which is the') 
    print('combination of both images.')    
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for EXPLANATION of SUBTRACTION of two images     *
#***********************************************************************
def subtraction_explanation():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF SUBTRACTION OF TWO IMAGES')
    print('\t\t', '-' * 60)
    
    # initialize the matrix of image1 and image2
    array_exp1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    array_exp2 = np.array([[10,11,12],[13,14,15],[16,17,18]])
    
    print('An image is made up of a lot of pixels which can be represented in')
    print('the form of matrix. So, how does subtraction of two images works?')
    
    # display the matrix of image1 and image2
    print('\nExample of subtraction of two images: ')
    print("Matrix of image 1 :\n")
    print(array_exp1)
    print("\nMatrix of image 2 :\n")
    print(array_exp2)
    
    # calculate and display subtraction of both matrix
    print("\nDifference of the both matrix is :-\n")
    print(array_exp1)
    print(" - ")
    print(array_exp2)
    print(" = ")
    array_sub = array_exp1 - array_exp2
    print(array_sub)
    print('\nHence, the matrix represents the subtraction of image 2 from image 1,')
    print('which will form a new image.')
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for EXPLANATION of MULTIPLICATION of two images  *
#***********************************************************************
def multiplication_explanation():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF MULTIPLICATION OF TWO IMAGES')
    print('\t\t', '-' * 60)
    
    # initialize the matrix of image1 and image2
    array_exp1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    array_exp2 = np.array([[10,11,12],[13,14,15],[16,17,18]])
    
    print('An image is made up of a lot of pixels which can be represented in')
    print('the form of matrix. So, how does multiplication of two images works?')
    
    # display the matrix of image1 and image2
    print('\nExample of multiplication of two images: ')
    print("Matrix of image 1 :\n")
    print(array_exp1)
    print("\nMatrix of image 2 :\n")
    print(array_exp2)
    
    # calculate and display multiplication of both matrix
    print("\nMultiplication of the both matrix is :-\n")
    print(array_exp1)
    print(" x ")
    print(array_exp2)
    print(" = ")
    array_mul = array_exp1 * array_exp2
    print(array_mul)
    print('\nHence, the matrix represents the multiplication of both images,')
    print('which will form a new image.') 
    print('-' * 77)
    
    return

#*************************************************************************
# Function definition for EXPLANATION of converting an image to GRAYSCALE*
#*************************************************************************
def grayscale_explanation():
    
    # display the message for explanation of converting an image to grayscale
    print('\n\t\t', '-' * 60)
    print('\t\t\t\tEXPLANATION OF GRAYSCALE CONVERSION OF AN IMAGE')
    print('\t\t', '-' * 60)
    print('''
- An image consists of many pixels. A colour image is a combination of 3 
  matrices, which are red (R), green (G), blue (B).  
- In the RGB, a pixel is represented as a tri-dimensional vector (r,g,b),
  where each pixel contains different value of R, G, B that varies from 0 
  to 255. 
- Different combination of RGB will produce different colours.
- To convert a colour image into grayscale, the components of each new 
  pixel is obtained by calculating the average value of RGB and assign it 
  back to the RGB value of pixel. Let's look at the example!

Before conversion to grayscale image:
    
    R ≠ G ≠ B ---> The different values of RGB in the tri-dimensional
                   vector will produce different colour in the image.
                    
Matrix formula to convert an image to grayscale:
    
    | R' |        | R |
    | G' |  = T x | G |   , where T is a transformation matrix.  
    | B' |        | B |
    
                                         | 1/3  1/3  1/3 |
    To get the average value of RGB, T = | 1/3  1/3  1/3 |
                                         | 1/3  1/3  1/3 |
                                         
According to the formula, we will get:
    
    R' = (R + G + B) / 3 
    G' = (R + G + B) / 3 
    B' = (R + G + B) / 3
    
Let's look at the changes of values of RGB by using a simple 3 x 1 matrix!
(tri-dimensional vector)  
        ''')
        
    # initialize value of RGB (an example)
    RGB_before = np.array([[78],
                           [162],
                           [234]])
    
    # define matrix T
    T = np.array([[1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3]])
    
    # find the product of T and RGB
    RGB_after = T @ RGB_before
    
    # display the result
    print('Initial value of RGB: ')
    print(RGB_before)
    print('\nRGB value after multiplying with matrix T (grayscale conversion):')
    print(RGB_after)
    print('''
---> We can see that the value of R = G = B after assigning the 
     average value of RGB into each entries of tri-dimensional vector.
---> Since the RGB value of the pixel is the same, it will not produce
     any colour because the ratio of R:G:B is 1:1:1, hence the image 
     will become grayscale.
        ''')
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for EXPLANATION of ENLARGEMENT of an image       *
#***********************************************************************
def enlargement_explanation():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF ENLARGEMENT OF AN IMAGE')
    print('\t\t', '-' * 60)
    
    print('An image can be enlarged by multiplying its dimension with a scale factor.')
    # initialize the coordinates of both shapes using coord
    coord = [[1,1],[1,3],[3,3],[3,1]]
    coord1 =[[2,2],[2,6],[6,6],[6,2]]
    
    #repeat the first point to create a 'closed loop'
    coord.append(coord[0]) 
    coord1.append(coord1[0])

    #create lists of x and y values
    x1, y1 = zip(*coord)  
    x2, y2 = zip(*coord1) 

    # plot both shapes into the graphs
    figure, axis = plt.subplots(2)
    plt.figure()
    axis[0].plot(x1,y1)
    axis[1].plot(x2,y2)
    plt.show() 
    
    print('\nScale factor = 2')
    print('Original length = 3, width = 3')
    print('Enlarged length = 6, width = 6')
    print('\nFrom the graph, we can see that the second graph is the enlargement') 
    print('of the first graph with a scale factor of 2.\n')
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for EXPLANATION of CROPPING an image             *
#***********************************************************************
def cropping_explanation():
    
    # display the message of explanation of cropping an image
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF CROPPING OF IMAGE')
    print('\t\t', '-' * 60)
    print('''
To crop an image, we have to remove the unwanted pixel.

For example, we have a square image with size of m x m, if we need to crop
only the centre of image with n x n pixel, we have to find the starting point 
of the cropped image first. 

Formula to find the starting point of cropped image:
    starting value = (m - n) / 2
    (noted that the starting value of x = starting value of y)
    
If m = 20 and n = 8,
    starting value = (20 - 8) / 2
                   = 6
    So, the starting point coordinate of cropped image will be (6,6),
    up to 6 pixel in the x direction and 6 pixel of y direction    
    ''')
    
    # plot the graph to show the cropped image using matplotlib
    plt.axes()
    rectangle = plt.Rectangle((6,6), 12, 12, fc='green',ec="black")
    plt.gca().add_patch(rectangle)
    plt.axis('scaled')                      
    plt.grid()                              # add grid to the graph
    plt.xlabel("x")                         # add label to x-axis
    plt.ylabel("y")                         # add label to y-axis
    plt.title("Cropped image dimension")    # add title to graph
    plt.xlim(0, 20)                         # set the range of x axis
    plt.ylim(0, 20)                         # set the range of y-axis
    plt.show()                              # show the plot
    
    print('''
Initially, the starting coordinate (left bottom point) of the image is (0,0).
\nAfter cropping of image, from the diagram, we can see that:
\nStarting coordinate = (6,6)
\nDimension = 12 x 12 (smaller compared to original dimension, 20 x 20)
    ''')
    print('-' * 77)

    return

#***********************************************************************
# Function definition for EXPLANATION of ROTATION of two images        *
#***********************************************************************
def rotation_explanation():
    
    # display the message for explanation of rotation of an image
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tEXPLANATION OF ROTATION OF AN IMAGE')
    print('\t\t', '-' * 60)
    print('''
Each pixel of an image has a coordinate (x,y) that shows their position from
origin. To rotate an image, the new location of each pixel is obtained by 
using this transformation matrix formula:
    
    | x2 | = |  cos𝜃   sin𝜃 |  | x |
    | y2 |   | -sin𝜃   cos𝜃 |  | y |

Let's see what will happen if we substitute the coordinate of a single pixel 
into the formula!''')
    
    # to demonstate the rotation of image by applying the formula above, 
    # initialize the coordinate (x,y) to (4,2) and the angle of rotation is 90
         
    coordinate_before = np.array([[4],  # create and initialize a 2x1 matrix 
                                  [2]]) # for coordinate (x,y)
    angle = 90                          # initialize the angle to 90 degrees
    radian = np.radians(angle)          # convert the angle into radians
    cosine = np.cos(radian)             # find the cosine of angle      
    sine = np.sin(radian)               # find the sine of angle
    
    # create the 2x2 transformation matrix T and initialize the each entries 
    # in the matrix to cosine or sine accordingly
    T = np.array([[cosine, sine],
                  [-1 * sine, cosine]])
    
    # find the coordinate after rotation,(x2,y2) by multiplying coordinate 
    # (x,y) with transformation matrix T
    coordinate_after = T @ coordinate_before
    
    # display the two points in a graph to show the rotation of a point
    x_coordinates = [coordinate_before[0], coordinate_after[0]]
    y_coordinates = [coordinate_before[1], coordinate_after[1]]
    plt.scatter(x_coordinates, y_coordinates)
    plt.annotate("\tBefore rotation",(coordinate_before[0], coordinate_after[0]))
    plt.annotate("\tAfter rotation",(coordinate_before[1], coordinate_after[1]))
    plt.title("Rotation of a point")
    plt.axhline()
    plt.axvline()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-2, 8)
    plt.ylim(-6, 4)
    plt.grid()
    plt.show()
    
    # display the message about rotation of the point
    print('\nCoordinate (x,y) before rotation of image:', end="")
    print('(', coordinate_before[0], ',', coordinate_before[1], ')')
    print('\nAngle of rotation:', angle)
    print('\nCentre of rotation: (0,0)')
    print('\nCoordinate (x2,y2) after rotation of image:', end="")
    print('(', coordinate_after[0], ',', coordinate_after[1], ')')
    print('\nTherefore, from the graph, we can see that the initial point has rotated')
    print(angle, 'degrees in clockwise direction (because 90 is positive) about the origin.')
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for OUTPUT of ADDITION of two images             *
#***********************************************************************
def addition_output(arr1, arr2, nameImg1, nameImg2):
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\t\tOUTPUT OF ADDITION OF TWO IMAGES')
    print('\t\t', '-' * 60)
   
    # add matrix of two images together
    addition = arr1 + arr2
    
    # name and save the addition of image
    img = Image.fromarray(addition, 'RGB')
    img.save('addition.png')
    
    # display the two original images using matplotlib
    print('\n\t****Image 1****') # image 1
    img = mpimg.imread(nameImg1) 
    imgplot = plt.imshow(img)
    plt.show()
    print('\n\t****Image 2****') # image 2
    img = mpimg.imread(nameImg2)
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the addition of two images using matplotlib
    print('\n **Image 1 + Image 2**')
    img = mpimg.imread('addition.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the message about application of addition of images in real life
    print('\nApplication of addition of image in real life:')
    print('\nBy applying addition in image processing, we can add a logo inside an')
    print('image, and also mix the colour of logo with the colour of another image.')
    print('-' * 77)
    
    return

#***********************************************************************
# Function definition for OUTPUT of SUBTRACTION of two images          *                           
#***********************************************************************
def subtraction_output(arr1, arr2, nameImg1, nameImg2):
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tOUTPUT OF SUBTRACTION OF TWO IMAGES')
    print('\t\t', '-' * 60)
    
    # subtract image1 from image2
    subtraction = array2 - array1
    
    # name and save the subtraction of images
    img = Image.fromarray(subtraction, 'RGB')
    img.save('subtraction.png')
    
    # display the two original images
    print('\n\t****Image 1****') # image 1
    img = mpimg.imread(nameImg1) 
    imgplot = plt.imshow(img)
    plt.show()
    print('\n\t****Image 2****') # image 2
    img = mpimg.imread(nameImg2)
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the subtraction of images
    print('\n **Image 1 - Image 2**')
    img = mpimg.imread('subtraction.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the application of subtraction of images in real life 
    print('\nApplication of subtraction of images in real life:')
    print('\n- By applying subtraction in image processing, we can mix the') 
    print('colour of logo with the colour of another image. ')
    print('- However, unlike addition, the background of the image will be')
    print('brighter by subtracting Image 1 from Image 2.')
    print('-' * 77)
    
    return 

#***********************************************************************
# Function definition for OUTPUT of MULTIPLICATION of two image        *
#***********************************************************************
def multiplication_output(arr1, arr2, nameImg1, nameImg2):
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tOUTPUT OF MULTIPLICATION OF TWO IMAGES')
    print('\t\t', '-' * 60)
    
    # multiply image1 with image2 
    # (multiply each element by its corresponding element) 
    multiply = array1 * array2
    
    # name and save the multiplication of two images
    img = Image.fromarray(multiply, 'RGB')
    img.save('multiplication.png')
    
    # display the two original images
    print('\n\t****Image 1****') # image 1
    img = mpimg.imread(nameImg1) 
    imgplot = plt.imshow(img)
    plt.show()
    print('\n\t****Image 2****') # image 2
    img = mpimg.imread(nameImg2)
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the multiplication of two images
    print('\n **Image 1 x Image 2**')
    img = mpimg.imread('multiplication.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # display the application of multiplication of images in real life  
    print('\nApplication of multiplication of images in real life:')
    print('\nBy applying multiplication in image processing, we can add a logo')
    print('inside an image, and also mix the colour of logo with the colour')
    print('of another image.')
    print('-' * 77)
    
    return  

#*********************************************************************
# Function definition for OUTPUT of GRAYSCALE of an image            *                           
#*********************************************************************
def grayscale_output():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\t\tOUTPUT OF GRAYSCALE OF AN IMAGE')
    print('\t\t', '-' * 60)
    
    # read the image 
    oldImage = mpimg.imread("lena.png")
            
    # get the length and width of original image
    # the shape of the matrix is (length x width x other dimension)
    # So, 'oldImage.shape[:2]' means we only wants to read the first 
    # two dimensions (length & width)
    length, width = oldImage.shape[:2]
    
    # create an 3d array for new image dimension with 3 attributes in 
    # each pixel (Red, Green, Blue (RGB)), initialize the elements to zero  
    newImage = np.zeros([length, width, 3])
     
    # to turn the image into grayscale, we have to calculate
    # the average value of RGB and assign it to RGB value of pixel 
    for i in range(length):
        for j in range(width):   
          
          # get the RGB value in each pixel (with index 0, 1, 2 in the 3rd 
          # dimension of array)
          RGB = [float(oldImage[i][j][0]), float(oldImage[i][j][1]), 
                 float(oldImage[i][j][2])]
          
          # calculate the average of RGB value
          avg = float(np.mean(RGB))
          
          # assign the average of RGB value into the each RGB value 
          # pixel. Since all of the RGB values are the same in each
          # pixel, we will not be able to form any colour, hence 
          # grayscale image will be produced
          newImage[i][j][0] = avg
          newImage[i][j][1] = avg
          newImage[i][j][2] = avg
      
    # Name and save the grayscale image
    mpimg.imsave('grayscale.png', newImage)
    
    # Display the old image
    print('\n\t**Original Image**')
    img = mpimg.imread('lena.png')
    imgplot = plt.imshow(img)
    plt.show()   
    
    # Display the grayscale image 
    print('\n\t**Grayscale Image**')
    img = mpimg.imread('grayscale.png') 
    imgplot = plt.imshow(img)
    plt.show()  
    print('-' * 77)
    
    return

#********************************************************************
# Function definition for OUTPUT of ENLARGEMENT of an image         *                           
#********************************************************************
def enlargement_output():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\tOUTPUT OF ENLARGEMENT OF AN IMAGE')
    print('\t\t', '-' * 60)
    
    # read the image 
    oldImage = mpimg.imread("lena.png")
      
    # get the length and width of original image
    # 'oldImage.shape[:2]' here means we only wants to read the first 
    # two dimensions (length & width)
    length, width = oldImage.shape[:2]
    
    # to show the example of output of enlarged image, we fix the
    # scale factor into 1/2
    scaleFactor = 1/2 
    print('\nScale factor for enlargement =', scaleFactor)
    
    # calculate the new width and height of image after scaling
    newLength = int(length * scaleFactor)
    newWidth = int(width * scaleFactor)
     
    # create a matrix of newLength and newHeight with 3 attributes (R,G,B) 
    # values, initialize each entries in the matrix to zero
    scaledImage = np.zeros([newLength, newWidth, 3])
      
    # using for loop to assign the new value to each entries of the scaled
    # image matrix, the new value is calculated by dividing old value with
    # the scale factor
    for i in range(newLength):
        for j in range(newWidth):
            scaledImage[i, j]= oldImage[int(i / scaleFactor), 
                                        int(j / scaleFactor)]
    
    # Display the old image
    print('\n\t**Original Image**')
    img = mpimg.imread('lena.png')
    imgplot = plt.imshow(img)
    plt.show() 
    
    # print the length and width of original image
    print('Original length:', length)
    print('Original width:', width)
    
    # Save and display the scaled image
    print('\n\t**Enlarged Image**')
    mpimg.imsave('enlargement.png', scaledImage)
    img = mpimg.imread('enlargement.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # print length and width of enlarged image
    print('Enlarged image length:', newLength)
    print('Enlarged image width:', newWidth)
    print('\n**We can notice that the scale of image has been changed.')
    print('-' * 77)
    
    return 

#********************************************************************
# Function definition for OUTPUT of CROPPING of an image            *                           
#********************************************************************
def cropping_output():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\t\tOUTPUT OF CROPPING OF AN IMAGE')
    print('\t\t', '-' * 60)
    
    # read the image 
    oldImage = mpimg.imread("lena.png")
      
    # determining the length of original image
    # 'oldImage.shape[:2]' here means we only wants to read the first 
    # two dimensions (length & width) of the image
    length, width = oldImage.shape[:2]
    
    # get the length and width of image after cropping 
    # (we fix the new length and width to half of their original values)
    newLength = int(length * 1 / 2)
    newWidth = int(width * 1 / 2)
    
    # create a matrix of newLength and newHeight with 3 attributes (R,G,B) values
    # initialize each entries in the matrix to zero 
    newImage = np.zeros([newLength, newWidth, 3])
      
    # using for loop to assign the new value to each entries of the cropping
    # image matrix, get the new value by assigning (i+100)th and (j+100)th 
    # entries of old image to the ith and jth entries of new image respectively
    # (we choose to crop the image starts from 100th pixel of original image)
    for i in range(newLength):
        for j in range(newWidth):
            newImage[i, j]= oldImage[100 + i, 100 + j]
      
    # Display the original image
    print('\n\t**Original Image**')
    img = mpimg.imread('lena.png')
    imgplot = plt.imshow(img)
    plt.show()   
      
    # save and display the cropped image
    print('\n\t**Cropped Image**')
    mpimg.imsave('cropped.png', newImage)
    img = mpimg.imread('cropped.png')
    imgplot = plt.imshow(img)
    plt.show()
    print('-' * 77)
    
    return 

#********************************************************************
# Function definition for OUTPUT of ROTATION of an image            *                           
#********************************************************************
def rotation_output():
    
    print('\n\t\t', '-' * 60)
    print('\t\t\t\t\t\tOUTPUT OF ROTATION OF AN IMAGE')
    print('\t\t', '-' * 60)
    
    # read the image and convert it into matrix(array)
    image = np.array(Image.open("receipt.png"))
    
    # Display the original image
    print('\n\t**Original Image**')
    img = mpimg.imread('receipt.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # input the angle of rotation from user          
    print('\nThe following image shows a receipt which has not been straighten.')
    print('Please enter an angle of rotation to straighten the receipt.')
    print('(Positive angle - rotate clockwise; negative angle - rotate anticlockwise)')
    angle = int(input("Angle of rotation: "))                
    
    # define the variables
    angle = np.radians(angle)    # convert the angle into radians
    cosine = np.cos(angle)       # find the cosine of angle      
    sine = np.sin(angle)         # find the sine of angle

    # determining the length and width of original image
    # 'image.shape[:2]' here means we only wants to read the first 
    # two dimensions (length & width) of the image
    length, width = image.shape[:2];
    
    # Find the new length and new width of the image that is to be rotated
    # Formula to find the new dimensions:
    # newLength = |oldLength x cos𝜃| + |oldWidth x sin𝜃|
    # newWidth  = |oldWidth x cos𝜃| + |oldLength x sin𝜃|
    # round() function is used to round off the floating number to nearest integer
    # abs() function is used to return the absolute value of the number
    newLength  = round(abs(length * cosine) + abs(width * sine)) 
    newWidth  = round(abs(width * cosine)+ abs(length * sine)) 
    
    # create a matrix of newLength and newHeight with 3 attributes (R,G,B) values
    # initialize each entries in the matrix to zero 
    newImage = np.zeros((newLength, newWidth, 3))
   
    # Find the centre of the old image by dividing the length & width by 2
    old_centre_length = round((length / 2)) 
    old_centre_width = round((width / 2))
    
    # Find the centre of the new image by dividing the new length & width by 2
    new_centre_length = round((newLength / 2))      
    new_centre_width = round((newWidth / 2)) 
    
    for i in range(length):
        for j in range(width):
            # find the coordinates of pixel with respect to the 
            # centre of original image
            x = length - i - old_centre_length                   
            y = width - j - old_centre_width                      
                        
            # find coordinates of pixel with respect to the new image
            # the matrix formula to find newX and newY is:
            # | newX | = |  cos𝜃   sin𝜃 |  | x |
            # | newY |   | -sin𝜃   cos𝜃 |  | y |
            # hence by applying multiplication in matrix, 
            # newX = xcos𝜃 + ysin𝜃 and newY = -xsin𝜃 + ycos𝜃
            newX = round(x * cosine + y * sine)
            newY = round(-x * sine + y * cosine)
            
            # since image will be rotated, the centre will be changed, so we need to
            # change newX and newY with respect to the new centre length and width
            newX = new_centre_length - newX
            newY = new_centre_width - newY
    
            # if statement to check the values of newX & newY to avoid any error
            if 0 <= newX < newLength and 0 <= newY < newWidth and newX >=0 and newY >= 0:
                # assign the pixels to each entries in the new image matrix
                newImage[newX, newY, :] = image[i, j, :]
            else:
                print('Error')
           
    
    # Display the rotated image
    print('\n\t**Rotated Image**')
    # convert array to image
    pil_img=Image.fromarray((newImage).astype(np.uint8))                       
    pil_img.save("rotated_image.png")
    img = mpimg.imread('rotated_image.png')
    imgplot = plt.imshow(img)
    plt.show() 
    
    # display application of rotation in image processing in real life
    print('''
Application of rotation of image in real life:
    
From the rotated image above, we can see that rotation helps to straighten the 
image, and also straighten the words in the receipt, so that the words can be 
detected by the receipt-scanning apps.
''')        
    print('-' * 77)
    
    return 

########################################################################
#                             MAIN FUNCTION                            #
########################################################################

# open the image file, convert the images into matrices (arrays)
image1 = Image.open("brown.png")   # open image 1 
array1 = array(image1)             # convert image 1 into matrix(array)
name1 = "brown.png"                # name of image 1

image2 = Image.open("insta.png")   # open image 2 
array2 = array(image2)             # convert image 2 into matrix(array)
name2 = "insta.png"                # name of image 2

back_to_menu = True

# while loop to run the program if user does not exit the program
while back_to_menu == True:
    # display the message at the beginning of the program
    print('\n\t\t\t*********************************************************')
    print('\t\t\t\t\t  Welcome to our Image Processing System')
    print('\t\t\t*********************************************************\n')
    print('Hi, this is a program about the application of linear algebra in image processing.\n')
    print('What can I help you?')
    print('1. Display the explanation of matrix application in image processing system.')
    print('2. Display the output of image processing.')
    print('3. Exit.') 
    
    # function call to get the choice from user, the argument '3' means there 
    # are 3 choices
    choice = getChoice(3)
    
    # if choice of user is to display the explanation of matrix in image
    # processing system
    if choice == 1:
        
        continue_explanation = 1
        # while loop to display the explanation if user choose to continue
        while continue_explanation == 1:
            
            # display the message and function call to let user choose 
            # different matrix application
            print('\nWhich explanation of matrix in image processing do you want to explore?')
            displayMessage_application()
            
            # function call to get the choice from user, the argument '7' means 
            # there are 7 choices
            choice_explanation = getChoice(7)
            
            # function call to display explanation according to user's choice 
            if choice_explanation == 1:
                addition_explanation()
                
            elif choice_explanation == 2:
                subtraction_explanation()
                
            elif choice_explanation == 3:
                multiplication_explanation()
                
            elif choice_explanation == 4:
                grayscale_explanation()
                
            elif choice_explanation == 5:
                enlargement_explanation()
                
            elif choice_explanation == 6:
                cropping_explanation()
                
            elif choice_explanation == 7:
                rotation_explanation()
            
            # ask user if they want to continue on explanation
            print('\nDo you wish to continue on explanation of matrix? ')
            print('1 - Yes')
            print('2 - No. Back to main menu.')
            continue_explanation = getChoice(2)
            
            # if user does not choose to continue, while loop ends and
            # back to main menu
            if continue_explanation == 2:
                back_to_menu = True
    
    # if choice of user is to display the output of image processing
    elif choice == 2:
        
        continue_output = 1
        # while loop to display the explanation if user choose to continue
        while continue_output == 1:
            
            # display the message and function call to let user choose 
            # different output of image
            print('\nWhich type of output of image do you want to look at?')
            displayMessage_application()
            
            # function call to get the choice from user, the argument '7' means 
            # there are 7 choices
            choice_output = getChoice(7)
            
            if choice_output == 1:
                # output the addition of two images, pass the images' matrices 
                # and file names as arguments
                addition_output(array1, array2, name1, name2)
                
            elif choice_output == 2:
                # output the addition of two images, pass the images' matrices 
                # and file names as arguments
                subtraction_output(array1, array2, name1, name2)
                
            elif choice_output == 3:
                # output the multiplication of two images, pass the images' matrices 
                # and file names as arguments
                multiplication_output(array1, array2, name1, name2)
                
            elif choice_output == 4:
                # output the grayscale of an image
                grayscale_output()
                
            elif choice_output == 5:
                # output the enlargement of an image
                enlargement_output()
                
            elif choice_output == 6:
                # output the cropped image
                cropping_output()
                
            elif choice_output == 7:
                # output the rotated image
                rotation_output()
            
            # ask user if they want to continue on output of image processing
            print('\nDo you wish to continue on output of image processing? ')
            print('1 - Yes')
            print('2 - No. Back to main menu.')
            continue_output = getChoice(2)
            
            # if user does not choose to continue, while loop ends and
            # back to main menu
            if continue_output == 2:
                back_to_menu = True       
            
    else:
        # main menu while loop ends if user choose to exit this program
        # print thank you message
        print('\nThanks for exploring our program. Have a nice day!')
        back_to_menu = False
