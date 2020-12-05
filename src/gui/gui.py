"""
GUI "Paint" program to accompany team "A Cool Name"'s final project for CSCI 4622 Machine Learning.
Provides a GUI to draw images and submit them to the machine learning model.
"""


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from io import BytesIO

import cv2 as cv
import numpy as np
#from cycle_gan import CycleGan

class Gui:
    def __init__(self):
        self.mode = 0  # 0 = paint, 1 = erase
        self.x = 0
        self.y = 0

        self.root = tk.Tk()

        # Declare Variables Used in GUI
        self.size = tk.IntVar()  # Paint Brush Size
        self.size.set(5)
        self.selectedModel = tk.StringVar()  # What to generate in our GAN
        self.selectedModel.set('tree_cyclegan')
        self.selectedGAN = tk.StringVar()  # What GAN to use for real time display
        self.selectedGAN.set('pix2pix')
        self.canvasTitle = tk.StringVar()  # Title for drawing area
        self.canvasTitle.set('Drawing (Trees)')
        self.imageViewerTitle = tk.StringVar()  # Title for image viewing area
        self.imageViewerTitle.set('Generated Image (Pix2Pix)')

        self.root.title("Pix2Pix GAN GUI")
        self.root.resizable(width=False, height=False)
        self.root.geometry("1052x645+436+218")  # 1050x590 window centered on 1920x1080 displays

        # Next 3 lines to bring window to front on launch, but allow other apps in front afterwards
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Create all needed buttons
        self.drawButton = ttk.Button(self.root, text="Draw", command=self.setDraw).grid(row=0, column=0)
        self.eraseButton = ttk.Button(self.root, text="Erase", command=self.setErase).grid(row=0, column=1)
        self.clearAllButton = ttk.Button(self.root, text="Clear All", command=self.clearAll).grid(row=1, column=0)
        self.compareButton = ttk.Button(self.root, text="Compare", command=self.compare).grid(row=1, column=1)

        # Create Model Selector above image viewer
        self.modelSelectorLabel = ttk.Label(self.root, text='Select What to Generate:').grid(row=0, column=3)
        self.treeButton = ttk.Radiobutton(self.root, text="Trees", variable=self.selectedModel, value='tree_cyclegan', command=self.selectModel)
        self.treeButton.grid(row=0, column=4)
        self.quickDraw = ttk.Radiobutton(self.root, text="QuickDraw", variable=self.selectedModel, value='tree_quickdraw_cyclegan', command=self.selectModel)
        self.quickDraw.grid(row=0, column=5)
        self.pizzaButton = ttk.Radiobutton(self.root, text="Pizza", variable=self.selectedModel, value='pizza_cyclegan', command=self.selectModel)
        self.pizzaButton.grid(row=0, column=6)
        self.appleButton = ttk.Radiobutton(self.root, text="Apples", variable=self.selectedModel, value='apple_cyclegan', command=self.selectModel)
        self.appleButton.grid(row=0, column=7)

        # Create GAN Selector below model selector
        self.ganModelLabel = ttk.Label(self.root, text="Select GAN to Use:").grid(row=1, column=4, pady=(10, 0))
        self.pix2pixButton = ttk.Radiobutton(self.root, text="Pix2Pix", variable=self.selectedGAN, value='pix2pix', command=self.selectGan).grid(row=1, column=5, pady=(10, 0))
        self.cycleGanButton = ttk.Radiobutton(self.root, text="CycleGAN", variable=self.selectedGAN, value='cyclegan', command=self.selectGan).grid(row=1, column=6, pady=(10, 0))

        # Create Paint Size slider
        self.scaleLabel = tk.Label(self.root, text="Size").grid(row=1, column=2)
        self.scale = tk.Scale(self.root, variable=self.size, from_=1, to=10, cursor='target', orient=tk.HORIZONTAL).grid(row=0, column=2)

        # Create canvas to draw on, but don't pack it until adding event bindings
        self.canvasLabel = tk.Label(self.root, textvariable=self.canvasTitle, font=("Arial", 16)).grid(row=2, column=0, columnspan=3, pady=(25, 0))
        self.canvas = tk.Canvas(self.root, cursor='dot', height=512, width=512, bg='white', highlightthickness=3, highlightbackground='grey')

        self.canvas.bind('<ButtonPress-1>', self.setStartPoint)  # Gets initial left-click
        self.canvas.bind('<B1-Motion>', self.draw)  # Gets motion from mouse while left-click is held
        self.canvas.bind('<ButtonRelease-1>', self.updateImageViewer)  # Update image in real time on left-click release

        self.canvas.grid(row=3, column=0, columnspan=3, padx=5, pady=3)

        # Create Canvas to view images produced by GAN
        self.imageViewerLabel = tk.Label(self.root, textvariable=self.imageViewerTitle, font=("Arial", 16)).grid(row=2, column=3, columnspan=5, pady=(25, 0))
        self.imageViewer = tk.Canvas(self.root, cursor='arrow', height=512, width=512, bg='white', highlightthickness=3, highlightbackground='grey')
        self.imageViewer.grid(row=3, column=3, columnspan=5)

        self.root.mainloop()

    def setDraw(self):
        self.mode = 0

    def setErase(self):
        self.mode = 1

    def clearAll(self):
        self.canvas.delete('all')
        self.updateImageViewer()

    # Called on left-click release to update image in real time
    def updateImageViewer(self, event=None):
        canvasImage = self.getImageFromCanvas()

        # Pass drawn image to GAN and display output
        if self.selectedGAN.get() == 'pix2pix':
            #TODO: Generate Pix2Pix Image and display it
            ## self.pix2pix = Pix2Pix(self.selectedModel.get())
            ## pix2pixImage = self.pix2pix.generate_image(canvasImage)
            pix2pixImage = canvasImage.resize((512, 512))  # TODO: Replace canvasImage with pix2pixImage
            self.im2 = ImageTk.PhotoImage(pix2pixImage)

        elif self.selectedGAN.get() == 'cyclegan':
            #self.tree_cycle_gan = CycleGAN(self.selectedModel.get(), epoch=170)
            #cycleGanImage = self.tree_cycle_gan(canvasImage)
            cycleGanImage = canvasImage.resize((512, 512))  # TODO: Replace canvasImage with cycleGanImage
            self.im2 = ImageTk.PhotoImage(cycleGanImage)

        # Now that we have the image, actually display it
        self.imageViewer.create_image(0, 0, image=self.im2, anchor=tk.NW)

        if event:
            self.x = event.x
            self.y = event.y
        self.root.update()

    # Compare image from Canvas to all GAN outputs possible for selected model
    def compare(self):
        # Get 256x256 image drawn on canvas
        canvasImage = self.getImageFromCanvas()

        #TODO: organize models by input type, match to user selected class
        # self.selectedModel.get() contains option from list below
        #Options will be tree-cyclegan, tree-quickdraw-cyclegan, pizza-cyclegan, apple-cyclegan (should be passed now)
        # self.tree_cycle_gan = CycleGan(self.selectedModel.get(), epoch=170)
        
        # TODO: pass through right models
        fake_image = self.tree_cycle_gan(canvasImage)

        # TODO: create pix2pix model for each target (apple, tree, pizza)
        
        cv.imshow("CycleGAN", fake_image)
        cv.imshow("Pix2Pix", np.array(canvasImage)) # for now, same image

        # TODO: Display Drawing alongside any images generated from both CycleGAN + Pix2Pix


    # On initial left mouse click, set where the click occurred
    def setStartPoint(self, event):
        self.x = event.x
        self.y = event.y

    # When moving the mouse with left button down, draw on canvas
    def draw(self, event):
        color = 'black'
        if self.mode:
            color = 'white'
        self.canvas.create_line(self.x, self.y, event.x, event.y, width=self.size.get(), smooth=True, fill=color)
        self.x = event.x
        self.y = event.y

    def selectModel(self):
        modelSelected = self.selectedModel.get()
        if modelSelected == 'tree_cyclegan':
            self.canvasTitle.set('Drawing (Trees)')
        elif modelSelected == 'tree_quickdraw_cyclegan':
            self.canvasTitle.set('Drawing (Trees Quickdraw)')
        elif modelSelected == 'pizza_cyclegan':
            self.canvasTitle.set('Drawing (Pizza)')
        elif modelSelected == 'apple_cyclegan':
            self.canvasTitle.set('Drawing (Apples)')
        self.root.update()

    def selectGan(self):
        ganSelected = self.selectedGAN.get()
        if ganSelected == 'pix2pix':
            self.imageViewerTitle.set('Generated Image (Pix2Pix)')
        elif ganSelected == 'cyclegan':
            self.imageViewerTitle.set('Generated Image (CycleGAN)')
        self.root.update()

        self.updateImageViewer()

    def getImageFromCanvas(self):
        canvasImage = self.canvas.postscript(colormode='color')  # Get Canvas image as Unicode string
        im = Image.open(BytesIO(canvasImage.encode('utf-8')))  # Convert to image
        im = im.resize((256, 256))  # Resize 512x512 canvas to 256x256 to match ML model
        return im


if __name__ == "__main__":
    g = Gui()