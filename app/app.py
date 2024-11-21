import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image, ImageTk
import io

def upload_image():
    filepath = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        # Open the image and display it
        img = Image.open(filepath)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

        # Send the image to Flask server for prediction

# Set up the GUI window
root = tk.Tk()
root.title("Plant Disease Prediction")

# Create and pack widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="Prediction will appear here.", font=("Helvetica", 14))
result_label.pack(pady=20)

# Run the GUI main loop
root.mainloop()
