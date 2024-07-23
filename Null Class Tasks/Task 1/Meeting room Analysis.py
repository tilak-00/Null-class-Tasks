import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk

class Person:
    def __init__(self, gender, age, shirt_color):
        self.gender = gender
        self.age = age
        self.shirt_color = shirt_color

def detect_people_in_meeting_room(people):
    if len(people) < 2:
        return "Not enough people in the meeting room"
    
    females = 0
    males = 0
    total_age = 0
    
    for person in people:
        if person.shirt_color == "white":
            person.age = 23
        elif person.shirt_color == "black":
            person.age = "child"
        
        if person.gender == "female":
            females += 1
        elif person.gender == "male":
            males += 1
        
        if person.age != "child":
            total_age += person.age
    
    return f"Females: {females}, Males: {males}, Average Age: {total_age / (females + males)}"

# Example usage
people = [
    Person("female", 25, "white"),
    Person("male", 30, "black"),
    Person("female", 28, "red"),
    Person("male", 35, "blue")
    ]
win= Tk()
win.title("detecting image")
win.geometry("800x600")

frame = Frame(win, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)

# Create an object of tkinter ImageTk
image = ImageTk.PhotoImage(Image.open("C:/Users/nicei/Downloads/meeting room.jfif"))
# Create a Label Widget to display the text or Image
label = Label(frame, image = image)
label.pack()       
win.geometry('800x600')     
btn = Button(win, text = 'detect persons', 
                command = win.destroy  ) 
btn.pack(side = 'bottom')
heading = Label(win, text="Meeting hall Detector ")
heading.pack()

print(detect_people_in_meeting_room(people))
