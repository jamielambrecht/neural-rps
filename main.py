
# STANDARD LIBRARIES
from collections import deque
from enum import IntEnum
from random import choice, shuffle
from re import M
import sys
from tkinter.constants import TOP
from typing import Text
#Pillow
from PIL import Image, ImageTk
#Tkinter
from tkinter import Text, Tk, LEFT, RIGHT, BOTH, RAISED, INSERT, CENTER, TOP
from tkinter.ttk import Frame, Button, Style, Label

# THIRD-PARTY LIBRARIES
import cv2 # Open-CV
import numpy as np # NumPy
from scipy import stats # SciPy
# SK Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.python.ops.gen_math_ops import Imag

# Enumeration
class Players(IntEnum):
    player = 0,
    computer = 1

class ImageClass(IntEnum):
    rock = 0,
    paper = 1,
    scissors = 2,
    nothing = 3

# Raw image data
image_caches = {
    ImageClass.rock:     [],
    ImageClass.paper:    [],
    ImageClass.scissors: [],
    ImageClass.nothing:  []
}


def collect_training_images_for_model(samples_amount, window, tk):

    def handle_input_event(button_press):
        nonlocal start_recording; start_recording = True
        nonlocal image_class; image_class = button_press

    def exit_screen(quit):
        nonlocal continue_flag; continue_flag = True
        nonlocal quit_flag; quit_flag = quit

    ## UI ATTRIBUTES
    window_name = "Collecting training data images"
    window.master.title(window_name)
    ## UI ELEMENTS
    ### Camera Feed Image
    window_frame = Frame(window, relief=RAISED, borderwidth=1)
    window_frame.pack(fill=BOTH, expand=True)
    image_tklabel = Label(window_frame)
    image_tklabel.pack()
    window.pack(fill=BOTH, expand=True)
    ### Buttons / Etc.
    ui_text = Label(window_frame, text="Prepare to take images of the selected gesture...")
    ui_text.pack(side=LEFT, padx=5, pady=5)
    quit_button = Button(window_frame, text="Quit", command=lambda m=True: exit_screen(quit=m))
    quit_button.pack(side=RIGHT, padx=5, pady=5), 
    continueButton = Button(window_frame, text="Continue", command=lambda m=False : exit_screen(quit=m),  state="disable")
    continueButton.pack(side=RIGHT, padx=5, pady=5), 
    nothingButton = Button(window_frame, text="Nothing", command=lambda m=ImageClass.nothing: handle_input_event(m))
    nothingButton.pack(side=RIGHT, padx=5, pady=5), 
    scissorsButton = Button(window_frame, text="Scissors", command=lambda m=ImageClass.scissors: handle_input_event(m))
    scissorsButton.pack(side=RIGHT, padx=5, pady=5)
    paperButton = Button(window_frame, text="Paper", command=lambda m=ImageClass.paper: handle_input_event(m))
    paperButton.pack(side=RIGHT, padx=5, pady=5)
    rockButton = Button(window_frame, text="Rock", command=lambda m=ImageClass.rock: handle_input_event(m))
    rockButton.pack(side=RIGHT, padx=5, pady=5)
    
    # OTHER ROUTINE DATA
    continue_flag = False
    quit_flag = False
    image_class = None
    capture = cv2.VideoCapture(0)
    start_recording = False
    region_of_interest_side_length = 234  # subtract 10px to get image size
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_recording = False

    while not continue_flag:
        frame_received, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        if not frame_received:
            print("Error reading from camera!")
            sys.exit(1)
        
        if image_class != None:
            if len(image_caches[image_class]) == samples_amount: # Reset capture process when enough samples are collected
                start_recording = False
                print(text, end='\n')
                [rockButton, paperButton, scissorsButton, nothingButton][image_class.value].configure(state="disable")
                image_class = None
            if sum(len(cache) for cache in image_caches.values()) == len(image_caches) * samples_amount:
                continueButton.configure(state="enable") 

        cv2.rectangle(frame,
                      (frame_width - region_of_interest_side_length, 0),
                      (frame_width, region_of_interest_side_length),
                      (0, 250, 150),
                      2
                      )  # Draw a box to show the region of interest

        if start_recording:
            data_image = frame[5: region_of_interest_side_length - 5,
                          frame_width - region_of_interest_side_length + 5: frame_width - 5
                          ] # Store the captured image frame (crop 5px from all sides of frame region)

            image_caches[image_class].append([data_image, str(image_class)])

            text = "Collected Samples of {}: {}".format(
                str(image_class.name), len(image_caches[image_class]))
            print(text, end='\r')
            ui_text.configure(text=text)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        image_tklabel.configure(image=frame)
        
        tk.update_idletasks()
        tk.update()

    for child in window.winfo_children():
        child.destroy()
    capture.release()
    return quit_flag


def build_model(rock, paper, scissors, nothing, window, tk):
    def update(text=None):
        nonlocal ui_text_stack
        nonlocal output_text_stack
        output_text_stack.append(text)
        for i, ui_text in enumerate(ui_text_stack):
            ui_text.configure(text=output_text_stack[i])
        tk.update_idletasks()
        tk.update()

    def exit_screen(quit):
        nonlocal continue_flag; continue_flag = True

    ## UI ATTRIBUTES
    window_name = "Training neural network model"
    window.master.title(window_name)
    ## UI ELEMENTS
    ### Camera Feed Image Area
    window_frame = Frame(window, relief=RAISED, borderwidth=1)
    window_frame.pack(fill=BOTH, expand=True)
    window.pack(fill=BOTH, expand=True)
    ### Buttons / Etc.
    ui_text_stack_container = Frame(window_frame)
    text_stack_length = 10
    output_text_stack = deque([" "] * text_stack_length, maxlen=text_stack_length)
    ui_text_stack = []
    for i in range(text_stack_length):
        ui_text_stack.append(Label(ui_text_stack_container, text=str(i)))
        ui_text_stack[i].grid(row=int(i), column=0, padx=5, pady=5)
    ui_text_stack_container.pack(side=TOP, pady=32)
    header_text = Label(window_frame, text="Training neural network model...")
    header_text.pack(side=TOP, pady=32)
    continue_button = Button(window_frame, text="Continue", command=lambda m=False : exit_screen(quit=m),  state="disable")
    continue_button.pack(side=RIGHT, padx=5, pady=5), 
    update()

    # OTHER ROUTINE DATA
    continue_flag = False

    image_labels = [tupl[1] for tupl in rock + paper + scissors + nothing]
    images = [tupl[0] for tupl in rock + paper + scissors + nothing]

    images = np.array(images, dtype="float") / 255.0 # Normalize the image data.

    label_encoder = LabelEncoder() 
    # IMPORTANT: Each value for the field in the tuple is a string that will be enumerated
    # beginning at 1 in alphabetical order, i.e 1. nothing, 2. paper, 3. rock, 4. scissors. 
    labels_sorted_strings_enum = label_encoder.fit_transform(image_labels) 
    
    # Convert the label enumerators into one-hot format, i.e. 0 = [1,0,0,0], etc.
    one_hot_labels = to_categorical(labels_sorted_strings_enum, 4)
    
    # Split the data, allocate 75% of the data for training and 25% for testing.
    (trainX, testX, trainY, testY) = train_test_split(
        images, one_hot_labels, test_size=0.25, random_state=50)

    del images # Free up memory by clearing the raw images from RAM.
    del rock
    del paper
    del scissors
    del nothing
    
    image_size = 224 # Our model accepts this image size.

    # Loading pre-trained NASNETMobile Model, exluding the head via include_top = False
    nasnet_mobile = tf.keras.applications.NASNetMobile(input_shape=(
        image_size, image_size, 3), include_top=False, weights='imagenet')

    nasnet_mobile.trainable = False # Freeze the model

    nnm_output = nasnet_mobile.output # Custom head based on output feature maps from NASNETMobile.
    nnm_output = GlobalAveragePooling2D()(nnm_output) # use global average pooling to minimize overfitting
    nnm_output = Dense(512, activation='relu')(nnm_output) # a dense neural network layer with 712-d output vector, Recitified Linear Unit activation
    nnm_output = Dropout(0.40)(nnm_output) # Drop 40% of activations to reduce overfitting
    nnm_output = Dense(4, activation='softmax')(nnm_output) # a probability distribution with 4 output classes, softmax activation

    model = Model(inputs=nasnet_mobile.input, outputs=nnm_output)

    augment_data = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        horizontal_flip=False,
        fill_mode="nearest"
    ) # Helpful transformations

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    epochs, batch_size = 10, 10 # Set epochs and batch_size appropriately for your system

    # Callback functions used by the training process to give status updates.
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            keys = list(logs.keys())
            text = "Start epoch {} of training; got log keys: {}".format(epoch, keys)
            update(text)

        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            text = "End epoch {} of training; got log keys: {}".format(epoch, keys)
            update(text)

        def on_test_begin(self, logs=None):
            keys = list(logs.keys())
            text = "Start testing; got log keys: {}".format(keys)
            update(text)

        def on_test_end(self, logs=None):
            keys = list(logs.keys())
            text = "Stop testing; got log keys: {}".format(keys)
            update(text)

        def on_predict_begin(self, logs=None):
            keys = list(logs.keys())
            text = "Start predicting; got log keys: {}".format(keys)
            update(text)

        def on_predict_end(self, logs=None):
            keys = list(logs.keys())
            text = "Stop predicting; got log keys: {}".format(keys)
            update(text)

        def on_train_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Training: start of batch {}; got log keys: {}".format(batch, keys)
            update(text)

        def on_train_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Training: end of batch {}; got log keys: {}".format(batch, keys)
            update(text)

        def on_test_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Evaluating: start of batch {}; got log keys: {}".format(batch, keys)
            update(text)

        def on_test_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Evaluating: end of batch {}; got log keys: {}".format(batch, keys)
            update(text)

        def on_predict_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Predicting: start of batch {}; got log keys: {}".format(batch, keys)
            update(text)

        def on_predict_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            text = "Predicting: end of batch {}; got log keys: {}".format(batch, keys)
            update(text)
    
    # Start training
    model.fit(x=augment_data.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size, epochs=epochs,
                        verbose = 1, callbacks=[CustomCallback()])

    continue_button.configure(state="enable") 
    for child in window.winfo_children():
        child.destroy()

    return model


def check_model(model, label_names, window, tk):
    def exit_screen(quit):
        nonlocal continue_flag; continue_flag = True
        nonlocal quit_flag; quit_flag = quit

    ## UI ATTRIBUTES
    window_name = "Checking model for accuracy..."
    window.master.title(window_name)
    ## UI ELEMENTS
    ### Camera Feed Image
    window_frame = Frame(window, relief=RAISED, borderwidth=1)
    window_frame.pack(fill=BOTH, expand=True)
    image_tklabel = Label(window_frame)
    image_tklabel.pack()
    window.pack(fill=BOTH, expand=True)
    ### Buttons / Etc.
    ui_text_frame = Frame(window_frame, borderwidth=1)
    ui_text = Label(ui_text_frame, text="Check gestures to ensure model accuracy...")
    ui_text.grid(column = 0, row = 0)
    ui_text_2 = Label(ui_text_frame, text="")
    ui_text_2.grid(column = 0, row = 1)
    ui_text_frame.pack(side=LEFT, padx=5, pady=5)
    quit_button = Button(window_frame, text="Quit", command=lambda m=True: exit_screen(quit=m))
    quit_button.pack(side=RIGHT, padx=5, pady=5)
    continueButton = Button(window_frame, text="Continue", command=lambda m=False : exit_screen(quit=m))
    continueButton.pack(side=RIGHT, padx=5, pady=5)
    
    # OTHER ROUTINE DATA
    continue_flag = False
    quit_flag = False
    cap = cv2.VideoCapture(0)
    region_of_interest_side_length = 234
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while not continue_flag:

        sampleCollected, frame = cap.read()
        if not sampleCollected:
            break

        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (width - region_of_interest_side_length, 0),
                      (width, region_of_interest_side_length), (0, 250, 150), 2)

        image = frame[5: region_of_interest_side_length - 5, width - region_of_interest_side_length + 5: width - 5]
        image = np.array([image]).astype('float64') / 255.0 # Normalize the image and convert to float64 array.
        prediction = model.predict(image)
        prediction_label_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        text = "Detected image class: {} {:.2f}%".format(label_names[prediction_label_index], confidence*100)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        image_tklabel.configure(image=frame)

        ui_text_2.configure(text=text)
        
        tk.update_idletasks()
        tk.update()

    for child in window.winfo_children():
        child.destroy()

    cap.release()
    return quit_flag


def determine_winner(player_move: ImageClass, computer_move: ImageClass):
    if player_move.value == computer_move.value:
        return None
    elif (player_move.value + 1) % 3 == computer_move.value:
        return Players.computer
    else:
        return Players.player



def play_game(window, tk):
    def update():
        tk.update_idletasks()
        tk.update()

    def exit_screen(quit):
        nonlocal continue_flag; continue_flag = True
        nonlocal quit_flag; quit_flag = quit

    ## UI ATTRIBUTES
    window_name = "Play Rock, Paper, Scissors!"
    window.master.title(window_name)
    ## UI ELEMENTS
    ### Camera Feed Image Area
    window_frame = Frame(window, relief=RAISED, borderwidth=1)
    window_frame.pack(fill=BOTH, expand=True)
    image_tklabel = Label(window_frame)
    image_tklabel.pack()
    window.pack(fill=BOTH, expand=True)
    ### Buttons / Etc.
    ui_text_frame = Frame(window_frame)
    ui_text_a1 = Label(ui_text_frame, text="Begin match by making rock, paper, or scissors gesture in the box.")
    ui_text_a1.grid(row = 0, column = 0, padx=5)
    ui_text_b1 = Label(ui_text_frame, text="")
    ui_text_b1.grid(row = 0, column = 1, padx=5)
    ui_text_c1 = Label(ui_text_frame, text="")
    ui_text_c1.grid(row = 0, column = 2, padx=5)
    ui_text_a2 = Label(ui_text_frame, text="")
    ui_text_a2.grid(row = 1, column = 0, padx=5)
    ui_text_b2 = Label(ui_text_frame, text="")
    ui_text_b2.grid(row = 1, column = 1, padx=5)
    ui_text_c2 = Label(ui_text_frame, text="")
    ui_text_c2.grid(row = 1, column = 2, padx=5)
    ui_text_frame.pack(side=LEFT, padx=5, pady=5)
    quit_button = Button(window_frame, text="Quit", command=lambda m=True: exit_screen(quit=m))
    quit_button.pack(side=RIGHT, padx=5, pady=5)
    update()

    # OTHER ROUTINE DATA
    continue_flag = False
    quit_flag = False

    capture = cv2.VideoCapture(0)
    region_of_interest = 234
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    attempts = 5 # the number of matches in a game. E.g. "best of 5", etc.
    player_move, computer_move = ImageClass.nothing, ImageClass.nothing

    image_labels = [ImageClass.nothing, ImageClass.paper,
            ImageClass.rock, ImageClass.scissors]

    scoreboard = {Players.player: 0,
                Players.computer: 0,
                None: 0}

    image_box_color = (255, 0, 0) # Color of the bounding box
    hand_detected = False # Use this variable to determine whether we have seen a hand or not
    matches_left = attempts # Used to count down the number of matches in the game set
    confidence_threshold = 0.70 # Minimum confidence value to assume the image recognition is correct 
    prediction_queue = 5 # use the mode of x predictions to reduce false positives
    moves_buffer = deque([ImageClass.nothing] * prediction_queue, maxlen=prediction_queue)

    while not continue_flag:

        sampleCollected, frame = capture.read()
        if not sampleCollected: # break the loop iif there is an error reading frames
            break
        frame = cv2.flip(frame, 1) # Flip horizontally to remove "mirror" effect

        image = frame[5: region_of_interest - 5, 
                    width - region_of_interest + 5: width - 5
                    ] # extract image from region of interest
        image = np.array([image]).astype('float64') / 255.0 # normalize image values
        
        prediction = model.predict(image) # Attempt to determine player's move
        moveIndex = np.argmax(prediction[0]) # index of the image's predicted class
        prediction_label = image_labels[moveIndex] # enumeration of the predicted class
        prob = np.max(prediction[0]) # Get the confidence rating of the prediction
        
        if prob >= confidence_threshold: # Make sure the confidence level is sufficient
            moves_buffer.appendleft(prediction_label.value) # add the move to deque list from left
            try: # Get the mode of the queue of predictions.
                player_move = ImageClass(stats.mode(moves_buffer)[0][0]) 
            except:
                print('Warning: exception thrown by scipy.stats')
                continue

            # use the predicition as the player's move if they have removed their hand from view since the last move 
            if player_move != ImageClass.nothing and hand_detected == False:
                hand_detected = True # Used to lock this subroutine until the user has lowered their hand 
                computer_move = choice([ImageClass.rock,
                                    ImageClass.paper,
                                    ImageClass.scissors]
                                    ) # Computer chooses a move at random
                winner = determine_winner(player_move, computer_move) # Determine the winner.
                matches_left -= 1 # decrement game match counter

                image_box_color = {Players.computer: (0, 0, 255), # Red when the computer wins
                                   Players.player: (0, 250, 0), # Green when the player wins
                                   None: (255, 250, 255)   # White for a tie.
                                  }[winner] # Change the color of the image outline accordingly
                scoreboard[winner] += 1 # Add a point to the match-winner's score
                
                if matches_left == 0: # Go to game over screen
                    continue_flag = True

            elif player_move == ImageClass.nothing:
                hand_detected = False # Unlock the subroutine that determines the player's move 
                image_box_color = (255, 0, 0) # Reset the image region frame color to neutral setting

        cv2.rectangle(frame, (width - region_of_interest, 0),
        (width, region_of_interest), image_box_color, 2)

        if player_move != None and player_move != ImageClass.nothing:
            ui_text_a2.configure(text="Player's Move: {}".format(player_move.name))
            ui_text_b2.configure(text="Computer's Move: {}".format(computer_move.name))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        image_tklabel.configure(image=frame)

        ui_text_a1.configure(text="Player: {} \t".format(scoreboard[Players.player]))
        ui_text_b1.configure(text="Computer: {} \t".format(scoreboard[Players.computer]))
        ui_text_c1.configure(text="Matches left: {} ".format(matches_left))

        tk.update_idletasks()
        tk.update()

    for child in window.winfo_children():
        child.destroy()
    capture.release()
    return quit_flag, scoreboard


def game_over_screen(window, tk, player_score, computer_score):
    def exit_screen(quit):
        nonlocal continue_flag; continue_flag = True
        nonlocal quit_flag; quit_flag = quit

    ## UI ATTRIBUTES
    window_name = "Game over!"
    window.master.title(window_name)
    ## UI ELEMENTS
    ### Outcome image
    window_frame = Frame(window, relief=RAISED, borderwidth=1)
    window_frame.pack(fill=BOTH, expand=True)
    image_tklabel = Label(window_frame)
    image_tklabel.pack()
    window.pack(fill=BOTH, expand=True)
    ### Buttons / Etc.
    quit_button = Button(window_frame, text="Quit", command=lambda m=True: exit_screen(quit=m))
    quit_button.pack(side=RIGHT, padx=5, pady=5)
    continueButton = Button(window_frame, text="Play Again", command=lambda m=False : exit_screen(quit=m))
    continueButton.pack(side=RIGHT, padx=5, pady=5)
    
    # OTHER ROUTINE DATA
    continue_flag = False
    quit_flag = False

    if player_score > computer_score:
        frame = cv2.imread("images/win.png")
    elif player_score < computer_score:
        frame = cv2.imread("images/lose.png")
    else:
        frame = cv2.imread("images/tie.png")

    cv2.resize(frame, (640, 480))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(image=frame)
    image_tklabel.configure(image=frame)

    while not continue_flag:
        tk.update_idletasks()
        tk.update()

    for child in window.winfo_children():
        child.destroy()

    return quit_flag


if __name__ == '__main__':

    def quit(tk: Tk):
        for child in tk.winfo_children():
            child.destroy()
        tk.destroy()
        sys.exit(0)

    input_str = None
    while input_str not in ['y', 'n']:
        input_str = input(
            "Would you like to use a pre-existing image recognition model? (Y/n): ").lower()
        if input_str == "y":
            try:
                model = load_model('model')
                load_new_model = False
            except:
                load_model_message = "Failed to find/load model. Please take new training images..."
                load_new_model = True
            break
        elif input_str == "n":
            load_model_message = "No model selected. Please take new training images..."
            load_new_model = True
            break

    tk = Tk()
    tk.geometry("960x540+960+540")
    window = Frame()
    window.style = Style()
    window.style.theme_use("default")

    if load_new_model:
        print(load_model_message)

        collect_training_images_for_model(100, window, tk)
        model = build_model(image_caches[ImageClass.rock],
                        image_caches[ImageClass.paper],
                        image_caches[ImageClass.scissors],
                        image_caches[ImageClass.nothing],
                        window, tk
                        )
        del image_caches
        model.save('model')

    label_names = list(ImageClass._member_names_)
    label_names.sort()
    if check_model(model, label_names, window, tk): quit(tk)

    quit_flag = False
    while not quit_flag:
        quit_flag, scoreboard = play_game(window, tk)
        if not quit_flag: 
            quit_flag = game_over_screen(window, tk, scoreboard[Players.player], scoreboard[Players.computer])

    quit(tk)