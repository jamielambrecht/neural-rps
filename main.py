
# STANDARD LIBRARIES
from collections import deque
from enum import IntEnum
from random import choice, shuffle
import sys

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


class ImageClass(IntEnum):
    rock = 0,
    paper = 1,
    scissors = 2,
    nothing = 3


image_caches = {
    ImageClass.rock:     [],
    ImageClass.paper:    [],
    ImageClass.scissors: [],
    ImageClass.nothing:  []
}

class Players(IntEnum):
    player = 0,
    computer = 1

windowTitle = "Rock, Paper, Scissors"

def collect_training_images_for_model(samples_amount):

    user_input_key_mappings = {
        ord('r') : ImageClass.rock,
        ord('p') : ImageClass.paper,
        ord('s') : ImageClass.scissors,
        ord('n') : ImageClass.nothing,
        ord('q') : None
    }

    image_class = None
    capture = cv2.VideoCapture(0)
    start_recording = False
    region_of_interest_side_length = 234  # subtract 10px to get image size
    frame_width = int(capture.get(3))
    window_name = "Collecting training data images"

    while True:

        frame_received, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        if not frame_received:
            print("Error reading from camera!")
            sys.exit(1)

        # Reset capture process when enough samples are collected
        if image_class != None:
            if len(image_caches[image_class]) == samples_amount:
                start_recording = False

        cv2.rectangle(frame,
                      (frame_width - region_of_interest_side_length, 0),
                      (frame_width, region_of_interest_side_length),
                      (0, 250, 150),
                      2
                      )  # Draw a box to show the region of interest

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Open the window

        if start_recording:
            image = frame[5: region_of_interest_side_length - 5,
                          frame_width - region_of_interest_side_length + 5: frame_width - 5
                          ] # Store the captured image frame (crop 5px from all sides of frame region)

            image_caches[image_class].append([image, str(image_class)])

            text = "Collected Samples of {}: {}".format(
                str(image_class.name), len(image_caches[image_class]))

        else:
            text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissors and 'n' for nothing"


        cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA) # UI text

        cv2.imshow(window_name, frame) # Display the window

        user_input_key = int(cv2.waitKey(1))

        if not start_recording and user_input_key in user_input_key_mappings:
            image_class = user_input_key_mappings[user_input_key]
            if image_class != None:
                start_recording = True
            else: 
                break


    #  Release the camera and destroy the window
    capture.release()
    cv2.destroyAllWindows()


def build_model(rock, paper, scissors, nothing):

    image_labels = [tupl[1] for tupl in rock + paper + scissors + nothing]
    images = [tupl[0] for tupl in rock + paper + scissors + nothing]

    images = np.array(images, dtype="float") / 255.0 # Normalize the image data.

    print('Total images: {}\tTotal Labels: {}'.format(len(image_labels), len(images)))
    
    label_encoder = LabelEncoder() 
    # IMPORTANT: Each value for the field in the tuple is a string that will be enumerated
    # beginning at 1 in alphabetical order, i.e 1. nothing, 2. paper, 3. rock, 4. scissors. 
    labels_sorted_strings_enum = label_encoder.fit_transform(image_labels) 
    
    # Convert the label enumerators into one-hot format, i.e. 0 = [1,0,0,0], etc.
    one_hot_labels = to_categorical(labels_sorted_strings_enum, 4)
    
    # Split the data, allocate 75% of the data for training and 25% for testing.
    (trainX, testX, trainY, testY) = train_test_split(
        images, one_hot_labels, test_size=0.25, random_state=50)

    images = [] # Free up memory by clearing the raw images from RAM.
    image_size = 224 # Our model accepts this image size.

    # Loading pre-trained NASNETMobile Model, exluding the head via include_top = False
    nasnet_mobile = tf.keras.applications.NASNetMobile(input_shape=(
        image_size, image_size, 3), include_top=False, weights='imagenet')

    nasnet_mobile.trainable = False # Freeze the model

    nnm_output = nasnet_mobile.output # Custom head based on output feature maps from NASNETMobile.
    nnm_output = GlobalAveragePooling2D()(nnm_output) # use global average pooling to minimize overfitting
    nnm_output = Dense(712, activation='relu')(nnm_output) # a dense neural network layer with 712-d output vector
    nnm_output = Dropout(0.40)(nnm_output) # Drop 40% of activations to reduce overfitting
    nnm_output = Dense(4, activation='softmax')(nnm_output) # a probability distribution with 4 output classes.

    model = Model(inputs=nasnet_mobile.input, outputs=nnm_output)

    print("Number of layers in model: {}".format(len(model.layers[:])))

    augment = ImageDataGenerator(
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

    # Start training
    model.fit(x=augment.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size, epochs=epochs)

    return model


def check_model(model, label_names):
    cap = cv2.VideoCapture(0)
    region_of_interest_side_length = 234
    width = int(cap.get(3))

    while True:

        sampleCollected, frame = cap.read()
        if not sampleCollected:
            break

        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (width - region_of_interest_side_length, 0),
                      (width, region_of_interest_side_length), (0, 250, 150), 2)

        cv2.namedWindow(windowTitle, cv2.WINDOW_NORMAL)

        image = frame[5: region_of_interest_side_length - 5, width - region_of_interest_side_length + 5: width - 5]
        image = np.array([image]).astype('float64') / 255.0 # Normalize the image and convert to float64 array.
        prediction = model.predict(image)
        prediction_label_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        print(label_names[prediction_label_index])
        cv2.putText(frame, "Detected image class: {} {:.2f}%".format(label_names[prediction_label_index], confidence*100),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(windowTitle, frame)

        user_input_key = cv2.waitKey(1)
        if user_input_key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def determine_winner(player_move: ImageClass, computer_move: ImageClass):

    if player_move.value == computer_move.value:
        return None
    elif (player_move.value + 1) % 3 == computer_move.value:
        print(str((player_move.value + 1) % 3) + " " + str(computer_move.value))
        return Players.computer
    else:
        return Players.player
"""
rock     0      2
paper    1      1
scissors 2      0
"""

def display_winner(player_score, computer_score):

    if player_score > computer_score:
        image = cv2.imread("images/win.png")
    elif player_score < computer_score:
        image = cv2.imread("images/lose.png")
    else:
        image = cv2.imread("images/tie.png")

    origin_y = 530
    for text_content in ["Press 'ENTER' to play again.", "Press any other key to quit."]:
        cv2.putText(image, text_content, (150, origin_y), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        origin_y += 30

    cv2.imshow(windowTitle, image)

    user_input_key = cv2.waitKey(0)

    return True if user_input_key == 13 else False


if __name__ == '__main__':

    ######################### INITIAL SET UP #########################
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
        
    if load_new_model:
        print(load_model_message)
        samples_amount = 100 # TODO Consider allowing the user to configure this
        collect_training_images_for_model(samples_amount)
        model = build_model(image_caches[ImageClass.rock],
                        image_caches[ImageClass.paper],
                        image_caches[ImageClass.scissors],
                        image_caches[ImageClass.nothing]
                        )
        model.save('model')

    input_str = None
    while input_str not in ['y', 'n']:
        input_str = input("Would you like to check the current image recognition model for accuracy? (Y/n): ").lower()
        if input_str == "y":
            image_class_names = list(ImageClass._member_names_)
            image_class_names.sort()
            check_model(model, image_class_names)
            break
        elif input_str == "n":
            break

    ######################## PLAYING THE GAME ########################

    capture = cv2.VideoCapture(0)
    region_of_interest = 234
    width = int(capture.get(3))

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
    smooth_factor = 5 # use the mode of x predictions to reduce false positives
    moves_buffer = deque([ImageClass.nothing] * smooth_factor, maxlen=smooth_factor)

    while True:

        sampleCollected, frame = capture.read()
        if not sampleCollected: # break the loop iif there is an error reading frames
            break
        frame = cv2.flip(frame, 1) # Flip horizontally to remove "mirror" effect
        cv2.namedWindow(windowTitle, cv2.WINDOW_NORMAL) # Launch the application window

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
                print(player_move.name)
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
                print("Player's move: " + str(player_move.name)) # Display player move
                print("Computer's move: " + str(computer_move.name)) # Display CPU move

                image_box_color = {Players.computer: (0, 0, 255), # Red when the computer wins
                                   Players.player: (0, 250, 0), # Green when the player wins
                                   None: (255, 250, 255)   # White for a tie.
                                  }[winner] # Change the color of the image outline accordingly
                scoreboard[winner] += 1 # Add a point to the match-winner's score
                
                if matches_left == 0: # Show the outcome of the game
                    play_again = display_winner(scoreboard[Players.player], scoreboard[Players.computer])
                    if play_again: # Reset game variables if the game is reset
                        scoreboard, matches_left = dict((key, 0) for key in scoreboard), attempts
                    else: # Otherwise, quit the game.
                        break

            elif player_move == ImageClass.nothing:
                hand_detected = False # Unlock the subroutine that determines the player's move 
                image_box_color = (255, 0, 0) # Reset the image region frame color to neutral setting

        # UI ELEMENTS ================================================================================================
        cv2.putText(frame, "Player's Move: " + player_move.name,
                    (420, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "Computer's Move: " + computer_move.name,
                    (2, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "Player's Score: " + str(scoreboard[Players.player]),
                    (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "Computer's Score: " + str(scoreboard[Players.computer]),
                    (2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "Matches left: {}".format(matches_left), (190, 400), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (100, 2, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, (width - region_of_interest, 0),
                    (width, region_of_interest), image_box_color, 2)
        # END UI ELEMENTS =============================================================================================

        cv2.imshow(windowTitle, frame) # Display the image
        userInputKey = cv2.waitKey(10) # Delay for input event handling.
        if userInputKey == ord('q'): break # Press 'q' to quit the game.

    capture.release()
    cv2.destroyAllWindows()