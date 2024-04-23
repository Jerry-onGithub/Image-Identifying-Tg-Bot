import telebot
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Initialize the Telegram Bot
bot = telebot.TeleBot("712XXXX955:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Load the pre-trained model for image classification
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Function to classify items in the image
def classify_image(image_url):
    try:
        # Download and preprocess the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = image.resize((299, 299))  # InceptionV3 input size
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Perform inference to classify the image
        predictions = model.predict(image)

        # Decode the prediction
        decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions)

        # Extract the top prediction
        top_prediction = decoded_predictions[0][0]

        # Get the class label of the top prediction
        item = top_prediction[1]

        return item
    except Exception as e:
        print("Error:", e)
        return "Unable to classify the image"

# Define the command handler for receiving images
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Get the file ID of the photo
        file_id = message.photo[-1].file_id

        # Get the file path of the photo
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path

        # Download the photo from Telegram servers
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_path}"

        # Classify the image
        item = classify_image(file_url)

        # Send the result back to the user
        bot.reply_to(message, f"The item in the image is: {item}")
    except Exception as e:
        print("Error:", e)
        bot.reply_to(message, "Sorry, something went wrong.")

# Start the bot
bot.polling()
