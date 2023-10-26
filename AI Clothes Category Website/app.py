# import the necessary packages
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
from flask import send_from_directory

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')
model = None

def load_model():
    # load the pre-trained Keras model
    global model
    model = tf.keras.models.load_model('model/fashion_model')

load_model()
    
def prepare_image(image):
	image = image.convert('L')  # Load as grayscale

	# Set the global threshold value
	threshold = 128

	# Perform global (inverse) thresholding
	image = image.point(lambda p: p > threshold and 255)
     
	# Invert the colors
	image = Image.eval(image, lambda p: 255 - p)
      
    #resize the image to match the input layer of the model
	image = image.resize((28, 28))
     
    #normalize the data
	new_image = np.array(image)
	new_image = new_image.astype('float32')
	new_image /= 255

	return new_image.reshape(-1, 28, 28, 1)

#returns the given value from the model's top prediction in a more readable fomat
def confidenceChance(confidence):
    numstr = str(confidence)

    if (numstr[0] == '1'):
        numstr = '99%'
    else:
        numstr = numstr[2:4] + '%'
    return numstr

#returns static images to the user
@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory('static', filename)

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        file = flask.request.files['user_image']

        #check for exceptions here such as
        #a corrupted jpg file
        try:
            #read the image in PIL format
            image = Image.open(file)
        except Exception as e:
            #add text saying that the image submitted doesn't work
            return flask.render_template('main.html', badfile=1)

        #save image differently for jpeg and pngs
        # if (image.format.lower() == 'jpeg'):
        #     image.save('images/user_image.jpeg')
        # elif(image.format.lower() == 'png'):
        #     image.save('images/user_image.png')
        
        #preproccess image for model. 
        image = prepare_image(image)
        
        #pass the image as a list to the model to make a prediction
        prediction = model.predict([image])
        
        #find the prediction with the highest chance
        categories = ['tshirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
        max_value = -1
        index = 0
        max_index = 0
        for i in prediction[0]:
            if (i > max_value):
                max_value = i
                max_index = index
            index += 1
    
        # Render the form again, but add in the prediction
        return flask.render_template('main.html', confidence=confidenceChance(max_value),result= categories[max_index])


if __name__ == '__main__':
	app.run()