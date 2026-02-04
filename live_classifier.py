import cv2
import numpy as np
import tensorflow as tf

# --- 1. Load the Trained Model ---
MODEL_PATH = 'waste_classifier_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Define Class Labels ---
# Make sure this order matches the training order from `flow_from_directory`
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 3. Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

# --- 4. Live Classification Loop ---
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # --- Pre-process the frame for the model ---
    # Resize the frame to the model's expected input size (224x224)
    img_resized = cv2.resize(frame, (224, 224))
    
    # Convert the image to a numpy array and rescale
    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
    
    # The model expects a batch of images, so add a dimension
    img_array = np.expand_dims(img_array, axis=0)

    # --- Make a prediction ---
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction)

    # --- Display the result on the frame ---
    # Prepare the text to display
    display_text = f"{predicted_class_name}: {confidence:.2f}"
    
    # Set font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (20, 50)
    font_scale = 1.5
    font_color = (0, 255, 0) # Green
    line_type = 2

    # Put the text on the frame
    cv2.putText(frame, display_text, position, font, font_scale, font_color, line_type)

    # Display the resulting frame
    cv2.imshow('Live Waste Classification - Press Q to Quit', frame)

    # --- Exit on 'q' key press ---
    if cv2.waitKey(1) == ord('q'):
        break

# --- 5. Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
