import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained models (ensure they are saved before running this part)
@st.cache_resource
def load_models():
    return {
        'densenet': load_model("C:/Guvi Projects/emotional/image/densenet_model.keras"),
        'resnet': load_model("C:/Guvi Projects/emotional/image/resnet_model.keras"),
        'vgg': load_model("C:/Guvi Projects/emotional/image/vgg_model.keras"),
        'mobilenet': load_model("C:/Guvi Projects/emotional/image/mobilenet_model.keras"),
        'efficientnet': load_model("C:/Guvi Projects/emotional/image/efficientnet_model.keras"),
        'custom_cnn': load_model("C:/Guvi Projects/emotional/image/custom_cnn_model.keras")
    }


# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]  # Get height and width; ignore channels if present
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    return image

# Function to predict emotion using a given model
def predict_emotion(model, image):
    image_resized = cv2.resize(image, (48, 48))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    image_rgb = np.expand_dims(image_rgb, axis=0)  # Add batch dimension
    image_rgb = image_rgb / 255.0  # Normalize pixel values
    prediction = model.predict(image_rgb)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

def main():
    st.title("Emotion Detection from Uploaded Images")

    # Load models once at the start
    models = load_models()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a page:", ["Project Overview", "Facial Detection", "Emotion Classification", "Ethical Analysis"])

    if options == "Project Overview":
        st.header("Project Overview")
        st.write("""
            This project aims to develop a comprehensive system that enables users to upload an image 
            through a Streamlit application and accurately detect and classify the emotion present in 
            the image using Convolutional Neural Networks (CNNs).

            ### Key Components:
            - User Interface Development
            - Facial Detection Implementation
            - Facial Feature Extraction using Mediapipe
            - Emotion Classification using various CNN models including:
                - DenseNet
                - ResNet
                - VGG
                - MobileNet
                - EfficientNet
                - Custom CNN

            ### Applications:
            This system can be applied in various fields such as healthcare, education, and customer service 
            where understanding human emotions is crucial.
        """)

    elif options == "Facial Detection":
        st.header("Facial Detection")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Detecting faces...")
            image_with_faces = detect_faces(image)
            st.image(image_with_faces, caption='Processed Image with Detected Faces', use_column_width=True)

    elif options == "Emotion Classification":
        st.header("Emotion Classification")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            # Load and check the uploaded image
            image = np.array(Image.open(uploaded_file))
            
            # Check if the image is valid
            if image is None or len(image.shape) < 2:
                st.error("Uploaded file is not a valid image.")
                return
            
            # Check for empty images
            if image.size == 0:
                st.error("Uploaded image is empty.")
                return
            
            # Convert to grayscale only if it's not already grayscale
            if len(image.shape) == 2:  # Already grayscale
                gray_image = image
            else:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Classifying emotions...")

            # Predict emotions using different models
            emotions_predictions = {}
            for model_name in models.keys():
                emotions_predictions[model_name] = predict_emotion(models[model_name], gray_image)

            # Display the detected emotions in a user-friendly format
            st.header("Detected Emotions")
            for model_name in emotions_predictions.keys():
                st.write(f"{model_name}: {emotions_predictions[model_name]}")
    elif options == "Ethical Analysis":
        st.title("Ethical Analysis")
        st.write("""
        ### Ethical Considerations in Emotion Detection
        Emotion detection technology has significant potential applications in various fields, but it also raises important ethical considerations. Here are some key points to consider:

        #### Privacy Concerns:
        - **Data Collection**: Collecting and storing images of individuals can raise privacy concerns. It is essential to ensure that users are aware of how their data will be used and stored.
        - **Consent**: Obtaining explicit consent from users before collecting their images is crucial. Users should have the option to opt-out at any time.

        #### Bias and Fairness:
        - **Bias in Training Data**: Emotion detection models can inherit biases present in the training data. It is important to use diverse and representative datasets to train the models.
        - **Fairness**: The models should be evaluated for fairness across different demographic groups to ensure that they do not disproportionately affect certain groups.

        #### Ethical Use:
        - **Applications**: The use of emotion detection technology should be carefully considered in sensitive areas such as healthcare and law enforcement. The potential for misuse should be minimized.
        - **Transparency**: Users should be informed about how the technology works and its limitations. Transparency can help build trust and ensure responsible use.

        #### Mitigation Strategies:
        - **Data Anonymization**: Anonymizing data can help protect user privacy.
        - **Regular Audits**: Conducting regular audits of the models and their performance can help identify and address biases.
        - **User Education**: Educating users about the ethical implications of emotion detection technology can promote responsible use.

        By addressing these ethical considerations, we can ensure that emotion detection technology is used responsibly and benefits society as a whole.
    """)
                
if __name__ == "__main__":
    main()