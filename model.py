from fastai.vision.all import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import uvicorn

# Load the trained ResNet model using FastAI
learn = load_learner(
    "resnet.pkl"
)  # Ensure the model is saved as resnet.pkl after training

# Dictionary of average shelf life in days
shelf_life_dict = {
    "freshapples": 30,
    "freshbanana": 7,
    "freshcucumber": 10,
    "freshokra": 8,
    "freshoranges": 21,
    "freshpatato": 60,
    "freshtamto": 14,
    "rottenapples": 0,
    "rottenbanana": 0,
    "rottencucumber": 0,
    "rottenokra": 0,
    "rottenoranges": 0,
    "rottenpatato": 0,
    "rottentamto": 0,
}

# FastAPI initialization
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Helper function to decode base64 image
def decode_image(image_base64: str) -> Image.Image:
    try:
        # Check if the base64 string contains a metadata prefix and remove it
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode the base64 string into bytes
        image_data = base64.b64decode(image_base64)

        # Convert bytes into a BytesIO object and then into a PIL image
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")  # Ensure it's in RGB mode

        return image
    except Exception as e:
        raise ValueError(f"Could not decode the image: {str(e)}")


# Function to predict freshness and shelf life
def predict_freshness_and_shelf_life(img: Image.Image):
    # Convert the PIL.Image.Image object to a BytesIO stream
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")  # Save as JPEG or the appropriate format
    img_byte_arr = img_byte_arr.getvalue()

    # Create a FastAI PILImage from the byte stream
    img_fastai = PILImage.create(BytesIO(img_byte_arr))

    prediction, _, probs = learn.predict(img_fastai)

    predicted_freshness = prediction
    freshness_percentage = float(probs.max() * 100)

    # Retrieve the shelf life based on prediction
    shelf_life_days = shelf_life_dict[predicted_freshness]

    # Create result
    if shelf_life_days == 0:
        result_text = {
            "status": "rotten",
            "item": predicted_freshness.split("rotten")[1],
            "freshness_percentage": freshness_percentage,
            "shelf_life_days": shelf_life_days,
            "message": f"The item is classified as {predicted_freshness.split('rotten')[1]} and is already rotten.",
        }
    else:
        result_text = {
            "status": "fresh",
            "item": predicted_freshness.split("fresh")[1],
            "freshness_percentage": freshness_percentage,
            "shelf_life_days": shelf_life_days,
            "message": (
                f"The item is classified as {predicted_freshness.split('fresh')[1]}.\n"
                f"Estimated freshness: {freshness_percentage:.2f}%\n"
                f"Estimated shelf life: {shelf_life_days} days remaining."
            ),
        }
    return result_text


# Request model for image input
class ImageInput(BaseModel):
    image_base64: str  # Base64 encoded image


@app.get("/")
async def root():
    return {"message": "Welcome to the Freshness Prediction API!"}


# API endpoint for image prediction
@app.post("/predict/")
async def predict(image_input: ImageInput):
    try:
        # Decode the base64 image
        image = decode_image(image_input.image_base64)

        # Make prediction
        result = predict_freshness_and_shelf_life(image)

        # Return result as JSON response
        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
