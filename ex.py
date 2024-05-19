from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount a directory containing images
app.mount("/cropped_image", StaticFiles(directory="cropped_image"), name="cropped_image")

@app.get("/get_images")
async def get_image():
    # Assuming you have an image stored in the 'images' directory
    # You can change the path as per your directory structure
    image_path = "cropped_image/OIP.jpg"
    return {"d":12,"data":FileResponse(image_path, media_type="image/jpeg")}

@app.get("/")
async def main():
    html_content = """
    <html>
        <head>
            <title>Image Viewer</title>
        </head>
        <body>
            <h1>Image Viewer</h1>
            <img src="/get_image" alt="Sample Image" width="500" height="500">
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

