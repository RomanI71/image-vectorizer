from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
# --- 💡 NEW IMPORT: StaticFiles for serving static assets ---
from fastapi.staticfiles import StaticFiles 
# PIL imports updated to include necessary modules from vector.py
from PIL import Image, ImageDraw, ImageFilter, ImageOps, Image as PILImage 
import numpy as np
import cv2
import io
import os
import uuid
import datetime
import threading
import time
import logging
import logging.handlers
import traceback
import base64 # Added for SVG embedding
import re     # Added for hex color validation

# -------- AI Models -------- #
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
    print("✅ Rembg AI model loaded successfully")
except ImportError:
    REMBG_AVAILABLE = False
    print("❌ Rembg not available, using fallback methods")

# ----------------- Logging Setup ----------------- #
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
logger = logging.getLogger("AI_API") # Changed logger name for combined API
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(log_folder, "app.log"),
    maxBytes=5*1024*1024,
    backupCount=5
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------- Folders ----------------- #
UPLOAD_FOLDER = 'uploads'
REMOVEBG_FOLDER = 'removebg'
VECTOR_FOLDER = 'vectorized' # Added folder for SVG outputs
# --- 📁 NEW FOLDER: Static folder assumed for Index.html ---
STATIC_FOLDER = 'static'
for folder in [UPLOAD_FOLDER, REMOVEBG_FOLDER, VECTOR_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}

# ----------------- FastAPI App ----------------- #
app = FastAPI(title="AI Image Processing API (BG Removal & SVG Converter)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------------------------------
# --- Core SVG Logic Class (Copied from vector.py) ---
# ----------------------------------------------------
class SVGProcessor:
    def __init__(self):
        # Use PILImage alias to match vector.py's implementation
        self.resampling_method = PILImage.Resampling.LANCZOS if hasattr(PILImage, 'Resampling') else PILImage.LANCZOS

    async def colorful_svg_conversion(self, image_data, simplification: int, color_palette_size: int) -> str:
        """
        Convert image to a colorful vector-like SVG with quantization and simplification.
        """
        try:
            # 1. Open and Process Image with PIL
            image = PILImage.open(io.BytesIO(image_data))
            
            # Ensure image is in RGB format
            image = image.convert('RGB')
            
            # 2. Color Quantization - Reduce colors for vector-like appearance
            if color_palette_size > 0 and color_palette_size < 256:
                # Quantize to limited color palette
                quantized_image = image.quantize(colors=color_palette_size).convert('RGB')
            else:
                quantized_image = image
            
            # 3. Image Simplification (safe smoothing)
            processed_image = quantized_image
            if simplification > 0:
                # Use safe simplification that won't crash
                processed_image = self._safe_simplify(quantized_image, simplification)
            
            # 4. Resize for optimization
            max_size = 1200
            if max(processed_image.size) > max_size:
                processed_image.thumbnail((max_size, max_size), self.resampling_method)
            
            # 5. Convert to PNG and Base64 for SVG embedding
            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 6. Create SVG with embedded PNG image
            final_width, final_height = processed_image.size
            svg_content = f'''<svg width="{final_width}" height="{final_height}" viewBox="0 0 {final_width} {final_height}" xmlns="http://www.w3.org/2000/svg">
                <image href="data:image/png;base64,{img_str}" width="100%" height="100%"/>
            </svg>'''
            
            return svg_content
            
        except (IOError, OSError, ValueError) as e:
            logger.error(f"PIL/Image Processing ERROR in colorful_svg: {e}") 
            raise HTTPException(status_code=422, detail=f"Image processing failed. Is the file valid? Error: {e}")
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR in colorful_svg: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error during conversion.")

    def _safe_simplify(self, image, simplification_level):
        """Safe image simplification without causing filter size errors"""
        try:
            # Convert simplification level (1-10) to appropriate filter parameters
            if simplification_level <= 3:
                # Light smoothing
                return image.filter(ImageFilter.SMOOTH)
            elif simplification_level <= 6:
                # Medium smoothing
                return image.filter(ImageFilter.SMOOTH_MORE)
            else:
                # Strong smoothing with Gaussian blur (safe)
                blur_radius = min(2.0, (simplification_level - 6) * 0.5)
                return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except Exception as e:
            logger.error(f"Image simplification failed: {e}")
            # If any filter fails, return original image
            return image

    async def threshold_svg_conversion(self, image_data, threshold: int, stroke_color: str) -> str:
        """
        Original threshold-based conversion for black and white vector effect
        """
        try:
            # 1. Open and Process Image with PIL
            image = PILImage.open(io.BytesIO(image_data))
            
            # Ensure image is in RGB format
            image = image.convert('RGB')
            
            # Convert to grayscale for thresholding
            grayscale_image = image.convert('L')
            
            # Apply thresholding
            threshold_image = grayscale_image.point(
                lambda x: 0 if x < threshold else 255
            )
            
            # 2. Prepare Final Output Image
            width, height = image.size
            final_image = PILImage.new('RGB', (width, height), color='white')
            
            # Parse the hex color
            stroke_rgb = tuple(int(stroke_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            pixels = threshold_image.load()
            final_pixels = final_image.load()
            
            for x in range(width):
                for y in range(height):
                    # If the pixel is black (0) after thresholding, color it with stroke_color
                    if pixels[x, y] == 0:
                        final_pixels[x, y] = stroke_rgb
            
            # 3. Resize and Prepare for Base64
            max_size = 750
            if max(final_image.size) > max_size:
                final_image.thumbnail((max_size, max_size), self.resampling_method)
            
            buffered = io.BytesIO()
            # Use PNG for better quality with sharp edges
            final_image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 4. Create SVG with embedded PNG image
            final_width, final_height = final_image.size
            svg_content = f'''<svg width="{final_width}" height="{final_height}" viewBox="0 0 {final_width} {final_height}" xmlns="http://www.w3.org/2000/svg">
                <image href="data:image/png;base64,{img_str}" width="100%" height="100%"/>
            </svg>'''
            
            return svg_content
            
        except Exception as e:
            logger.error(f"Threshold conversion error: {e}")
            raise HTTPException(status_code=500, detail=f"Threshold processing failed: {str(e)}")

# Instantiate the SVG Processor
svg_processor = SVGProcessor() 

# ----------------- Utility Functions (BG Removal) ----------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 1. 🖼️ সাবজেক্টের সীমানা অনুযায়ী ক্রপ করা
def crop_to_subject(image: PILImage.Image) -> PILImage.Image:
    """Crops the image to the smallest bounding box containing non-transparent pixels."""
    try:
        if image.mode != 'RGBA':
            return image
            
        alpha = np.array(image)[:, :, 3]
        coords = np.argwhere(alpha > 0)
        
        if coords.size == 0:
            return image
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add a small padding
        padding = 10
        width, height = image.size
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding + 1)
        y_max = min(height, y_max + padding + 1)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        logger.info(f"Image cropped from {width}x{height} to {cropped_image.size[0]}x{cropped_image.size[1]}.")
        return cropped_image
        
    except Exception as e:
        logger.error(f"Image cropping failed: {e}")
        return image

# 2. 🎨 উন্নত Refine Edge: রঙের দূষণ অপসারণ (Decontamination)
def decontaminate_foreground(image: PILImage.Image) -> PILImage.Image:
    """
    Advanced decontamination to remove background color bleed/fringing (Refine Edge feature). 
    Increased kernel size for better hair refinement.
    """
    try:
        img_array = np.array(image, dtype=np.float32)
        color = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        # 1. Create a mask of the semi-transparent edge
        edge_mask = (alpha > 0) & (alpha < 255)
        
        # 2. Prepare the color channels for blurring
        opaque_color = color.copy()
        opaque_color[alpha == 0] = 255 

        # 3. Blur the colors for Decontamination
        kernel_size = 11 
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_color = cv2.GaussianBlur(opaque_color.astype(np.uint8), (kernel_size, kernel_size), 0).astype(np.float32)

        # 4. Apply the blurred color (clean foreground) to the edge pixels
        color[edge_mask] = blurred_color[edge_mask]
        
        # 5. Recombine
        img_array[:, :, :3] = color
        
        return PILImage.fromarray(img_array.astype(np.uint8), 'RGBA')
        
    except Exception as e:
        logger.error(f"Decontamination (Refine Edge) failed: {e}")
        return image


# 3. ✨ মাস্ক স্মুথিং
def refine_mask_smoothing(image: PILImage.Image) -> PILImage.Image:
    """Optimized alpha channel smoothing for cleaner, softer edges."""
    try:
        img_array = np.array(image)
        alpha = img_array[:, :, 3].astype(np.float32)
        
        kernel = np.ones((5, 5), np.uint8)
        alpha_uint8 = alpha.astype(np.uint8)
        
        # Clean small holes and noise
        alpha_cleaned = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Smooth the mask for a better transition (Feathering)
        alpha_smoothed = cv2.GaussianBlur(alpha_cleaned.astype(np.float32), (5, 5), 0.5)
        
        img_array[:, :, 3] = np.clip(alpha_smoothed, 0, 255).astype(np.uint8)
        
        return PILImage.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        logger.error(f"Mask smoothing failed: {e}")
        return image

# 4. ⚫ কালো ছায়া দূর করা
def clean_dark_artifacts(image: PILImage.Image) -> PILImage.Image:
    """Removes dark color bleed (shadow artifacts) from semi-transparent edges."""
    try:
        img_array = np.array(image)
        color = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        edge_mask = (alpha > 0) & (alpha < 255)
        
        dark_threshold = 40
        is_dark = (color[:, :, 0] < dark_threshold) & \
                  (color[:, :, 1] < dark_threshold) & \
                  (color[:, :, 2] < dark_threshold)
        
        pixels_to_remove = edge_mask & is_dark

        if np.any(pixels_to_remove):
            img_array[pixels_to_remove, 3] = 0
            logger.info(f"Cleaned {np.sum(pixels_to_remove)} dark edge pixels.")

        return PILImage.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        logger.error(f"Dark artifact cleaning failed: {e}")
        return image


# --------- Main Background Removal Function (Optimized) --------- #
# --- UPDATED: Accepts 'quality' parameter for conditional post-processing ---
def remove_background_optimized(image: PILImage.Image, quality: str) -> PILImage.Image:
    """
    Main function that uses the best model and applies custom cleaning.
    quality='high' includes post-processing (Refine Edge).
    quality='standard' skips post-processing for speed.
    """
    if not REMBG_AVAILABLE:
        logger.warning("AI (rembg) not available, cannot process.")
        return image.convert("RGBA")
        
    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", quality=95)
        img_bytes = buf.getvalue()
        
        # 1. AI Background Removal
        result_bytes = rembg_remove(
            img_bytes, 
            session_name='u2net_human_seg', 
            post_process_mask=True # Rembg's internal feathering/mask refinement
        )
        
        result_img = PILImage.open(io.BytesIO(result_bytes)).convert("RGBA")
        
        # 2. Conditional Post-Processing Pipeline (Refine Edge)
        if quality.lower() == 'high':
            logger.info("Applying advanced post-processing (Refine Edge).")
            result_img = refine_mask_smoothing(result_img)     
            result_img = decontaminate_foreground(result_img) 
            result_img = clean_dark_artifacts(result_img)      
        else:
            logger.info("Skipping advanced post-processing (Standard quality).")
        
        # 3. Final Crop (Applied to both qualities)
        result_img = crop_to_subject(result_img)           
        
        return result_img
        
    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        logger.error(traceback.format_exc())
        return image.convert("RGBA")

# ----------------- Routes ----------------- #

# 1. 🖼️ Static Files কনফিগারেশন
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

# 2. 🏠 হোমপেজ রুট
@app.get("/")
async def root():
    # নিশ্চিত করুন আপনার HTML ফাইলটির নাম Index.html (কেস-সেনসিটিভ)
    index_path = os.path.join(STATIC_FOLDER, "Index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"error": "Index.html not found in static folder"}, status_code=404)
    return FileResponse(index_path)

# 3. API রুট স্ট্যাটাস
@app.get("/api-status") # Renamed from "/" to avoid conflict with FileResponse on "/"
async def api_status():
    return {
        "message": "🚀 AI Image Processing API running! (BG Removal & SVG)", 
        "ai_available": REMBG_AVAILABLE,
        "bg_removal_quality": "Optimized High Quality (Advanced Refine Edge) or Standard (AI only)",
        "svg_conversion": "Available"
    }

# ----------------------------------------------------
# --- SVG Vectorization Routes (Copied from vector.py) ---
# ----------------------------------------------------

@app.post("/vectorize") 
async def convert_to_svg(
    file: UploadFile = File(...),
    simplification: int = Form(2, description="Simplification level (0-10, higher = more simplified)"),
    color_palette_size: int = Form(32, description="Number of colors in palette (0 = original colors)"),
    mode: str = Form("colorful", description="Conversion mode: 'colorful' or 'threshold'")
):
    """Convert uploaded image to SVG with different modes"""
    
    # 1. Basic File Validation
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Must be JPG, PNG, WebP, or GIF.")

    # 2. Validate Parameters
    if not 0 <= simplification <= 10:
        raise HTTPException(status_code=400, detail="Simplification must be between 0 and 10.")

    if not 0 <= color_palette_size <= 256:
        raise HTTPException(status_code=400, detail="Color palette size must be between 0 and 256.")

    # 3. Read File Data
    image_data = await file.read()
    
    try:
        # 4. Process and Convert based on mode
        if mode == "threshold":
            # Use threshold mode with default parameters
            svg_result = await svg_processor.threshold_svg_conversion(
                image_data, 
                threshold=128,
                stroke_color="#000000"
            )
        else:
            # Use colorful mode
            svg_result = await svg_processor.colorful_svg_conversion(
                image_data, 
                simplification=simplification,
                color_palette_size=color_palette_size
            )
        
        # 5. Return SVG response
        return Response(content=svg_result, media_type="image/svg+xml")
    
    except Exception as e:
        logger.error(f"Vectorize general error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        await file.close()

@app.post("/vectorize/threshold")
async def threshold_vectorize(
    file: UploadFile = File(...),
    threshold: int = Form(128, description="Threshold value (0-255)"),
    stroke_color: str = Form("#000000", description="Stroke color in hex format")
):
    """Original threshold-based vectorization"""
    
    # Validate parameters
    if not 0 <= threshold <= 255:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 255.")

    if not re.fullmatch(r'^#([A-Fa-f0-9]{6})$', stroke_color):
        raise HTTPException(status_code=400, detail="Invalid stroke_color format. Must be a 6-digit hex code (e.g., #000000).")

    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_data = await file.read()
    
    try:
        svg_result = await svg_processor.threshold_svg_conversion(
            image_data, 
            threshold=threshold,
            stroke_color=stroke_color
        )
        return Response(content=svg_result, media_type="image/svg+xml")
    except Exception as e:
        logger.error(f"Vectorize threshold error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Threshold conversion failed: {str(e)}")
    finally:
        await file.close()

# ----------------------------------------------------
# --- BG Removal Routes (Existing) ---
# ----------------------------------------------------
@app.get("/get-image/{filename}")
async def get_image(filename: str):
    """Serve the processed image for preview."""
    file_path = os.path.join(REMOVEBG_FOLDER, filename)
    if os.path.exists(file_path):
        if filename.lower().endswith('.png'):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"
            
        return FileResponse(file_path, media_type=media_type)
        
    return JSONResponse({"error": "Preview file not found"}, status_code=404)

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download the processed file."""
    file_path = os.path.join(REMOVEBG_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            filename=filename, 
            media_type='application/octet-stream'
        )
    return JSONResponse({"error": "File not found for download"}, status_code=404)


@app.post("/remove-bg")
async def remove_bg(
    image: UploadFile = File(...), 
    background_color: str = Form(default="transparent"),
    quality: str = Form(default="high", description="Processing quality: 'high' (Refine Edge + AI) or 'standard' (AI only).")
):
    """
    Remove background with improved quality selection
    """
    try:
        if not allowed_file(image.filename):
            return JSONResponse({"error": "File type not allowed"}, status_code=400)
        
        file_bytes = await image.read()
        
        try:
            img = PILImage.open(io.BytesIO(file_bytes))
            img.verify()
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGBA")
        except Exception as e:
            return JSONResponse({"error": f"Invalid image file: {str(e)}"}, status_code=400)
        
        # --- UPDATED CALL: Pass quality parameter ---
        processed_img = remove_background_optimized(img, quality)
        # --------------------------------------------

        # Apply background color if provided
        bg_rgb = None
        output_format = 'PNG'
        
        if background_color.startswith("#") and len(background_color) == 7:
            try:
                bg_rgb = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
            except:
                bg_rgb = None

        if bg_rgb:
            bg_img = PILImage.new("RGB", processed_img.size, bg_rgb)
            bg_img.paste(processed_img, mask=processed_img.split()[3]) 
            processed_img = bg_img.convert('RGB')
            output_format = 'JPEG'
        
        # Save output
        filename = f"nobg_{uuid.uuid4().hex}.{output_format.lower()}"
        output_path = os.path.join(REMOVEBG_FOLDER, filename)
        
        if output_format == 'PNG':
            processed_img.save(output_path, format='PNG', optimize=True)
        else:
            processed_img.save(output_path, format='JPEG', quality=95, optimize=True)
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return JSONResponse({"error": "Failed to save processed image"}, status_code=500)
        
        quality_message = "Optimized quality with Refine Edge feature." if quality.lower() == 'high' else "Standard quality (AI only)."
        
        return {
            "success": True, 
            "filename": filename, 
            "previewUrl": f"/get-image/{filename}",
            "downloadUrl": f"/download/{filename}",
            "message": f"Background removed with {quality_message}", # UPDATED MESSAGE
            "ai_used": REMBG_AVAILABLE,
            "format": output_format,
            "background": background_color
        }
        
    except Exception as e:
        logger.error(f"Remove BG error: {traceback.format_exc()}")
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

# ----------------- Background Cleanup ----------------- #
def cleanup_files():
    while True:
        try:
            now = datetime.datetime.now()
            # --- 🚮 Added STATIC_FOLDER for cleanup ---
            for folder in [UPLOAD_FOLDER, REMOVEBG_FOLDER, VECTOR_FOLDER, STATIC_FOLDER]: 
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    # Skip cleanup for the main Index.html file
                    if folder == STATIC_FOLDER and fname.lower() == 'index.html':
                        continue
                        
                    if os.path.isfile(fpath):
                        ctime = datetime.datetime.fromtimestamp(os.path.getctime(fpath))
                        if (now - ctime).total_seconds() > 3600:
                            os.remove(fpath)
                            logger.info(f"Removed old file: {fpath}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        time.sleep(1800)

threading.Thread(target=cleanup_files, daemon=True).start()

# ----------------- Run ----------------- #
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Image Processing API...")
    print("=" * 60)
    print("✨ Available Services:")
    print("  1. Web Interface (Root: /)")
    print("  2. Background Removal (API: /remove-bg)")
    print("  3. SVG Vectorization (API: /vectorize)")
    print("=" * 60)
    print(f"🤖 AI Model Available: {'✅ Yes' if REMBG_AVAILABLE else '❌ No'}")
    print("🌐 Server URL: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)