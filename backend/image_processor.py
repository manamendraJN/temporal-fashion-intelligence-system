# """
# Image Preprocessing - Convert color images to body masks
# Author: manamendraJN
# Date: 2025-11-18
# """

# import cv2
# import numpy as np
# from PIL import Image
# import io
# from rembg import remove

# class ImageProcessor:
#     """Process uploaded images and convert to body masks"""
    
#     def __init__(self):
#         pass
    
#     def remove_background(self, image_bytes):
#         """
#         Remove background from image using AI
        
#         Args:
#             image_bytes: Raw image bytes
        
#         Returns:
#             Image with transparent background (PIL Image)
#         """
#         try:
#             # Load image
#             input_image = Image.open(io.BytesIO(image_bytes))
            
#             # Remove background using rembg (AI-based)
#             output_image = remove(input_image)
            
#             return output_image
#         except Exception as e:
#             print(f"❌ Background removal failed: {e}")
#             # Fallback: return original image
#             return Image.open(io.BytesIO(image_bytes))
    
#     def create_silhouette(self, image_with_transparency):
#         """
#         Convert image with transparent background to silhouette (mask)
        
#         Args:
#             image_with_transparency: PIL Image with alpha channel
        
#         Returns:
#             Grayscale silhouette (numpy array)
#         """
#         # Convert to numpy array
#         img_array = np.array(image_with_transparency)
        
#         # Check if image has alpha channel
#         if img_array.shape[2] == 4:
#             # Extract alpha channel (transparency)
#             alpha = img_array[:, :, 3]
            
#             # Create binary mask (white body, black background)
#             # Where alpha > threshold, set to white (255), else black (0)
#             mask = np.where(alpha > 128, 255, 0).astype(np.uint8)
#         else:
#             # No alpha channel, use grayscale conversion
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#             # Apply threshold
#             _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
#         return mask
    
#     def refine_mask(self, mask):
#         """
#         Refine mask using morphological operations
        
#         Args:
#             mask: Binary mask (numpy array)
        
#         Returns:
#             Refined mask
#         """
#         # Remove noise
#         kernel_small = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
#         # Fill holes
#         kernel_large = np.ones((7, 7), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
#         # Smooth edges
#         mask = cv2.GaussianBlur(mask, (5, 5), 0)
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
#         return mask
    
#     def resize_mask(self, mask, target_size=(512, 384)):
#         """
#         Resize mask to target size
        
#         Args:
#             mask: Binary mask
#             target_size: (height, width)
        
#         Returns:
#             Resized mask
#         """
#         # Resize maintaining aspect ratio
#         mask = cv2.resize(mask, (target_size[1], target_size[0]), 
#                          interpolation=cv2.INTER_LINEAR)
#         return mask
    
#     def process_image(self, image_bytes, target_size=(512, 384)):
#         """
#         Complete image processing pipeline
        
#         Args:
#             image_bytes: Raw image bytes from upload
#             target_size: Target size (height, width)
        
#         Returns:
#             Processed mask as bytes (PNG format)
#         """
#         try:
#             # Step 1: Remove background
#             img_no_bg = self.remove_background(image_bytes)
            
#             # Step 2: Create silhouette
#             mask = self.create_silhouette(img_no_bg)
            
#             # Step 3: Refine mask
#             mask = self.refine_mask(mask)
            
#             # Step 4: Resize to target size
#             mask = self.resize_mask(mask, target_size)
            
#             # Convert back to bytes (PNG format)
#             mask_pil = Image.fromarray(mask)
#             output_buffer = io.BytesIO()
#             mask_pil.save(output_buffer, format='PNG')
#             mask_bytes = output_buffer.getvalue()
            
#             return mask_bytes
            
#         except Exception as e:
#             print(f"❌ Image processing failed: {e}")
#             raise
    
#     def process_and_preview(self, image_bytes, target_size=(512, 384)):
#         """
#         Process image and return both mask and preview
        
#         Returns:
#             dict with 'mask_bytes' and 'preview_bytes'
#         """
#         try:
#             # Remove background
#             img_no_bg = self.remove_background(image_bytes)
            
#             # Create mask
#             mask = self.create_silhouette(img_no_bg)
#             mask = self.refine_mask(mask)
#             mask = self.resize_mask(mask, target_size)
            
#             # Create preview (side-by-side comparison)
#             original = Image.open(io.BytesIO(image_bytes))
#             original = original.resize((target_size[1], target_size[0]))
#             original_array = np.array(original.convert('RGB'))
            
#             # Create colored mask for preview
#             mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
#             mask_colored[:, :, 0] = 0  # Remove red
#             mask_colored[:, :, 1] = mask  # Green channel
#             mask_colored[:, :, 2] = 0  # Remove blue
            
#             # Blend original with mask
#             blended = cv2.addWeighted(original_array, 0.6, mask_colored, 0.4, 0)
            
#             # Convert to bytes
#             mask_pil = Image.fromarray(mask)
#             preview_pil = Image.fromarray(blended)
            
#             mask_buffer = io.BytesIO()
#             preview_buffer = io.BytesIO()
            
#             mask_pil.save(mask_buffer, format='PNG')
#             preview_pil.save(preview_buffer, format='PNG')
            
#             return {
#                 'mask_bytes': mask_buffer.getvalue(),
#                 'preview_bytes': preview_buffer.getvalue()
#             }
            
#         except Exception as e:
#             print(f"❌ Process and preview failed: {e}")
#             raise

# # Global instance
# image_processor = ImageProcessor()

# """
# FAST Image Processing - No AI model download needed!
# Uses traditional computer vision (OpenCV only)
# """

# import cv2
# import numpy as np
# from PIL import Image
# import io

# class ImageProcessor:
#     """Ultra-fast processor - no AI model download"""
    
#     def __init__(self):
#         print("✅ Fast OpenCV processor initialized!")
    
#     def remove_background_simple(self, image_bytes):
#         """
#         Simple background removal using GrabCut
#         NO MODEL DOWNLOAD REQUIRED
#         """
#         try:
#             # Load image
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 raise ValueError("Invalid image")
            
#             # Resize for faster processing
#             height, width = img.shape[:2]
#             max_dim = 800
#             if max(height, width) > max_dim:
#                 scale = max_dim / max(height, width)
#                 new_width = int(width * scale)
#                 new_height = int(height * scale)
#                 img = cv2.resize(img, (new_width, new_height))
            
#             # Initialize mask
#             mask = np.zeros(img.shape[:2], np.uint8)
            
#             # Create rectangle for GrabCut (assume body is centered)
#             h, w = img.shape[:2]
#             rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
            
#             # Background and foreground models
#             bgd_model = np.zeros((1, 65), np.float64)
#             fgd_model = np.zeros((1, 65), np.float64)
            
#             # Apply GrabCut
#             cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
#             # Create binary mask
#             mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
#             # Apply mask
#             result = img * mask2[:, :, np.newaxis]
            
#             # Convert to grayscale mask
#             gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#             _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
#             return binary_mask
            
#         except Exception as e:
#             print(f"⚠️ GrabCut failed, using threshold: {e}")
#             # Fallback: simple thresholding
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
#             # Apply Otsu's thresholding
#             _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             return mask
    
#     def refine_mask(self, mask):
#         """Clean up mask"""
#         # Remove noise
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
#         # Fill holes
#         kernel = np.ones((7, 7), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
#         return mask
    
#     def resize_mask(self, mask, target_size=(512, 384)):
#         """Resize to target size"""
#         return cv2.resize(mask, (target_size[1], target_size[0]), 
#                          interpolation=cv2.INTER_LINEAR)
    
#     def process_image(self, image_bytes, target_size=(512, 384)):
#         """
#         Complete processing pipeline - FAST, NO DOWNLOAD
#         """
#         try:
#             # Remove background
#             mask = self.remove_background_simple(image_bytes)
            
#             # Refine
#             mask = self.refine_mask(mask)
            
#             # Resize
#             mask = self.resize_mask(mask, target_size)
            
#             # Convert to bytes
#             _, buffer = cv2.imencode('.png', mask)
#             return buffer.tobytes()
            
#         except Exception as e:
#             print(f"❌ Processing failed: {e}")
#             raise
    
#     def process_and_preview(self, image_bytes, target_size=(512, 384)):
#         """Generate preview"""
#         try:
#             # Process mask
#             mask = self.remove_background_simple(image_bytes)
#             mask = self.refine_mask(mask)
#             mask = self.resize_mask(mask, target_size)
            
#             # Load original for preview
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             original = cv2.resize(original, (target_size[1], target_size[0]))
            
#             # Create colored overlay (green)
#             mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#             mask_colored = np.zeros_like(original)
#             mask_colored[:, :, 1] = mask  # Green channel
            
#             # Blend
#             preview = cv2.addWeighted(original, 0.6, mask_colored, 0.4, 0)
            
#             # Convert to bytes
#             _, mask_buffer = cv2.imencode('.png', mask)
#             _, preview_buffer = cv2.imencode('.png', preview)
            
#             return {
#                 'mask_bytes': mask_buffer.tobytes(),
#                 'preview_bytes': preview_buffer.tobytes()
#             }
            
#         except Exception as e:
#             print(f"❌ Preview generation failed: {e}")
#             # Return empty mask on failure
#             empty_mask = np.zeros((target_size[0], target_size[1]), np.uint8)
#             _, buffer = cv2.imencode('.png', empty_mask)
#             return {
#                 'mask_bytes': buffer.tobytes(),
#                 'preview_bytes': buffer.tobytes()
#             }

# # Global instance
# image_processor = ImageProcessor()

# """
# IMPROVED Image Processing - Better mask quality
# Author: manamendraJN
# Date: 2025-11-18
# """

# import cv2
# import numpy as np
# from PIL import Image
# import io

# class ImageProcessor:
#     """Improved image processor with better segmentation"""
    
#     def __init__(self):
#         print("✅ Improved OpenCV processor initialized!")
    
#     def remove_background_improved(self, image_bytes):
#         """
#         Improved background removal with multiple techniques
#         """
#         try:
#             # Load image
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 raise ValueError("Invalid image")
            
#             # Resize for consistent processing
#             height, width = img.shape[:2]
#             max_dim = 800
#             if max(height, width) > max_dim:
#                 scale = max_dim / max(height, width)
#                 new_width = int(width * scale)
#                 new_height = int(height * scale)
#                 img = cv2.resize(img, (new_width, new_height))
            
#             h, w = img.shape[:2]
            
#             # METHOD 1: Try GrabCut with better initialization
#             try:
#                 mask = np.zeros(img.shape[:2], np.uint8)
#                 bgd_model = np.zeros((1, 65), np.float64)
#                 fgd_model = np.zeros((1, 65), np.float64)
                
#                 # Better rectangle - assumes person is centered
#                 margin = 0.05
#                 rect = (
#                     int(w * margin), 
#                     int(h * margin), 
#                     int(w * (1 - 2*margin)), 
#                     int(h * (1 - 2*margin))
#                 )
                
#                 # Run GrabCut algorithm
#                 cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
                
#                 # Create binary mask
#                 mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
#                 # Check if mask is reasonable
#                 foreground_ratio = np.sum(mask2) / (h * w)
                
#                 if 0.1 < foreground_ratio < 0.8:
#                     result_mask = mask2 * 255
#                 else:
#                     raise ValueError("GrabCut failed - unusual foreground ratio")
                    
#             except Exception as e:
#                 print(f"⚠️ GrabCut failed: {e}, using edge detection")
                
#                 # METHOD 2: Edge-based segmentation (fallback)
#                 # Convert to grayscale
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
#                 # Apply bilateral filter (preserves edges)
#                 blur = cv2.bilateralFilter(gray, 9, 75, 75)
                
#                 # Adaptive thresholding
#                 thresh = cv2.adaptiveThreshold(
#                     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                     cv2.THRESH_BINARY_INV, 11, 2
#                 )
                
#                 # Find contours
#                 contours, _ = cv2.findContours(
#                     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                 )
                
#                 # Create mask from largest contour
#                 result_mask = np.zeros_like(gray)
#                 if contours:
#                     # Find largest contour
#                     largest_contour = max(contours, key=cv2.contourArea)
#                     cv2.drawContours(result_mask, [largest_contour], -1, 255, -1)
#                 else:
#                     # Last resort: simple thresholding
#                     _, result_mask = cv2.threshold(
#                         gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#                     )
            
#             return result_mask
            
#         except Exception as e:
#             print(f"❌ All methods failed: {e}")
#             raise
    
#     def refine_mask_advanced(self, mask):
#         """
#         Advanced mask refinement with better edge preservation
#         """
#         # Remove small noise
#         kernel_small = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
#         # Fill small holes
#         kernel_medium = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
#         # Dilate slightly to recover edges
#         kernel_dilate = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
#         # Apply median blur to smooth edges
#         mask = cv2.medianBlur(mask, 5)
        
#         # Final thresholding to ensure binary mask
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
#         return mask
    
#     def resize_mask(self, mask, target_size=(512, 384)):
#         """Resize with good interpolation"""
#         return cv2.resize(
#             mask, (target_size[1], target_size[0]), 
#             interpolation=cv2.INTER_LINEAR
#         )
    
#     def process_image(self, image_bytes, target_size=(512, 384)):
#         """
#         Complete processing pipeline with improved quality
#         """
#         try:
#             # Step 1: Remove background
#             mask = self.remove_background_improved(image_bytes)
            
#             # Step 2: Refine mask
#             mask = self.refine_mask_advanced(mask)
            
#             # Step 3: Resize
#             mask = self.resize_mask(mask, target_size)
            
#             # Convert to bytes
#             _, buffer = cv2.imencode('.png', mask)
#             return buffer.tobytes()
            
#         except Exception as e:
#             print(f"❌ Processing failed: {e}")
#             raise
    
#     def process_and_preview(self, image_bytes, target_size=(512, 384)):
#         """
#         Generate preview with better visualization
#         """
#         try:
#             # Process mask
#             mask = self.remove_background_improved(image_bytes)
#             mask = self.refine_mask_advanced(mask)
#             mask_resized = self.resize_mask(mask, target_size)
            
#             # Load and resize original
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             original = cv2.resize(original, (target_size[1], target_size[0]))
            
#             # Create better preview with alpha blending
#             # Green overlay on detected body
#             mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
#             green_overlay = np.zeros_like(original)
#             green_overlay[:, :, 1] = mask_resized  # Green channel only
            
#             # Blend original with green overlay
#             alpha = 0.5
#             preview = cv2.addWeighted(original, 1-alpha, green_overlay, alpha, 0)
            
#             # Add contour for better visualization
#             contours, _ = cv2.findContours(
#                 mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#             )
#             cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
            
#             # Convert to bytes
#             _, mask_buffer = cv2.imencode('.png', mask_resized)
#             _, preview_buffer = cv2.imencode('.png', preview)
            
#             return {
#                 'mask_bytes': mask_buffer.tobytes(),
#                 'preview_bytes': preview_buffer.tobytes()
#             }
            
#         except Exception as e:
#             print(f"❌ Preview generation failed: {e}")
#             # Return empty on failure
#             empty_mask = np.zeros((target_size[0], target_size[1]), np.uint8)
#             _, buffer = cv2.imencode('.png', empty_mask)
#             return {
#                 'mask_bytes': buffer.tobytes(),
#                 'preview_bytes': buffer.tobytes()
#             }

# # Global instance
# image_processor = ImageProcessor()























# """
# Smart Image Processing - Detects masks vs real photos
# Author: manamendraJN
# Date: 2025-11-18
# """

# import cv2
# import numpy as np
# from PIL import Image
# import io
# import mediapipe as mp

# class ImageProcessor:
#     """Smart processor that detects if input is already a mask"""
    
#     def __init__(self):
#         try:
#             # Initialize MediaPipe for real photos
#             self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
#             self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
#                 model_selection=1
#             )
#             print("✅ MediaPipe segmentation initialized!")
#         except Exception as e:
#             print(f"❌ Failed to initialize MediaPipe: {e}")
#             self.segmenter = None
    
#     def is_already_mask(self, img):
#         """
#         Detect if image is already a binary mask
        
#         Returns:
#             True if image is already a mask (binary/grayscale)
#             False if image is a color photo needing segmentation
#         """
#         # Convert to grayscale
#         if len(img.shape) == 3:
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = img
        
#         # Check 1: Is it mostly black and white? (binary)
#         unique_values = np.unique(gray)
        
#         # If image has only 2-3 unique colors, it's likely a mask
#         if len(unique_values) <= 5:
#             print("🎭 Detected: Image is already a binary mask")
#             return True
        
#         # Check 2: Histogram analysis
#         # Masks typically have bimodal distribution (peaks at 0 and 255)
#         hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
#         # Normalize histogram
#         hist = hist.flatten() / hist.sum()
        
#         # Check if most pixels are near 0 (black) or 255 (white)
#         dark_pixels = hist[0:30].sum()
#         bright_pixels = hist[226:256].sum()
        
#         if (dark_pixels + bright_pixels) > 0.85:
#             print("🎭 Detected: Image is already a grayscale mask")
#             return True
        
#         print("📸 Detected: Image is a color photo (needs segmentation)")
#         return False
    
#     def clean_existing_mask(self, mask):
#         """
#         Clean and optimize an existing mask
#         Just remove tiny artifacts
#         """
#         # Minimal processing to preserve quality
        
#         # Remove very small noise
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
#         # Fill tiny holes
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
#         # Ensure binary
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
#         return mask
    
#     def segment_body(self, img):
#         """
#         Segment body from color photo using MediaPipe
#         """
#         try:
#             # Convert BGR to RGB
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#             # Process with MediaPipe
#             results = self.segmenter.process(img_rgb)
            
#             # Get binary mask
#             condition = results.segmentation_mask > 0.5
#             binary_mask = np.where(condition, 255, 0).astype(np.uint8)
            
#             # Refine
#             kernel = np.ones((5, 5), np.uint8)
#             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
#             return binary_mask
            
#         except Exception as e:
#             print(f"❌ MediaPipe segmentation failed: {e}")
#             raise
    
#     def resize_mask(self, mask, target_size=(512, 384)):
#         """Resize mask to target size"""
#         return cv2.resize(
#             mask, 
#             (target_size[1], target_size[0]),
#             interpolation=cv2.INTER_LINEAR
#         )
    
#     def process_image(self, image_bytes, target_size=(512, 384)):
#         """
#         Smart processing: Detect and handle masks vs photos
#         """
#         try:
#             # Load image
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 raise ValueError("Invalid image")
            
#             # SMART DETECTION
#             if self.is_already_mask(img):
#                 # Input is already a mask - just clean and resize
#                 print("✅ Using existing mask (minimal processing)")
                
#                 # Convert to grayscale if needed
#                 if len(img.shape) == 3:
#                     mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 else:
#                     mask = img
                
#                 # Clean lightly
#                 mask = self.clean_existing_mask(mask)
                
#             else:
#                 # Input is a photo - apply MediaPipe segmentation
#                 print("🔄 Applying AI segmentation to photo")
                
#                 if self.segmenter is None:
#                     raise RuntimeError("MediaPipe not initialized")
                
#                 mask = self.segment_body(img)
            
#             # Resize to target
#             mask = self.resize_mask(mask, target_size)
            
#             # Convert to bytes
#             _, buffer = cv2.imencode('.png', mask)
#             return buffer.tobytes()
            
#         except Exception as e:
#             print(f"❌ Image processing failed: {e}")
#             raise
    
#     def process_and_preview(self, image_bytes, target_size=(512, 384)):
#         """
#         Generate preview with smart detection
#         """
#         try:
#             # Load original
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if original is None:
#                 raise ValueError("Invalid image")
            
#             # Smart detection
#             if self.is_already_mask(original):
#                 # Already a mask
#                 if len(original.shape) == 3:
#                     mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#                 else:
#                     mask = original
                
#                 mask = self.clean_existing_mask(mask)
                
#                 # For preview, convert grayscale to color
#                 original_for_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
#             else:
#                 # Real photo - segment it
#                 if self.segmenter is None:
#                     raise RuntimeError("MediaPipe not initialized")
                
#                 mask = self.segment_body(original)
#                 original_for_preview = original
            
#             # Resize
#             original_resized = cv2.resize(
#                 original_for_preview,
#                 (target_size[1], target_size[0])
#             )
#             mask_resized = self.resize_mask(mask, target_size)
            
#             # Create preview with green overlay
#             mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
#             green_overlay = np.zeros_like(original_resized)
#             green_overlay[:, :, 1] = mask_resized
            
#             # Blend
#             preview = cv2.addWeighted(original_resized, 0.6, green_overlay, 0.4, 0)
            
#             # Add contour
#             contours, _ = cv2.findContours(
#                 mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#             )
#             cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
            
#             # Convert to bytes
#             _, mask_buffer = cv2.imencode('.png', mask_resized)
#             _, preview_buffer = cv2.imencode('.png', preview)
            
#             return {
#                 'mask_bytes': mask_buffer.tobytes(),
#                 'preview_bytes': preview_buffer.tobytes()
#             }
            
#         except Exception as e:
#             print(f"❌ Preview generation failed: {e}")
#             raise

# # Global instance
# image_processor = ImageProcessor()

















"""
Professional Image Processing using rembg AI
Best quality human body segmentation
Author: manamendraJN
Date: 2025-11-18
"""

import cv2
import numpy as np
from PIL import Image
import io
from rembg import remove, new_session

class ImageProcessor:
    """Professional-grade body segmentation using rembg AI"""
    
    def __init__(self):
        try:
            print("🔄 Loading rembg AI model (u2net_human_seg)...")
            print("   This model is specialized for human body segmentation")
            print("   First run will download ~60MB model file...")
            
            # Use u2net_human_seg - BEST model for human bodies
            self.session = new_session("u2net_human_seg")
            
            print("✅ rembg AI model loaded successfully!")
            print("   Model: u2net_human_seg (Human-specialized)")
            
        except Exception as e:
            print(f"⚠️ u2net_human_seg failed: {e}")
            print("   Falling back to u2net (general model)...")
            try:
                self.session = new_session("u2net")
                print("✅ rembg u2net model loaded!")
            except Exception as e2:
                print(f"❌ Failed to initialize rembg: {e2}")
                self.session = None
    
    def is_already_mask(self, img):
        """
        Smart detection: Is this already a binary mask?
        Returns True if image is already processed
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Check unique values
        unique_values = np.unique(gray)
        
        # Binary mask has very few unique values
        if len(unique_values) <= 5:
            print("🎭 Smart Detection: Image is already a mask (skipping AI)")
            return True
        
        # Histogram check
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        dark_pixels = hist[0:30].sum()
        bright_pixels = hist[226:256].sum()
        
        if (dark_pixels + bright_pixels) > 0.85:
            print("🎭 Smart Detection: Image is already a mask (skipping AI)")
            return True
        
        print("📸 Smart Detection: Color photo detected (applying AI segmentation)")
        return False
    
    def remove_background_ai(self, image_bytes):
        """
        Remove background using state-of-the-art AI
        Returns RGBA image with transparent background
        """
        try:
            # Load image
            input_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Optimize size for speed (optional)
            original_size = input_image.size
            max_size = 1024
            
            if max(original_size) > max_size:
                ratio = max_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                input_image = input_image.resize(new_size, Image.LANCZOS)
                print(f"   Resized from {original_size} to {new_size} for faster processing")
            
            print("🤖 AI is removing background...")
            
            # Remove background with best quality settings
            output_image = remove(
                input_image,
                session=self.session,
                only_mask=False,  # Return full RGBA image
                post_process_mask=True,  # Clean up mask edges
                alpha_matting=False,  # Faster, still good quality
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
            )
            
            print("✅ Background removed successfully!")
            
            return output_image
            
        except Exception as e:
            print(f"❌ AI background removal failed: {e}")
            raise
    
    def create_clean_mask(self, rgba_image):
        """
        Convert RGBA image to clean binary mask
        """
        # Convert to numpy array
        img_array = np.array(rgba_image)
        
        # Extract alpha channel (transparency = mask)
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
        else:
            # Fallback if no alpha
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Create binary mask (white = body, black = background)
        mask = np.where(alpha > 10, 255, 0).astype(np.uint8)
        
        return mask
    
    def refine_mask(self, mask):
        """
        Refine mask for perfect edges
        """
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill small holes
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Final threshold
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def resize_mask(self, mask, target_size=(512, 384)):
        """
        Resize mask to target dimensions
        target_size: (height, width)
        """
        return cv2.resize(
            mask,
            (target_size[1], target_size[0]),  # OpenCV uses (width, height)
            interpolation=cv2.INTER_LINEAR
        )
    
    def process_image(self, image_bytes, target_size=(512, 384)):
        """
        Complete AI processing pipeline
        
        Args:
            image_bytes: Raw image bytes
            target_size: Output size (height, width)
        
        Returns:
            Processed mask as PNG bytes
        """
        try:
            print("\n" + "="*60)
            print("🎯 Starting Image Processing Pipeline")
            print("="*60)
            
            # Load image for detection
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image - cannot decode")
            
            print(f"📐 Input image size: {img.shape[1]}x{img.shape[0]} pixels")
            
            # SMART DETECTION
            if self.is_already_mask(img):
                # Already a mask - minimal processing
                print("⚡ Fast path: Using existing mask")
                
                if len(img.shape) == 3:
                    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    mask = img
                
                # Just ensure binary
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
            else:
                # Color photo - apply full AI pipeline
                print("🚀 AI path: Processing color photo")
                
                if self.session is None:
                    raise RuntimeError("rembg AI model not loaded")
                
                # Step 1: Remove background with AI
                img_no_bg = self.remove_background_ai(image_bytes)
                
                # Step 2: Extract mask from alpha channel
                print("🎭 Extracting body mask...")
                mask = self.create_clean_mask(img_no_bg)
                
                # Step 3: Refine edges
                print("✨ Refining mask edges...")
                mask = self.refine_mask(mask)
            
            # Step 4: Resize to target
            print(f"📏 Resizing to {target_size[1]}x{target_size[0]} pixels...")
            mask = self.resize_mask(mask, target_size)
            
            # Convert to PNG bytes
            _, buffer = cv2.imencode('.png', mask)
            
            print("✅ Processing complete!")
            print("="*60 + "\n")
            
            return buffer.tobytes()
            
        except Exception as e:
            print(f"\n❌ PROCESSING FAILED: {e}")
            print("="*60 + "\n")
            import traceback
            traceback.print_exc()
            raise
    
    def process_and_preview(self, image_bytes, target_size=(512, 384)):
        """
        Generate preview with original, overlay, and final mask
        """
        try:
            print("\n🖼️  Generating preview...")
            
            # Load original
            nparr = np.frombuffer(image_bytes, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original is None:
                raise ValueError("Invalid image")
            
            # Process based on type
            if self.is_already_mask(original):
                # Already mask
                if len(original.shape) == 3:
                    mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                else:
                    mask = original
                
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                original_for_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
            else:
                # Apply AI
                if self.session is None:
                    raise RuntimeError("rembg not initialized")
                
                img_no_bg = self.remove_background_ai(image_bytes)
                mask = self.create_clean_mask(img_no_bg)
                mask = self.refine_mask(mask)
                original_for_preview = original
            
            # Resize everything
            original_resized = cv2.resize(original_for_preview, (target_size[1], target_size[0]))
            mask_resized = self.resize_mask(mask, target_size)
            
            # Create green overlay preview
            mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            green_overlay = np.zeros_like(original_resized)
            green_overlay[:, :, 1] = mask_resized  # Green channel
            
            # Blend
            preview = cv2.addWeighted(original_resized, 0.6, green_overlay, 0.4, 0)
            
            # Add contour outline
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
            
            # Convert to bytes
            _, mask_buffer = cv2.imencode('.png', mask_resized)
            _, preview_buffer = cv2.imencode('.png', preview)
            
            print("✅ Preview generated!\n")
            
            return {
                'mask_bytes': mask_buffer.tobytes(),
                'preview_bytes': preview_buffer.tobytes()
            }
            
        except Exception as e:
            print(f"❌ Preview generation failed: {e}\n")
            import traceback
            traceback.print_exc()
            raise

# Global instance
image_processor = ImageProcessor()