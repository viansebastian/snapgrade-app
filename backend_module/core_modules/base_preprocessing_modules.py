import cv2
import numpy as np

print(cv2.__version__)
print(np.__version__)

def resize_image(image, target_size=(800, 1000)):
  '''Resize the image while maintaining aspect ratio'''
  h, w = image.shape[:2]
  scale = min(target_size[1] / w, target_size[0] / h)
  new_w = int(w * scale)
  new_h = int(h * scale)
  resized_image = cv2.resize(image, (new_w, new_h))
  return resized_image

def logarithmic_transformation(image, epsilon=1e-5):
  '''Apply logarithmic transformation to the image with zero value handling'''
  c = 255 / np.log(1 + np.max(image))
  # Epsilon zero-handling technique
  log_image = c * (np.log(1 + image + epsilon))
  log_image = np.array(log_image, dtype=np.uint8)

  return log_image

def contrast_stretching(image):
  min_val = np.min(image)
  max_val = np.max(image)
  stretched = (image - min_val) * (255 / (max_val - min_val))
  return stretched.astype(np.uint8)

def gaussian_blur(image, mode='Soft'):
  if mode == 'Soft':
    kernel_size = (3,3)
  elif mode == 'Medium':
    kernel_size = (5,5)
  elif mode == 'Hard':
    kernel_size = (7,7)
  else:
    raise ValueError("Mode must be 'Soft', 'Medium', or 'Hard'")

  return cv2.GaussianBlur(image, kernel_size, 0)

def measure_blurriness(image):
  # Apply the Laplacian operator to detect edges
  laplacian = cv2.Laplacian(image, cv2.CV_64F)
  # Variance of Laplacian
  variance = laplacian.var()

  return variance

def adaptive_gaussian_blur(image, desired_blur=100, max_iterations=100):
  # Measure initial blur level
  initial_blur = measure_blurriness(image)

  # Set a starting kernel size
  kernel_size = 5

  for iteration in range(max_iterations):
      # Apply Gaussian blur with the current kernel size
      blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

      # Measure the blur after applying Gaussian blur
      current_blur = measure_blurriness(blurred_image)

      # If the current blur exceeds the desired blur, stop
      if current_blur > desired_blur:
          kernel_size += 2
      else:
        break

  final_blurred_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
  final_blur = measure_blurriness(final_blurred_img)

  print(f"Initial Blur: {initial_blur}, Final Blur: {final_blur}, Kernel Size: {kernel_size}, Iterations: {iteration+1}")

  return final_blurred_img

def clahe_equalization(image):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  equalized_img = clahe.apply(image)
  return equalized_img

def otsu_thresholding(image):
  _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  return binary_image

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
  return cv2.Canny(image, low_threshold, high_threshold)

def find_extreme_corners(contours):
  '''Find the extreme corners of the image'''
  all_points = np.vstack(contours)
  top_left = all_points[np.argmin(all_points[:, :, 0] + all_points[:, :, 1])]
  bottom_right = all_points[np.argmax(all_points[:, :, 0] + all_points[:, :, 1])]
  top_right = all_points[np.argmax(all_points[:, :, 0] - all_points[:, :, 1])]
  bottom_left = all_points[np.argmin(all_points[:, :, 0] - all_points[:, :, 1])]
  return top_left[0], top_right[0], bottom_left[0], bottom_right[0]

def apply_perspective_transformation(image, corners):
  '''Apply perspective transformation to the image'''
  tl, tr, bl, br = corners
  width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
  height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

  dst_pts = np.array([
      [0, 0],
      [width - 1, 0],
      [0, height - 1],
      [width - 1, height - 1]
  ], dtype="float32")

  src_pts = np.array([tl, tr, bl, br], dtype="float32")

  M = cv2.getPerspectiveTransform(src_pts, dst_pts)
  warped = cv2.warpPerspective(image, M, (width, height))
  return warped

def automatic_warp_transformation(image, target_size=(800, 1000)):
  '''Automatic Cropping using Adaptive Warp Transformation'''
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = resize_image(gray_image, target_size)
  brightened_image = logarithmic_transformation(resized_image)
  contrast_image = contrast_stretching(brightened_image)
  blurred_image = gaussian_blur(contrast_image, mode='Soft')
  binary_image = otsu_thresholding(blurred_image)
  edges = canny_edge_detection(binary_image)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Getting Contours (Drawing Contours in image, useful for debugging)
  contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

  corners = find_extreme_corners(contours)
  for corner in corners:
      cv2.circle(contour_image, tuple(corner), 5, (0, 0, 255), -1)

  warped_image = apply_perspective_transformation(resized_image, corners)
  print(f'Initial image {image.shape} processed to {warped_image.shape}')

  return warped_image

def image_uniformization(master_image, student_image):
  '''Precision Image Resizing'''
  master_shape = master_image.shape
  student_shape = student_image.shape

  master_height = master_shape[0]
  master_width = master_shape[1]

  student_height = student_shape[0]
  student_width = student_shape[1]

  min_height = min(master_height, student_height)
  min_width = min(master_width, student_width)

  resized_master = cv2.resize(master_image, (min_width, min_height))
  resized_student = cv2.resize(student_image, (min_width, min_height))

  print(f'master_key {master_image.shape} and student_answer {student_image.shape} uniformed to {resized_master.shape}')

  return resized_master, resized_student