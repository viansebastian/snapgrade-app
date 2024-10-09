import cv2 
import numpy as np

print(np.__version__)

from core_modules.base_preprocessing_modules import (
    gaussian_blur, 
    adaptive_gaussian_blur, 
    clahe_equalization,
    contrast_stretching, 
    logarithmic_transformation, 
    otsu_thresholding
)

def morph_open(image):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  eroded_img = cv2.erode(image, kernel, iterations = 1)
  dilated_img = cv2.dilate(eroded_img, kernel, iterations = 1)

  return dilated_img

def core_preprocessing(image):
  '''Core Preprocessing Module'''
  blurred_img = gaussian_blur(image, mode='Hard')
  contrast_img = contrast_stretching(blurred_img)
  log_img = logarithmic_transformation(contrast_img)
  binary_img = otsu_thresholding(log_img)
  opened_img = morph_open(binary_img)

  return opened_img

def core_preprocessing_v2(image):
  '''
  Core Preprocessing Module V2:
  - Uses CLAHE for lighting handling
  - Uses Adaptive Gaussian Blur to ensure optimal thresholding
  '''
  clahe_img = clahe_equalization(image)
  blurred_img = adaptive_gaussian_blur(clahe_img, desired_blur=100, max_iterations=100)
  contrast_img = contrast_stretching(blurred_img)
  log_img = logarithmic_transformation(contrast_img)
  binary_img = otsu_thresholding(log_img)
  opened_img = morph_open(binary_img)

  return opened_img

def draw_full_contours(contours, cont_image, radius = 7):
  '''Draw Full Circles'''
  for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      # Draw a filled circle at the center of the contour
      cv2.circle(cont_image, (cX, cY), radius, (0, 255, 0), -1)

  return cont_image

def extract_and_draw_contours(image):
  contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  unique_values = []
  for columns in image:
    for pixel in columns:
      if pixel not in unique_values:
        unique_values.append(pixel)

  contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  contour_image = draw_full_contours(contours, contour_image)

  return contours, contour_image

def extract_and_draw_circle_contours(image):
  contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  circle_contours = []
  contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  for contour in contours:
      # Approximate the enclosing circle for each contour
      (x, y), radius = cv2.minEnclosingCircle(contour)
      circle_area = np.pi * (radius ** 2)

      # Calculate the actual contour area
      contour_area = cv2.contourArea(contour)
      
      # Tolerance range for being "circular"
      if radius < 5:
          if 0.6 <= contour_area / circle_area <= 1.4:
              circle_contours.append(contour)
      else:
          if 0.8 <= contour_area / circle_area <= 1.2:
              circle_contours.append(contour)
              
  contour_image = draw_full_contours(circle_contours, contour_image)

  return circle_contours, contour_image

def soft_morph_open(image):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  eroded_img = cv2.erode(image, kernel, iterations = 1)
  dilated_img = cv2.dilate(eroded_img, kernel, iterations = 1)

  return dilated_img

def final_scoring(new_student, processed_student, master_contours):
  '''Final Score Calculation'''
  test_answer = processed_student.copy()
  # drawing the Answer Key to the Student's test answer, extracting the mistakes information
  check_answers = draw_full_contours(master_contours, test_answer)

  # open the image to remove noise
  final_sheet = soft_morph_open(check_answers)

  # fetching mistakes contours
  final_contours, img = extract_and_draw_circle_contours(final_sheet)
  
  # calculating mistakes and final score
  mistakes = len(final_contours)
  total_questions = len(master_contours)
  print(f'total_questions: {total_questions}, mistakes: {mistakes}')
  final_score = ((total_questions - mistakes) / total_questions) * 100
  print(f'final score: {final_score}')

  student_correction = cv2.cvtColor(new_student, cv2.COLOR_GRAY2BGR)
  student_correction = draw_full_contours(master_contours, student_correction)

  return final_score, student_correction, total_questions, mistakes