# Video Processing Workflow: Pavement Defect Detection

This document describes the step-by-step workflow that occurs after a video is uploaded for pavement defect detection. It covers all intermediate steps, processing logic, and data flow, providing visibility into the backend pipeline.

---

## 1. **Video Upload (Client → Backend)**
- **Endpoint:** `/detect-video` (POST)
- **Input:** Video file (form-data), user info, coordinates, selected model type (e.g., All, Potholes, Cracks, Kerbs)
- **Action:**
  - The backend receives the video file and saves it to a temporary location on the server with a timestamp-based filename.
  - User and role information are extracted from the request/session.

---

## 2. **Original Video Upload to S3**
- **Action:**
  - The backend uploads the original video to AWS S3 for storage and future reference.
  - The S3 path is constructed using the user's role and username.
  - The S3 URL (relative path) is stored in the database.

---

## 3. **Database Record Creation**
- **Collection:** `video_processing`
- **Action:**
  - A new document is created in the `video_processing` collection with metadata:
    - `video_id` (unique)
    - `original_video_url` (S3 path)
    - `processed_video_url` (to be filled after processing)
    - `role`, `username`, `timestamp`, `models_run`, `status` (set to 'processing'), etc.

---

## 4. **Video Processing Pipeline**
- **Function:** `process_pavement_video()`
- **Action:**
  - The video is opened using OpenCV.
  - Video properties (frame rate, frame count, resolution) are read.
  - An output video writer is initialized for the processed video.
  - A defect tracker is initialized for tracking defects across frames.

### For Each Frame:
  1. **Frame Extraction:**
     - The next frame is read from the video.
  2. **Model Inference:**
     - The selected AI model(s) (YOLO, MiDaS) are run on the frame to detect defects (potholes, cracks, kerbs).
     - Depth estimation is performed if required (for potholes).
  3. **Defect Tracking:**
     - Detected defects are tracked across frames to avoid duplicates.
  4. **Visualization:**
     - Bounding boxes, segmentation masks, and labels are drawn on the frame.
  5. **Frame Encoding:**
     - The processed frame is encoded as JPEG and base64 for streaming.
  6. **Streaming to Client:**
     - The processed frame, detection results, and progress are streamed to the client via Server-Sent Events (SSE).
  7. **Performance Monitoring:**
     - Frame processing time, FPS, and (if using CUDA) GPU memory usage are tracked and reported.

---

## 5. **Processed Video Output**
- **Action:**
  - The processed frames are saved as a new video file on the server.
  - After processing, the processed video is uploaded to S3.
  - The S3 path for the processed video is stored in the database.
  - The local processed video file is deleted to save space.

---

## 6. **Detection Results Storage**
- **Action:**
  - All unique detected defects (with measurements, frame info, etc.) are collected.
  - The `video_processing` document is updated with model outputs (potholes, cracks, kerbs) and status is set to 'completed'.

---

## 7. **Client Receives Results**
- **Action:**
  - The client receives streamed frames and detection data in real time.
  - At the end, a summary is sent with:
    - Total frames processed
    - All unique detections
    - S3 path to the processed video
    - Performance summary (total time, FPS, etc.)

---

## 8. **Status & Retrieval APIs**
- **Endpoints:**
  - `/video-processing/<video_id>`: Get status/results for a specific video
  - `/video-processing/list`: List all video processing records (with filters)

---

## 9. **Cleanup**
- **Action:**
  - Temporary video files are deleted after processing and upload.
  - Old processed videos are periodically cleaned up from the server.

---

## **Summary Diagram**

```mermaid
flowchart TD
    A[Client Uploads Video] --> B[Backend Saves Temp File]
    B --> C[Upload Original to S3]
    C --> D[Create DB Record]
    D --> E[Process Video Frame by Frame]
    E --> F[Run AI Models (YOLO, MiDaS)]
    F --> G[Track & Visualize Defects]
    G --> H[Stream Processed Frames to Client]
    G --> I[Write Processed Video]
    I --> J[Upload Processed Video to S3]
    J --> K[Update DB Record with Results]
    K --> L[Client Receives Final Results]
    L --> M[Cleanup Temp Files]
```

---

## **Key Intermediate Data/Artifacts**
- Temporary video file (server, deleted after processing)
- Original video S3 path
- Processed video S3 path
- Video processing DB record (status, outputs, metadata)
- Streamed processed frames (base64 JPEG)
- Detection results (per frame, per defect)

---

## **Error Handling**
- If any step fails, the DB record is updated with status 'failed' and error message.
- The client is notified of errors via SSE or API response.

---

## **References**
- Main implementation: `backend/routes/pavement.py` (`detect_video`, `process_pavement_video`)
- Database: MongoDB collections (`video_processing`, etc.)
- S3: AWS S3 bucket/folder structure 