import cv2
import threading
import queue
import time


def getCurTime():
    """Get current time in seconds since epoch"""
    return time.time()


class ImageReader(object):
    def __init__(self, image_list, frame_rate, buffer_size=100) -> None:
        self.frame_rate = frame_rate
        self.buffer_size = buffer_size
        self.loaded_images = queue.Queue(maxsize=buffer_size)  # Use Queue instead of deque
        self._close_event = threading.Event()
        self.images_path = self.read_image_list(image_list)
        self.total_images = len(self.images_path)
        self._image_idx = 0

        # Start the reader thread
        self._reader_thread = threading.Thread(target=self._run, daemon=True)
        self._reader_thread.start()

    def close(self):
        """Stops the reader thread gracefully"""
        self._close_event.set()
        self._reader_thread.join()

    @staticmethod
    def read_image_list(image_list):
        """Reads the list of image paths from a text file"""
        with open(image_list) as file:
            images_path = file.readlines()
        return [path.strip() for path in images_path]

    def _run(self):
        """Background thread to load images into the queue"""
        while not self._close_event.is_set():
            if self._image_idx >= self.total_images:
                break  # Stop if we've loaded all images

            if not self.loaded_images.full():
                img = cv2.imread(self.images_path[self._image_idx])
                if img is not None:
                    self.loaded_images.put(img)  # Add frame to the queue
                    self._image_idx += 1

    def next(self):
        """Return the next image from the queue (non-blocking)"""
        try:
            return self.loaded_images.get(timeout=1)  # Wait for up to 1 second for a frame
        except queue.Empty:
            return None


class VideoReader(object):
    def __init__(self, video_file, frame_rate, buffer_size=100) -> None:
        self.frame_rate = frame_rate
        self.buffer_size = buffer_size
        self._cap = cv2.VideoCapture(video_file)
        self.loaded_frames = queue.Queue(maxsize=buffer_size)  # Use Queue instead of deque
        self._close_event = threading.Event()

        # Start the reader thread
        self._reader_thread = threading.Thread(target=self._run, daemon=True)
        self._reader_thread.start()

    def close(self):
        """Stops the reader thread gracefully"""
        self._close_event.set()
        self._reader_thread.join()
        if self._cap is not None:
            self._cap.release()

    def _run(self):
        """Background thread to load video frames into the queue"""
        while not self._close_event.is_set():
            if not self.loaded_frames.full():
                ret, frame = self._cap.read()
                if not ret:  # Video ended or error occurred
                    break
                if frame is not None:
                    self.loaded_frames.put(frame)  # Add frame to the queue

    def next(self):
        """Return the next video frame from the queue (non-blocking)"""
        try:
            return self.loaded_frames.get(timeout=1)  # Wait for up to 1 second for a frame
        except queue.Empty:
            return None


class FrameReader(object):
    def __init__(self, video_file, image_list, frame_rate, controller_client=None, buffer_size=100) -> None:
        self.frame_rate = frame_rate
        self.frame_interval = (1.0 / frame_rate) * 1e3  # Frame interval in milliseconds
        self._last_read_time = getCurTime() - (self.frame_interval * 1e-3)  # Track last frame read time
        self.controller_client = controller_client  # Reference to ControllerClient
        self.frame_id = 0

        # Use VideoReader or ImageReader based on file type
        if video_file != "":
            self._reader = VideoReader(video_file, frame_rate, buffer_size=buffer_size)
        elif image_list != "":
            self._reader = ImageReader(image_list, frame_rate, buffer_size=buffer_size)
        else:
            raise ValueError("Either a video file or an image list must be specified")

    def update_frame_interval(self, frame_rate):
        """Update frame interval whenever the frame rate changes"""
        self.frame_rate = frame_rate
        self.frame_interval = (1.0 / self.frame_rate) * 1e3  # Update interval in milliseconds
        print(f"Updated frame rate to {self.frame_rate} FPS (frame interval = {self.frame_interval:.2f} ms)")

    def close(self):
        """Close the underlying reader"""
        self._reader.close()

    def next(self):
        """Return the next frame from the reader, respecting the frame rate"""
        img = self._reader.next()
        if img is None:
            return None

        # Dynamically adjust the frame rate if the controller suggests a change
        if self.controller_client:
            current_frame_rate = self.controller_client.get_frame_rate()
            if current_frame_rate != self.frame_rate:
                self.update_frame_interval(current_frame_rate)

        # Maintain frame rate using frame intervals
        time_diff_ms = (getCurTime() - self._last_read_time) * 1e3  # Time since last frame
        if time_diff_ms < self.frame_interval:
            sleep_time_sec = (self.frame_interval - time_diff_ms) * 1e-3
            time.sleep(sleep_time_sec)  # Only sleep if frame load was fast

        frame = {
            "id": self.frame_id,
            "data": img
        }
        self.frame_id += 1
        self._last_read_time = getCurTime()
        return frame
