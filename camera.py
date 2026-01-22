import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


IMAGES_DIR = "images"


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Snapshot")

        os.makedirs(IMAGES_DIR, exist_ok=True)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.frame = None

        # ===== UI =====
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Name:").pack(side="left")
        self.name_entry = ttk.Entry(control_frame, width=30)
        self.name_entry.pack(side="left", padx=5)

        self.snapshot_button = ttk.Button(
            control_frame,
            text="Take Snapshot",
            command=self.take_snapshot
        )
        self.snapshot_button.pack(side="left", padx=10)

        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def take_snapshot(self):
        if self.frame is None:
            return

        name = self.name_entry.get().strip()
        if not name:
            return

        index = 1
        while True:
            filename = f"{name}-{index:03d}.png"
            path = os.path.join(IMAGES_DIR, filename)
            if not os.path.exists(path):
                break
            index += 1

        cv2.imwrite(path, self.frame)
        print(f"Saved: {path}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
