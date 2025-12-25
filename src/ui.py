import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(CURRENT_DIR, "main.py")

# -------------------------------
# Run inference command
# -------------------------------
def run_inference():
    if not img_path.get():
        messagebox.showerror("Error", "Please select an image first.")
        return

    output_text.delete("1.0", tk.END)

    try:
        # Ensure we run inside src/
        command = [
            sys.executable,  # uses venv python
            "main.py",
            "--mode", "infer",
            "--img_file", img_path.get()
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if stdout:
            output_text.insert(tk.END, "[OUTPUT]\n" + stdout)
        # if stderr:
        #     output_text.insert(tk.END, "\n[ERROR]\n" + stderr)

    except Exception as e:
        output_text.insert(tk.END, f"Exception: {str(e)}")

# -------------------------------
# Select image
# -------------------------------
def select_image():
    file = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if file:
        img_path.set(file)

# -------------------------------
# UI Setup
# -------------------------------
root = tk.Tk()
root.title("HTR")
root.geometry("700x500")

img_path = tk.StringVar()

tk.Label(root, text="Selected Image:").pack(anchor="w", padx=10)
tk.Entry(root, textvariable=img_path, width=80).pack(padx=10)

tk.Button(root, text="Browse Image", command=select_image).pack(pady=5)
tk.Button(root, text="Run Inference", command=run_inference).pack(pady=10)

output_text = tk.Text(root, height=20)
output_text.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()

"""
cd ..\SimpleHTR - Copy
.venv\Scripts\activate
cd src
python ui.py

"""
