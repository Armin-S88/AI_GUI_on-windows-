import customtkinter as ctk
import threading
import tkinter as tk
import os
from tkinter import filedialog
from mediapipe.tasks import python
from mediapipe.tasks.python import gen_ai
import time

# تنظیمات ظاهری
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class GemmaMultimodalApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Google AI Edge - Gemma Multimodal Interface")
        self.geometry("1000x700")
        
        # متغیرهای مدل و چت
        self.llm_inference = None
        self.model_path = ""
        self.attached_file = None # برای ذخیره مسیر فایل چسبانده شده

        # ساختار گرید اصلی
        self.grid_columnconfigure(1, weight=1) # ستون چت پهن‌تر باشد
        self.grid_rowconfigure(0, weight=1)

        # --- 1. پنل سمت چپ (تنظیمات و فایل‌ها) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Edge Studio", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Load LLM (.task)", command=self.load_model_dialog)
        self.btn_load.pack(pady=10, padx=10)
        
        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="No Model", text_color="orange", wraplength=180)
        self.lbl_status.pack(pady=5)

        ctk.CTkLabel(self.sidebar_frame, text="Multimodal Input:", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 5))
        
        # دکمه‌های پیوست فایل
        self.btn_attach_img = ctk.CTkButton(self.sidebar_frame, text="Attach Image", fg_color="#2c3e50", command=lambda: self.attach_file("image"))
        self.btn_attach_img.pack(pady=5, padx=10)
        
        self.btn_attach_audio = ctk.CTkButton(self.sidebar_frame, text="Attach Audio", fg_color="#2c3e50", command=lambda: self.attach_file("audio"))
        self.btn_attach_audio.pack(pady=5, padx=10)

        self.lbl_file_attached = ctk.CTkLabel(self.sidebar_frame, text="No file attached", text_color="gray", wraplength=180)
        self.lbl_file_attached.pack(pady=10)

        # --- 2. بخش اصلی چت (سمت راست) ---
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # نمایشگر چت
        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled", wrap="word", font=ctk.CTkFont(size=13))
        self.chat_display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # تگ‌های رنگی برای کاربر و هوش مصنوعی
        self.chat_display.tag_config("user_tag", foreground="#3498db")
        self.chat_display.tag_config("ai_tag", foreground="#2ecc71")
        self.chat_display.tag_config("file_tag", foreground="#e67e22", font=ctk.CTkFont(slant="italic"))

        # --- 3. بخش ورودی (پایین) ---
        self.input_area = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_area.grid(row=1, column=0, sticky="ew")
        self.input_area.grid_columnconfigure(0, weight=1)

        self.entry_msg = ctk.CTkEntry(self.input_area, placeholder_text="Ask Gemma anything... (use attached files placeholder)")
        self.entry_msg.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry_msg.bind("<Return>", self.start_generation)

        self.btn_send = ctk.CTkButton(self.input_area, text="Send", width=100, command=self.start_generation)
        self.btn_send.grid(row=0, column=1)

    # --- توابع مدل ---

    def load_model_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("MediaPipe Task", "*.task")])
        if file_path:
            self.model_path = file_path
            self.lbl_status.configure(text="Loading...", text_color="yellow")
            # اجرا در ترد جداگانه
            threading.Thread(target=self.init_mediapipe, daemon=True).start()

    def init_mediapipe(self):
        try:
            # تنظیمات پایه مدل
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            
            # تعریف کول‌بک برای Streaming
            def progress_callback(output_text, done):
                # این تابع وقتی هر تکه متن تولید شد فراخوانی می‌شود
                self.update_ai_response_stream(output_text, done)

            # تنظیمات با قابلیت Streaming
            options = gen_ai.LlmInferenceOptions(
                base_options=base_options,
                result_callback=progress_callback # فعال‌سازی استریم
            )
            
            # ایجاد نمونه اینفرنس (بصورت Async برای استریم)
            self.llm_inference = gen_ai.LlmInference.create_from_options(options)
            
            model_name = os.path.basename(self.model_path)
            self.lbl_status.configure(text=f"Active: {model_name}", text_color="green")
        except Exception as e:
            self.lbl_status.configure(text=f"Error: {str(e)}", text_color="red")

    # --- توابع مالتی‌مدیا ---

    def attach_file(self, file_type):
        file_types_map = {
            "image": [("Images", "*.png;*.jpg;*.jpeg")],
            "audio": [("Audio", "*.mp3;*.wav;*.ogg")]
        }
        file_path = filedialog.askopenfilename(filetypes=file_types_map[file_type])
        if file_path:
            self.attached_file = {"path": file_path, "type": file_type}
            file_name = os.path.basename(file_path)
            self.lbl_file_attached.configure(text=f"Attached {file_type}: {file_name}", text_color="#e67e22")
            # به کاربر پیشنهاد می‌دهیم که در پرامپت به فایل اشاره کند
            current_text = self.entry_msg.get()
            if "[file]" not in current_text:
                 self.entry_msg.insert(0, f"Based on this {file_type} [file], ")

    # --- توابع چت و استریمینگ ---

    def append_text(self, text, tag=None, markdown=False):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text, tag)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def start_generation(self, event=None):
        if not self.llm_inference:
            tk.messagebox.showwarning("Model Not Loaded", "Please load a Gemma .task model first.")
            return
        
        user_input = self.entry_msg.get()
        if not user_input and not self.attached_file:
            return

        # نمایش پیام کاربر
        self.append_text("\n👤 You: ", "user_tag")
        
        final_prompt = user_input
        
        # اگر فایلی پیوست شده باشد، در رابط کاربری نشان می‌دهیم (پردازش واقعی نیاز به مدل مالتی‌مدیا دارد)
        if self.attached_file:
            file_name = os.path.basename(self.attached_file['path'])
            self.append_text(f"[Attached {self.attached_file['type']}: {file_name}] ", "file_tag")
            # در نسخه فعلی، ما فقط متنی عمل می‌کنیم. پرامپت را کمی تغییر می‌دهیم
            if "[file]" in user_input:
                 final_prompt = user_input.replace("[file]", f"(user attached a {self.attached_file['type']} named {file_name})")
        
        self.append_text(user_input + "\n")
        self.entry_msg.delete(0, "end")
        
        # آماده‌سازی برای دریافت پاسخ AI
        self.append_text("🤖 Gemma: ", "ai_tag")
        self.current_ai_response_start_index = self.chat_display.index("end-1c")
        
        # غیرفعال کردن دکمه ارسال در حین تولید
        self.btn_send.configure(state="disabled")
        
        # **اجرای Async** (این تابع فوراً برمی‌گردد، نتیجه از طریق Callback می‌آید)
        try:
            self.llm_inference.generate_async(final_prompt)
        except Exception as e:
            self.append_text(f"\nError: {str(e)}")
            self.btn_send.configure(state="normal")

    def update_ai_response_stream(self, output_text, done):
        # این تابع توسط ترد مدیاپایپ فراخوانی می‌شود، باید تغییرات UI را به ترد اصلی بفرستیم
        self.after(0, lambda: self._safe_update_ui(output_text, done))

    def _safe_update_ui(self, output_text, done):
        # افزودن تکه متن جدید به انتهای ادیتور
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", output_text)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
        if done:
            # پایان تولید
            self.append_text("\n")
            self.btn_send.configure(state="normal")
            # پاک کردن فایل پیوست شده پس از ارسال
            self.attached_file = None
            self.lbl_file_attached.configure(text="No file attached", text_color="gray")

if __name__ == "__main__":
    app = GemmaMultimodalApp()
    # اضافه کردن آیکون در صورت وجود (اختیاری)
    # if os.path.exists("icon.ico"): app.iconbitmap("icon.ico")
    app.mainloop()
