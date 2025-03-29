import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class EnhancedJSONValidator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("地理数据校验工具")
        self.geometry("1200x800")

        # 初始化变量
        self.data = []
        self.current_index = 0
        self.image_dir = ""
        self.json_path = ""
        self.modified = False

        # 创建界面组件
        self.create_widgets()
        self.setup_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def create_widgets(self):
        # 菜单栏
        self.menubar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="打开JSON", command=self.load_json)
        self.file_menu.add_command(label="设置图片目录", command=self.set_image_dir)
        self.file_menu.add_command(label="另存为", command=self.save_as_json)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.on_exit)
        self.menubar.add_cascade(label="文件", menu=self.file_menu)
        self.config(menu=self.menubar)

        # 图像显示区域
        self.image_frame = ttk.LabelFrame(self, text="卫星影像预览")
        self.img_label = ttk.Label(self.image_frame)
        self.img_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 元数据区域
        self.meta_frame = ttk.LabelFrame(self, text="地理信息元数据")
        self.meta_text = tk.Text(self.meta_frame, height=12, width=60, font=('等线', 10))
        self.meta_scroll = ttk.Scrollbar(self.meta_frame, command=self.meta_text.yview)
        self.meta_text.configure(yscrollcommand=self.meta_scroll.set)

        # 编辑区域
        self.edit_frame = ttk.LabelFrame(self, text="标注内容编辑")
        self.cap_entry = tk.Text(self.edit_frame, height=5, width=80, font=('等线', 10))
        self.caption_entry = tk.Text(self.edit_frame, height=15, width=80, font=('等线', 10))

        # 控制面板
        self.control_frame = ttk.Frame(self)
        self.prev_btn = ttk.Button(self.control_frame, text="上一条 (←)", command=self.prev_item)
        self.next_btn = ttk.Button(self.control_frame, text="通过并下一张 (→)", command=self.next_item)
        self.save_btn = ttk.Button(self.control_frame, text="保存 (Ctrl+S)", command=self.save_json)

        # 状态栏
        self.status_bar = ttk.Label(self, relief=tk.SUNKEN, anchor=tk.W)

        # 绑定快捷键
        self.bind("<Control-s>", lambda e: self.save_json())
        self.bind("<Left>", lambda e: self.prev_item())
        self.bind("<Right>", lambda e: self.next_item())

    def setup_layout(self):
        # 主界面布局
        self.image_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew", rowspan=2)
        self.meta_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.edit_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.control_frame.grid(row=2, column=0, columnspan=2, pady=10)
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

        # 元数据区域布局
        self.meta_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.meta_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 编辑区域布局
        ttk.Label(self.edit_frame, text="简要描述 (cap):").pack(anchor=tk.W)
        self.cap_entry.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(self.edit_frame, text="详细描述 (caption):").pack(anchor=tk.W)
        self.caption_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # 按钮布局
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 权重分配
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(1, weight=1)

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)['data']
                self.json_path = file_path
                self.current_index = 0
                self.show_current_item()
                self.update_status()
                self.modified = False
                messagebox.showinfo("加载成功", f"已加载 {len(self.data)} 条数据")
            except Exception as e:
                messagebox.showerror("错误", f"加载JSON文件失败：{str(e)}")

    def set_image_dir(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            if not self.data:
                messagebox.showwarning("警告", "请先加载JSON文件")
                return
            self.show_current_item()
            self.update_status()

    def show_current_item(self):
        if not self.data or not self.image_dir:
            return

        item = self.data[self.current_index]

        # 加载图片
        img_path = os.path.join(self.image_dir, f"{item['name']}.png")  # 修改扩展名为实际格式
        try:
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{str(e)}\n路径：{img_path}")

        # 显示元数据
        self.meta_text.delete(1.0, tk.END)
        meta_info = f"分辨率: {item['info']['resolution']}\n"
        meta_info = f"卫星: {item['info']['satellite']}\n"
        meta_info = f"传感器: {item['info']['sensor']}\n"
        meta_info = f"获取日期: {item['info']['acquisition_date']}\n"
        meta_info += f"经纬度: {item['info']['coordinates']}\n"
        # meta_info += "地理位置:\n - " + "\n - ".join(item['info']['location'])
        meta_info += "\n\n属性特征:\n"
        for attr in item['attrs']:
            meta_info += " - " + ", ".join(f"{k}: {v}" for k, v in attr.items()) + "\n"
        self.meta_text.insert(tk.END, meta_info)

        # 加载文本内容
        self.cap_entry.delete(1.0, tk.END)
        self.cap_entry.insert(tk.END, item['cap'])
        self.caption_entry.delete(1.0, tk.END)
        self.caption_entry.insert(tk.END, item['caption'])

    def prev_item(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_item()
            self.update_status()

    def next_item(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.show_current_item()
            self.update_status()

    def update_status(self):
        status = f"当前项目：{self.current_index + 1}/{len(self.data)} | "
        status += f"图片目录：{self.image_dir}" if self.image_dir else "请先设置图片目录"
        self.status_bar.config(text=status)

    def save_as_json(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            self.json_path = file_path
            self.save_json()

    def zoom_image(self, event):
        item = self.data[self.current_index]
        img_path = os.path.join(self.image_dir, f"{item['name']}.jpg")
        try:
            zoom_window = tk.Toplevel(self)
            img = Image.open(img_path)
            photo = ImageTk.PhotoImage(img)
            ttk.Label(zoom_window, image=photo).pack()
            zoom_window.title(f"原始尺寸查看 - {item['name']}")
            zoom_window.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{str(e)}")

    def save_json(self):
        if not self.data:
            return

        self.data[self.current_index]['cap'] = self.cap_entry.get("1.0", tk.END).strip()
        self.data[self.current_index]['caption'] = self.caption_entry.get("1.0", tk.END).strip()

        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump({'data': self.data}, f, indent=2, ensure_ascii=False)
            self.modified = False
            self.status_bar.config(text=f"保存成功：{self.json_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存文件：{str(e)}")

    def on_exit(self):
        if self.modified:
            if messagebox.askyesno("保存修改", "检测到未保存的修改，是否保存后再退出？"):
                self.save_json()
        self.destroy()


if __name__ == "__main__":
    app = EnhancedJSONValidator()
    app.mainloop()