import cv2
import gradio as gr
import insightface
from insightface.app import FaceAnalysis
import os
from huggingface_hub import hf_hub_download
import numpy as np

# Mac (CoreML) နှင့် CPU အတွက် Provider Setting
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

# Model ဖိုင် ရှိမရှိ စစ်ဆေးခြင်း နှင့် Download ပြုလုပ်ခြင်း
model_file = 'inswapper_128.onnx'

if not os.path.exists(model_file):
    print(f"📥 {model_file} မရှိသေးပါ... Hugging Face မှ Download လုပ်နေပါသည်...")
    try:
        hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print("✅ Download အောင်မြင်ပါသည်။")
    except Exception as e:
        print(f"❌ Download မရပါ: {e}")

# Insightface စနစ် စတင်ခြင်း
app = FaceAnalysis(name='buffalo_l', providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(model_file, providers=providers)

def process_image(source_img, target_img):
    if source_img is None or target_img is None: return None
    
    # Source မျက်နှာကို ရှာဖွေခြင်း
    source_faces = app.get(source_img)
    if len(source_faces) == 0: 
        print("Source တွင် မျက်နှာမတွေ့ပါ")
        return target_img
    
    # မျက်နှာများစွာတွေ့ပါက အကြီးဆုံး သို့မဟုတ် ပထမဆုံးကို ယူမည်
    source_face = sorted(source_faces, key=lambda x: x.bbox[0])[0]
    
    # Target မျက်နှာကို ရှာဖွေခြင်း
    target_faces = app.get(target_img)
    if len(target_faces) == 0:
        print("Target တွင် မျက်နှာမတွေ့ပါ")
        return target_img

    res_img = target_img.copy()
    
    # Target ပုံရှိ မျက်နှာအားလုံးကို လိုက်ပြောင်းမည်
    for face in target_faces:
        res_img = swapper.get(res_img, face, source_face, paste_back=True)
    
    return res_img

def process_video(source_img, video_path):
    if source_img is None or video_path is None: return None
    
    source_faces = app.get(source_img)
    if len(source_faces) == 0: return video_path
    source_face = sorted(source_faces, key=lambda x: x.bbox[0])[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🎥 Video Processing စတင်နေပါပြီ...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        try:
            faces = app.get(frame)
            for face in faces:
                frame = swapper.get(frame, face, source_face, paste_back=True)
            out.write(frame)
        except Exception as e:
            print(f"Frame Error: {e}")
            out.write(frame)

    cap.release()
    out.release()
    print("✅ Video ပြီးစီးပါပြီ!")
    return output_path

# UI တည်ဆောက်ခြင်း
with gr.Blocks(title="Mini Face Swap (Mac)") as demo:
    gr.Markdown("# 🚀 Mini Face Swap (Auto-Launch & Public Link)")
    gr.Markdown("Note: ပထမဆုံးအကြိမ် Run လျှင် Model ဒေါင်းလုပ်ဆွဲချိန် ခေတ္တစောင့်ပေးပါ။")
    
    with gr.Tabs():
        with gr.TabItem("🖼️ Photo Swap"):
            with gr.Row():
                with gr.Column():
                    p_src = gr.Image(label="Source Face (မျက်နှာမူရင်း)", type="numpy")
                    p_tgt = gr.Image(label="Target Image (ပြောင်းမည့်ပုံ)", type="numpy")
                    btn_photo = gr.Button("Swap Photo", variant="primary")
                with gr.Column():
                    p_out = gr.Image(label="Result (ရလဒ်)")
            
            btn_photo.click(process_image, inputs=[p_src, p_tgt], outputs=p_out)
        
        with gr.TabItem("🎥 Video Swap"):
            with gr.Row():
                with gr.Column():
                    v_src = gr.Image(label="Source Face", type="numpy")
                    v_tgt = gr.Video(label="Target Video")
                    btn_video = gr.Button("Swap Video", variant="primary")
                with gr.Column():
                    v_out = gr.Video(label="Result Video")
            
            btn_video.click(process_video, inputs=[v_src, v_tgt], outputs=v_out)

if __name__ == "__main__":
    # ဒီနေရာမှာ ပြင်ထားပါတယ်
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        inbrowser=True,  # Browser အလိုလိုပွင့်ရန်
        share=True       # Public Link ထုတ်ပေးရန်
    )
