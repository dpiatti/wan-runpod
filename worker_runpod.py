import os, json, requests, random, time, cv2, ffmpeg, cloudinary, cloudinary.uploader, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_hunyuan, nodes_flux, nodes_model_advanced, nodes_custom_sampler, nodes_images

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CLIPVisionEncode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
TextEncodeHunyuanVideo_ImageToVideo = nodes_hunyuan.NODE_CLASS_MAPPINGS["TextEncodeHunyuanVideo_ImageToVideo"]()
HunyuanImageToVideo = nodes_hunyuan.NODE_CLASS_MAPPINGS["HunyuanImageToVideo"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAEDecodeTiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
SaveAnimatedWEBP = nodes_images.NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors", "default")[0]
    clip = DualCLIPLoader.load_clip("clip_l.safetensors", "llava_llama3_fp8_scaled.safetensors", "hunyuan_video")[0]
    vae = VAELoader.load_vae("hunyuan_video_vae_bf16.safetensors")[0]
    clip_vision = CLIPVisionLoader.load_clip("llava_llama3_vision.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def images_to_mp4(images, output_path, fps=24):
    try:
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            if img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            frames.append(img)
        temp_files = [f"temp_{i:04d}.png" for i in range(len(frames))]
        for i, frame in enumerate(frames):
            success = cv2.imwrite(temp_files[i], frame[:, :, ::-1])
            if not success:
                raise ValueError(f"Failed to write {temp_files[i]}")
        if not os.path.exists(temp_files[0]):
            raise FileNotFoundError("Temporary PNG files were not created")
        stream = ffmpeg.input('temp_%04d.png', framerate=fps)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        for temp_file in temp_files:
            os.remove(temp_file)
    except Exception as e:
        print(f"Error: {e}")

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        input_image = values['input_image']
        input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
        crop = values['crop'] # center
        prompt = values['prompt'] # a cute anime girl with massive fennec ears and a big fluffy tail wearing a maid outfit walking forward
        image_interleave = values['image_interleave'] # 4
        width = values['width'] # 720
        height = values['height'] # 720
        length = values['length'] # 57
        batch_size = values['batch_size'] # 1
        guidance_type = values['guidance_type'] # v2 (replace)
        guidance = values['guidance'] # 6.0
        shift = values['shift'] # 7.0
        sampler_name = values['sampler_name'] # euler
        scheduler = values['scheduler'] # simple
        steps = values['steps'] # 20
        denoise = values['denoise'] # 1.0
        noise_seed = values['noise_seed'] # 1.0
        if noise_seed == 0:
            random.seed(int(time.time()))
            noise_seed = random.randint(0, 18446744073709551615)
        tile_size = values['tile_size'] # 256
        overlap = values['overlap'] # 64
        temporal_size = values['temporal_size'] # 64
        temporal_overlap = values['temporal_overlap'] # 8
        fps = values['fps'] # 24
        filename_prefix = values['filename_prefix'] # hunyuan_i2v
        lossless = values['lossless'] # false
        quality = values['quality'] # 90
        method = values['method'] # default
        is_output_webp = values['is_output_webp'] # false
        cloudinary_folder = values['cloudinary_folder'] # generated_videos

        input_image = LoadImage.load_image(input_image)[0]
        clip_vision_output = CLIPVisionEncode.encode(clip_vision, input_image, crop)[0]
        positive = TextEncodeHunyuanVideo_ImageToVideo.encode(clip, clip_vision_output, prompt, image_interleave)[0]
        conditioning, latent_image = HunyuanImageToVideo.encode(positive, vae, width, height, length, batch_size, guidance_type, start_image=input_image)
        conditioning = FluxGuidance.append(conditioning, guidance)[0]
        unet_sd3 = ModelSamplingSD3.patch(unet, shift, multiplier=1000)[0]
        guider = BasicGuider.get_guider(unet_sd3, conditioning)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, denoise)[0]
        noise = RandomNoise.get_noise(noise_seed)[0]
        samples = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)[0]
        images = VAEDecodeTiled.decode(vae, samples, tile_size, overlap=overlap, temporal_size=temporal_size, temporal_overlap=temporal_overlap)[0]
        
        if(is_output_webp):
            out_video = SaveAnimatedWEBP.save_images(images, fps, filename_prefix, lossless, quality, method, num_frames=0, prompt=None, extra_pnginfo=None)["ui"]["images"][0]["filename"]
            video_path = f"/content/ComfyUI/output/{out_video}"
            result_url = cloudinary.uploader.upload(video_path, folder=cloudinary_folder, overwrite=True)["secure_url"]
        else:
            images_to_mp4(images, f"/content/ComfyUI/output/{filename_prefix}.mp4", fps)
            video_path = f"/content/ComfyUI/output/{filename_prefix}.mp4"
            result_url = cloudinary.uploader.upload(video_path, folder=cloudinary_folder, resource_type="video", overwrite=True)["secure_url"]
        
        return {"status": "DONE", "result": result_url}
    except Exception as e:
        return {"status": "ERROR", "result": str(e)}

runpod.serverless.start({"handler": generate})
